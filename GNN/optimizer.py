"""
Optimization pipeline:
1. GNN produces initial position hints.
2. scipy L-BFGS-B refines positions on the continuous loss.
3. Snap to discrete grid.
4. Assign objects to blocks.
"""
import numpy as np
import torch
from scipy.optimize import minimize
from typing import List, Tuple

from objects import PlacementObject, Road, Block
from gnn_model import LayoutGNN, build_node_features, build_edge_index
from metrics import compute_layout_loss, ROAD_ACCESS_DIST

GRID_SIZE = 1.0   # metres per grid cell


def _pack(objects: List[PlacementObject]) -> np.ndarray:
    return np.array([[o.x, o.y] for o in objects], dtype=np.float64).ravel()


def _unpack(vec: np.ndarray, objects: List[PlacementObject]):
    for i, obj in enumerate(objects):
        obj.x = float(vec[i * 2])
        obj.y = float(vec[i * 2 + 1])


def _loss_fn(vec, objects, roads, blocks, bounds):
    _unpack(vec, objects)
    return compute_layout_loss(objects, roads, blocks, bounds)


def _loss_and_grad(vec, objects, roads, blocks, bounds):
    eps = 0.5
    f0 = _loss_fn(vec, objects, roads, blocks, bounds)
    grad = np.zeros_like(vec)
    for i in range(len(vec)):
        vec[i] += eps
        fp = _loss_fn(vec, objects, roads, blocks, bounds)
        vec[i] -= eps
        grad[i] = (fp - f0) / eps
    _unpack(vec, objects)
    return f0, grad


def gnn_initial_placement(
    objects: List[PlacementObject],
    bounds: Tuple[float, float, float, float],
    type_map: dict,
    model: LayoutGNN,
) -> None:
    """Use GNN output to seed initial positions."""
    xmin, ymin, xmax, ymax = bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    W, H = xmax - xmin, ymax - ymin

    x_feat = build_node_features(objects, type_map)
    edge_idx = build_edge_index(objects)

    with torch.no_grad():
        out = model(x_feat, edge_idx).numpy()   # [N, 3]: (dx, dy, drot)

    # Spread objects initially in a grid, then apply GNN offsets
    n = len(objects)
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = max(1, int(np.ceil(n / cols)))
    step_x = W / (cols + 1)
    step_y = H / (rows + 1)

    for i, obj in enumerate(objects):
        col = i % cols
        row = i // cols
        base_x = xmin + step_x * (col + 1)
        base_y = ymin + step_y * (row + 1)
        # Scale GNN offsets to ±20% of field size
        dx = float(out[i, 0]) * W * 0.1
        dy = float(out[i, 1]) * H * 0.1
        obj.x = np.clip(base_x + dx, xmin + obj.half_w(), xmax - obj.half_w())
        obj.y = np.clip(base_y + dy, ymin + obj.half_h(), ymax - obj.half_h())
        obj.rotation = 0.0


def snap_to_grid(objects: List[PlacementObject], bounds):
    xmin, ymin, xmax, ymax = bounds
    for obj in objects:
        obj.x = round(obj.x / GRID_SIZE) * GRID_SIZE
        obj.y = round(obj.y / GRID_SIZE) * GRID_SIZE
        obj.x = np.clip(obj.x, xmin + obj.half_w(), xmax - obj.half_w())
        obj.y = np.clip(obj.y, ymin + obj.half_h(), ymax - obj.half_h())


def build_blocks(
    objects: List[PlacementObject],
    bounds: Tuple[float, float, float, float],
    block_size: float = 60.0,
) -> List[Block]:
    """Partition the field into square blocks and assign objects."""
    xmin, ymin, xmax, ymax = bounds
    W, H = xmax - xmin, ymax - ymin
    nx = max(1, int(np.ceil(W / block_size)))
    ny = max(1, int(np.ceil(H / block_size)))

    blocks: List[Block] = []
    bid = 0
    for iy in range(ny):
        for ix in range(nx):
            bx = xmin + ix * block_size
            by = ymin + iy * block_size
            bw = min(block_size, xmax - bx)
            bh = min(block_size, ymax - by)
            blocks.append(Block(id=bid, x=bx, y=by, width=bw, height=bh))
            bid += 1

    for obj in objects:
        for block in blocks:
            cx, cy = block.x + block.width / 2, block.y + block.height / 2
            if (block.x <= obj.x <= block.x + block.width and
                    block.y <= obj.y <= block.y + block.height):
                block.objects.append(obj.id)
                obj.block_id = block.id
                break

    return blocks


def orient_to_road(objects: List[PlacementObject], roads):
    """Rotate rectangles so their long side faces the nearest road."""
    if not roads:
        return
    for obj in objects:
        if obj.shape != "rectangle" or obj.width == obj.height:
            continue
        nearest = min(roads, key=lambda r: r.distance_to_point(obj.x, obj.y))
        rdx, rdy = nearest.direction()
        road_is_h = abs(rdx) >= abs(rdy)
        long_is_h = obj.width >= obj.height
        if long_is_h != road_is_h:
            obj.width, obj.height = obj.height, obj.width


def run_optimization(
    objects: List[PlacementObject],
    roads: List[Road],
    bounds: Tuple[float, float, float, float],
    type_map: dict,
    model: LayoutGNN,
    block_size: float = 60.0,
    max_iter: int = 300,
    progress_cb=None,
) -> List[Block]:
    if not objects:
        return []

    # Step 1: GNN seed
    gnn_initial_placement(objects, bounds, type_map, model)

    # Step 2: Scipy optimisation
    x0 = _pack(objects)

    def cb(xk):
        if progress_cb:
            loss = _loss_fn(xk.copy(), objects, roads, [], bounds)
            progress_cb(loss)

    result = minimize(
        _loss_and_grad,
        x0,
        args=(objects, roads, [], bounds),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": 1e-6},
        callback=cb,
    )
    _unpack(result.x, objects)

    # Step 3: Orient to roads
    orient_to_road(objects, roads)

    # Step 4: Snap to grid
    snap_to_grid(objects, bounds)

    # Step 5: Build blocks
    blocks = build_blocks(objects, bounds, block_size)

    return blocks
