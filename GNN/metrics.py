"""
Layout quality metrics, penalties, and bonuses.
All functions accept plain Python objects (no torch tensors).
"""
import numpy as np
from typing import List, Dict, Tuple
from objects import PlacementObject, Road, Block


ROAD_ACCESS_DIST = 5.0      # max distance object edge → road centre-line (m)
BLOCK_MIN = 4
BLOCK_MAX = 6

# Loss weights
W_FIRE = 5.0
W_OVERLAP = 10.0
W_TECH = 1.0
W_ROAD = 6.0
W_BLOCK = 2.0
W_BOUNDS = 8.0
W_ORIENTATION = 1.5


# ──────────────────────────────────────────────
# Pairwise geometry helpers
# ──────────────────────────────────────────────

def _rect_rect_gap(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    """Signed gap between two axis-aligned rectangles (negative = overlap)."""
    dx = abs(ax - bx) - (aw + bw) / 2
    dy = abs(ay - by) - (ah + bh) / 2
    if dx >= 0 and dy >= 0:
        return np.hypot(dx, dy)
    elif dx >= 0:
        return dx
    elif dy >= 0:
        return dy
    else:
        return max(dx, dy)   # overlap amount (negative)


def _circle_circle_gap(ax, ay, ar, bx, by, br) -> float:
    return np.hypot(ax - bx, ay - by) - ar - br


def object_gap(a: PlacementObject, b: PlacementObject) -> float:
    """Edge-to-edge gap between two placed objects."""
    if a.shape == "circle" and b.shape == "circle":
        return _circle_circle_gap(a.x, a.y, a.half_w(), b.x, b.y, b.half_w())
    if a.shape == "rectangle" and b.shape == "rectangle":
        return _rect_rect_gap(a.x, a.y, a.width, a.height,
                               b.x, b.y, b.width, b.height)
    # mixed: circle vs rectangle — approximate
    if a.shape == "circle":
        a, b = b, a
    # a=rect, b=circle
    closest_x = np.clip(b.x, a.x - a.half_w(), a.x + a.half_w())
    closest_y = np.clip(b.y, a.y - a.half_h(), a.y + a.half_h())
    return np.hypot(b.x - closest_x, b.y - closest_y) - b.half_w()


def object_edge_to_road_edge(obj: PlacementObject, road: Road) -> float:
    """Signed distance from nearest object edge to nearest road edge.
    Negative means the object overlaps the road.
    """
    centre_to_road = road.distance_to_point(obj.x, obj.y)
    # Distance from object centre to road edge
    dist_to_road_edge = centre_to_road - road.width / 2
    # Subtract object half-extent in the direction perpendicular to the road
    if obj.shape == "circle":
        obj_half = obj.half_w()
    else:
        # Road direction vector → perpendicular is the normal
        rdx, rdy = road.direction()   # unit tangent
        # Normal to road (perpendicular direction)
        nx, ny = -rdy, rdx
        # Project object half-extents onto normal
        obj_half = abs(nx) * obj.half_w() + abs(ny) * obj.half_h()
    return dist_to_road_edge - obj_half


# Keep old name as alias used in road_access_score
def object_edge_to_road(obj: PlacementObject, road: Road) -> float:
    return object_edge_to_road_edge(obj, road)


# ──────────────────────────────────────────────
# Individual metric functions (return 0-100 score)
# ──────────────────────────────────────────────

def fire_safety_score(objects: List[PlacementObject]) -> Tuple[float, Dict]:
    """Score 0-100: 100 = all fire gaps satisfied."""
    violations = 0
    total_pairs = 0
    details = []
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = objects[i], objects[j]
            required = max(a.fire_distance, b.fire_distance)
            gap = object_gap(a, b)
            total_pairs += 1
            if gap < required:
                violations += 1
                details.append(f"{a.name}↔{b.name}: зазор {gap:.1f} < {required:.1f} м")
    score = 100.0 * (1 - violations / total_pairs) if total_pairs else 100.0
    return score, {"violations": violations, "total_pairs": total_pairs, "details": details[:5]}


def tech_proximity_score(objects: List[PlacementObject]) -> Tuple[float, Dict]:
    """Score 0-100: 100 = all connected pairs are adjacent."""
    id_map = {obj.id: obj for obj in objects}
    pairs, total_dist = 0, 0.0
    details = []
    for obj in objects:
        for cid in obj.connections:
            if cid in id_map and cid > obj.id:
                other = id_map[cid]
                d = object_gap(obj, other)
                total_dist += max(0.0, d)
                pairs += 1
                details.append(f"{obj.name}↔{other.name}: {d:.1f} м")
    if pairs == 0:
        return 100.0, {"pairs": 0, "avg_dist": 0.0}
    avg = total_dist / pairs
    # Score: 100 at 0 m gap, 0 at 50 m gap
    score = max(0.0, 100.0 * (1 - avg / 50.0))
    return score, {"pairs": pairs, "avg_dist": avg, "details": details[:5]}


def road_access_score(objects: List[PlacementObject], roads: List[Road]) -> Tuple[float, Dict]:
    """Score 0-100: 100 = every object is within ROAD_ACCESS_DIST of a road."""
    if not roads:
        return 100.0, {"note": "Дорог нет"}
    ok = 0
    details = []
    for obj in objects:
        min_dist = min(object_edge_to_road_edge(obj, r) for r in roads)
        # 0..ROAD_ACCESS_DIST is the valid "near road but not on it" zone
        if 0 <= min_dist <= ROAD_ACCESS_DIST:
            ok += 1
        else:
            details.append(f"{obj.name}: {min_dist:.1f} м от дороги")
    score = 100.0 * ok / len(objects)
    return score, {"ok": ok, "total": len(objects), "violations": details[:5]}


def block_score(blocks: List[Block]) -> Tuple[float, Dict]:
    """Score 0-100: 100 = every non-empty block has 4-6 objects."""
    non_empty = [b for b in blocks if b.objects]
    if not non_empty:
        return 100.0, {"non_empty": 0}
    good = sum(1 for b in non_empty if BLOCK_MIN <= len(b.objects) <= BLOCK_MAX)
    score = 100.0 * good / len(non_empty)
    bad = [(b.id, len(b.objects)) for b in non_empty
           if not (BLOCK_MIN <= len(b.objects) <= BLOCK_MAX)]
    return score, {"non_empty": len(non_empty), "good": good, "bad_blocks": bad}


def orientation_score(objects: List[PlacementObject], roads: List[Road]) -> Tuple[float, Dict]:
    """Score: rectangles should have their long side facing the nearest road."""
    if not roads:
        return 100.0, {}
    rects = [o for o in objects if o.shape == "rectangle" and o.width != o.height]
    if not rects:
        return 100.0, {}
    ok = 0
    for obj in rects:
        # Find nearest road and its direction
        nearest = min(roads, key=lambda r: r.distance_to_point(obj.x, obj.y))
        rdx, rdy = nearest.direction()
        # Object long axis: if width > height, long axis is horizontal (1,0)
        long_is_horizontal = obj.width >= obj.height
        # Road parallel direction
        road_is_horizontal = abs(rdx) >= abs(rdy)
        # Long side should face road = long axis perpendicular to road direction
        # long axis ⊥ road direction → they should be parallel in angle terms
        if long_is_horizontal == road_is_horizontal:
            ok += 1
    score = 100.0 * ok / len(rects)
    return score, {"checked": len(rects), "ok": ok}


# ──────────────────────────────────────────────
# Combined loss for optimisation (torch-free, numpy)
# ──────────────────────────────────────────────

def compute_layout_loss(objects: List[PlacementObject],
                        roads: List[Road],
                        blocks: List[Block],
                        bounds: Tuple[float, float, float, float]) -> float:
    """Lower is better. Used by scipy optimizer."""
    xmin, ymin, xmax, ymax = bounds
    loss = 0.0
    id_map = {obj.id: obj for obj in objects}
    n = len(objects)

    for i in range(n):
        a = objects[i]

        # Bounds penalty
        margin_x = max(0, xmin - (a.x - a.half_w())) + max(0, (a.x + a.half_w()) - xmax)
        margin_y = max(0, ymin - (a.y - a.half_h())) + max(0, (a.y + a.half_h()) - ymax)
        loss += W_BOUNDS * (margin_x ** 2 + margin_y ** 2)

        # Road access penalty
        if roads:
            dists = [object_edge_to_road_edge(a, r) for r in roads]
            min_dist = min(dists)
            # Too far from road edge
            excess = max(0.0, min_dist - ROAD_ACCESS_DIST)
            loss += W_ROAD * excess ** 2
            # ON the road (object overlaps road) — strong penalty
            if min_dist < 0:
                loss += W_ROAD * 4 * min_dist ** 2

        for j in range(i + 1, n):
            b = objects[j]
            gap = object_gap(a, b)

            # Overlap penalty
            if gap < 0:
                loss += W_OVERLAP * gap ** 2

            # Fire safety penalty
            required = max(a.fire_distance, b.fire_distance)
            if gap < required:
                loss += W_FIRE * (required - gap) ** 2

            # Tech proximity bonus (pull connected objects together)
            if b.id in a.connections or a.id in b.connections:
                loss += W_TECH * max(0.0, gap) ** 2

    # Orientation penalty
    for obj in objects:
        if obj.shape == "rectangle" and roads:
            nearest = min(roads, key=lambda r: r.distance_to_point(obj.x, obj.y))
            rdx, rdy = nearest.direction()
            long_is_h = obj.width >= obj.height
            road_is_h = abs(rdx) >= abs(rdy)
            if long_is_h != road_is_h:
                loss += W_ORIENTATION * 100

    return loss


# ──────────────────────────────────────────────
# Full metrics report
# ──────────────────────────────────────────────

def full_report(objects, roads, blocks):
    fs, fs_d = fire_safety_score(objects)
    tp, tp_d = tech_proximity_score(objects)
    ra, ra_d = road_access_score(objects, roads)
    bs, bs_d = block_score(blocks)
    ori, ori_d = orientation_score(objects, roads)

    overall = (fs * 0.35 + tp * 0.20 + ra * 0.25 + bs * 0.10 + ori * 0.10)

    return {
        "overall": overall,
        "fire_safety": {"score": fs, **fs_d},
        "tech_proximity": {"score": tp, **tp_d},
        "road_access": {"score": ra, **ra_d},
        "blocks": {"score": bs, **bs_d},
        "orientation": {"score": ori, **ori_d},
    }
