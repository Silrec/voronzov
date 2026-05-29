"""
Graph Neural Network for spatial object layout.
Uses message-passing over the technological-connection graph.
Outputs (x, y, rotation) offsets that are refined by the optimizer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class MessagePassingLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : [N, in_dim]
        edge_index : [2, E]  (src, dst)
        """
        N = x.size(0)
        if edge_index.numel() == 0:
            agg = torch.zeros(N, x.size(1), device=x.device)
        else:
            src, dst = edge_index[0], edge_index[1]
            msgs = x[src]                              # [E, in_dim]
            agg = torch.zeros(N, x.size(1), device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
        combined = torch.cat([x, agg], dim=1)          # [N, in_dim*2]
        return self.node_mlp(combined)


class LayoutGNN(nn.Module):
    """
    Node features:
        0: width (m)
        1: height (m)
        2: shape (0=rect, 1=circle)
        3: object_type (int encoded)
        4: fire_distance (m)
    Output per node: (dx, dy, drot) — position corrections in normalised space.
    """
    NODE_FEATURES = 5

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(self.NODE_FEATURES, hidden_dim)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),   # (dx, dy, drot)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
        return self.output_head(h)   # [N, 3]


def build_node_features(objects, type_map) -> torch.Tensor:
    rows = []
    for obj in objects:
        shape_enc = 1.0 if obj.shape == "circle" else 0.0
        type_enc = float(type_map.get(obj.object_type, 0))
        rows.append([
            obj.width,
            obj.height,
            shape_enc,
            type_enc,
            obj.fire_distance,
        ])
    return torch.tensor(rows, dtype=torch.float32)


def build_edge_index(objects) -> torch.Tensor:
    id_to_idx = {obj.id: i for i, obj in enumerate(objects)}
    edges = []
    for i, obj in enumerate(objects):
        for cid in obj.connections:
            if cid in id_to_idx:
                j = id_to_idx[cid]
                edges.append([i, j])
                edges.append([j, i])
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
