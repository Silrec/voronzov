from dataclasses import dataclass, field
from typing import List
import numpy as np


OBJECT_TYPES = ["Производство", "Склад", "Офис", "Энергетика", "Утилиты", "Другое"]
SHAPE_TYPES = ["rectangle", "circle"]


@dataclass
class PlacementObject:
    id: int
    name: str
    shape: str          # 'rectangle' or 'circle'
    width: float        # metres; diameter for circle
    height: float       # metres; same as width for circle
    object_type: str    # technological category
    fire_distance: float = 10.0   # min fire-safety gap to other objects (m)
    connections: List[int] = field(default_factory=list)  # IDs of tech-linked objects

    # Runtime placement state
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0   # degrees (rectangles only)
    block_id: int = -1

    def half_w(self) -> float:
        return self.width / 2

    def half_h(self) -> float:
        return self.height / 2

    def effective_radius(self) -> float:
        """Bounding circle radius for fast overlap checks."""
        if self.shape == "circle":
            return self.width / 2
        return np.hypot(self.width, self.height) / 2

    def corners(self):
        """Return 4 corners of rectangle in world coords (ignores rotation for grid snap)."""
        hw, hh = self.half_w(), self.half_h()
        return [
            (self.x - hw, self.y - hh),
            (self.x + hw, self.y - hh),
            (self.x + hw, self.y + hh),
            (self.x - hw, self.y + hh),
        ]


@dataclass
class Road:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 6.0  # road width in metres

    def length(self) -> float:
        return np.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def direction(self):
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        n = np.hypot(dx, dy) or 1
        return dx / n, dy / n

    def distance_to_point(self, px, py) -> float:
        """Perpendicular distance from point to road segment."""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        if dx == 0 and dy == 0:
            return np.hypot(px - self.x1, py - self.y1)
        t = ((px - self.x1) * dx + (py - self.y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx = self.x1 + t * dx
        cy = self.y1 + t * dy
        return np.hypot(px - cx, py - cy)


@dataclass
class Block:
    id: int
    x: float        # bottom-left corner
    y: float
    width: float
    height: float
    objects: List[int] = field(default_factory=list)

    def center(self):
        return self.x + self.width / 2, self.y + self.height / 2

    def contains(self, ox, oy, ow, oh):
        """True if object footprint fits inside block."""
        return (ox - ow / 2 >= self.x and ox + ow / 2 <= self.x + self.width and
                oy - oh / 2 >= self.y and oy + oh / 2 <= self.y + self.height)
