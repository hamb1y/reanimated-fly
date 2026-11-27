"""World objects, personality and sensing geometry (ADDENDUM C, physics side).

Units follow GET /model: mm, MuJoCo z-up. Object positions are [x, y] on the ground plane.
Bearing is the angle from the fly's heading to an object, in radians here and degrees in the
API. Positive bearing = object on the fly's LEFT (counter-clockwise seen from above), the same
sign as the motor command's turn (positive = left).

Sensing is purely geometric (no vision): distance, bearing, and for scary objects the apparent
angular size and its growth (looming). Nothing in this module talks to the brain.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

KINDS = ("food", "mate", "scary", "block")  # block: a solid obstacle the user places (app/obstacles.py)
PERSONALITY_DEFAULTS = {"hunger": 0.8, "fear": 0.9, "lust": 0.9, "receptivity": 0.9}
FOOD_AMOUNT = 100.0  # eating units; food disappears at 0
RADIUS_MM = {"food": 4.0, "mate": 3.0, "scary": 5.0, "block": 8.0}  # object radius for the looming geometry
MAX_ABS_POS_MM = 450.0  # the ground plane is 1000 mm square; keep objects on it


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading_from_quat(q) -> float:
    """Yaw (rad) of the body's +x axis from a MuJoCo wxyz quaternion. +x is the walking direction."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def sense(fly_xy, heading: float, obj_xy) -> tuple[float, float]:
    """Distance (mm) and bearing (rad, positive = left of the heading) from the fly to an object."""
    dx = float(obj_xy[0]) - float(fly_xy[0])
    dy = float(obj_xy[1]) - float(fly_xy[1])
    return math.hypot(dx, dy), wrap_pi(math.atan2(dy, dx) - heading)


def angular_size(radius: float, dist: float) -> float:
    """Apparent angular diameter-ish size (rad) of a sphere of this radius at this distance."""
    return 2.0 * math.atan2(radius, max(dist, 1e-6))


def falloff(dist: float, near: float, far: float) -> float:
    """1 at or inside `near`, linear to 0 at `far`, 0 beyond."""
    if far <= near:
        return 1.0 if dist <= near else 0.0
    return clamp((far - dist) / (far - near), 0.0, 1.0)


def clean_pos(pos) -> tuple[float, float]:
    """Validate an [x, y] position in mm. Raises ValueError."""
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        raise ValueError("pos must be [x, y]")
    x, y = (float(v) for v in pos)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("pos must be finite")
    if abs(x) > MAX_ABS_POS_MM or abs(y) > MAX_ABS_POS_MM:
        raise ValueError(f"pos must be within +-{MAX_ABS_POS_MM:g} mm of the origin")
    return x, y


@dataclass
class WorldObject:
    id: int
    kind: str
    x: float
    y: float
    amount: float | None = None  # food only

    def as_dict(self) -> dict:
        d = {"id": self.id, "kind": self.kind, "pos": [round(self.x, 2), round(self.y, 2)]}
        if self.kind == "food":
            d["amount"] = round(self.amount, 2)
        return d


class WorldStore:
    """Objects, personality and the published state. Single-threaded: used from the asyncio loop only.

    `version` changes on every mutation; `ws_part()` is the cached JSON the WS stream appends.
    """

    def __init__(self) -> None:
        self._objects: dict[int, WorldObject] = {}
        self._next_id = 1
        self.personality: dict[str, float] = dict(PERSONALITY_DEFAULTS)
        self.state: dict = _idle_state()
        self.version = 0
        self._ws_cache: tuple[int, str] | None = None

    def _bump(self) -> None:
        self.version += 1

    def add(self, kind: str, pos) -> WorldObject:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        x, y = clean_pos(pos)
        obj = WorldObject(self._next_id, kind, x, y, FOOD_AMOUNT if kind == "food" else None)
        self._objects[obj.id] = obj
        self._next_id += 1
        self._bump()
        return obj

    def move(self, oid: int, pos) -> WorldObject | None:
        obj = self._objects.get(oid)
        if obj is None:
            return None
        obj.x, obj.y = clean_pos(pos)
        self._bump()
        return obj

    def get(self, oid: int) -> WorldObject | None:
        return self._objects.get(oid)

    def delete(self, oid: int) -> bool:
        if self._objects.pop(oid, None) is None:
            return False
        self._bump()
        return True

    def clear(self) -> int:
        n = len(self._objects)
        self._objects.clear()
        self._bump()
        return n

    def objects(self) -> list[WorldObject]:
        return [self._objects[k] for k in sorted(self._objects)]

    def __len__(self) -> int:
        return len(self._objects)

    def consume(self, oid: int, units: float) -> float | None:
        """Take `units` from a food object. Removes it at 0. Returns what is left, None if gone."""
        obj = self._objects.get(oid)
        if obj is None or obj.kind != "food":
            return None
        obj.amount = max(0.0, obj.amount - units)
        if obj.amount <= 1e-9:
            del self._objects[oid]
            self._bump()
            return 0.0
        self._bump()
        return obj.amount

    def set_personality(self, **values: float) -> dict:
        changed = False
        for k, v in values.items():
            if k in PERSONALITY_DEFAULTS and v is not None and self.personality.get(k) != float(v):
                self.personality[k] = float(v)
                changed = True
        if changed:
            self._bump()
        return dict(self.personality)

    def set_state(self, state: dict) -> None:
        if state != self.state:
            self.state = state
            self._bump()

    def snapshot(self) -> dict:
        """GET /world body."""
        return {
            "arena": {"half_x": ARENA_HALF_X, "half_y": ARENA_HALF_Y},
            "objects": [o.as_dict() for o in self.objects()],
            "personality": dict(self.personality),
            "state": dict(self.state),
        }

    def ws_part(self) -> str:
        """JSON object text '{"objects":...,"state":...}' appended to each WS pose frame. Cached per version."""
        if self._ws_cache is None or self._ws_cache[0] != self.version:
            body = {"objects": [o.as_dict() for o in self.objects()], "state": self.state}
            self._ws_cache = (self.version, json.dumps(body, separators=(",", ":")))
        return self._ws_cache[1]


ARENA_HALF_X = 285.0  # mm: the explorer may use the whole counter (600 x 400); the props collide and are sensed (app/obstacles.py)
ARENA_HALF_Y = 185.0


def _idle_state() -> dict:
    return {"mode": "idle", "hunger_level": 0.0, "nearest": None, "drives": {}}
