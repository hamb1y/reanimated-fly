"""Obstacles: the kitchen props (app/data/kitchen_props.json) plus solid objects the user places, as 2D footprints.

Two consumers:
- flysim.py turns every footprint into a MuJoCo collision geom (real contact: she is physically blocked).
- Behavior senses them WITHOUT a map: `ObstacleSense` casts a fan of rays from her head on the ground plane and returns
  only the distance along each ray (cut at EYE_RANGE_MM), plus short "antenna" rays for touch. The steering filter
  (`avoid`) sees nothing but those distances and the goal bearing it is given.

What is real and what is authored:
- The footprints are authored simplifications of the drawn props (circles and boxes read off web/src/lib/kitchen.ts).
- The ray distances are computed analytically against those footprints. A real fly gets this kind of range cue from
  motion parallax / looming and antennal touch; here it is a geometric stand-in, not rendered vision and not a brain output.
- The steering rule (pick the free direction closest to the goal, keep the chosen side) is scripted, like the rest of
  behavior.py. MuJoCo contact is the physical backstop: if the rule fails, the body hits the geom instead of passing through.

Units: mm, radians, bearing positive = left (as in world.py).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

PROPS_PATH = Path(os.environ.get("KITCHEN_PROPS", Path(__file__).parent / "data" / "kitchen_props.json"))

SOLID_KINDS = {"block": 8.0, "scary": 5.0}  # user-placed kinds that collide and are sensed, with their footprint radius (mm)
SOLID_HEIGHT_MM = {"block": 16.0, "scary": 10.0}

EYE_RANGE_MM = 30.0  # rays see obstacles up to this distance
N_RAYS = 37  # -180..180 deg in 10 deg steps (flies see almost all round)
FOV_DEG = 180.0
ANTENNA_MM = 2.5  # antenna rays: anything this close in front counts as touch
HEAD_AHEAD_MM = 1.0  # ray origin: this far ahead of the thorax centre (the head)
CLEARANCE_MM = 2.5  # footprints are inflated by this for sensing (half the leg span), so a free ray means the body fits
LOOKAHEAD_MM = 14.0  # a direction is free if its ray is clear for min(goal distance, this)
FLIP_AFTER_TICKS = 120  # detouring this many filter calls (8 s at 15 Hz) without getting 1 mm closer to the goal: try the other side
FLIP_LOSS_MM = 12.0  # ... or as soon as the detour has taken her this much farther from the goal than her best
STOP_MM = 2.0  # nothing free and an obstacle closer than this ahead: forward drops to a crawl while turning
TURN_SCALE_DEG = 45.0  # same bearing rule as behavior.steer


@dataclass(frozen=True)
class Footprint:
    id: str
    shape: str  # 'circle' | 'box'
    x: float
    y: float
    r: float = 0.0  # circle radius
    hx: float = 0.0  # box half sizes in its own frame
    hy: float = 0.0
    yaw: float = 0.0  # box rotation (rad, counter-clockwise)
    h: float = 10.0  # height (MuJoCo geom only)
    source: str = "kitchen"  # 'kitchen' | 'user'

    def as_dict(self) -> dict:
        d = {"id": self.id, "shape": self.shape, "pos": [self.x, self.y], "h": self.h, "source": self.source}
        if self.shape == "circle":
            d["r"] = self.r
        else:
            d.update(half=[self.hx, self.hy], yaw=self.yaw)
        return d


def load_props(path: Path | str = PROPS_PATH) -> tuple[dict, list[Footprint]]:
    """-> (raw JSON, footprints). Raises ValueError on a malformed entry."""
    raw = json.loads(Path(path).read_text())
    out = []
    for p in raw["props"]:
        x, y = (float(v) for v in p["pos"])
        if p["shape"] == "circle":
            fp = Footprint(p["id"], "circle", x, y, r=float(p["r"]), h=float(p.get("h", 10)))
        elif p["shape"] == "box":
            hx, hy = (float(v) for v in p["half"])
            fp = Footprint(p["id"], "box", x, y, hx=hx, hy=hy, yaw=float(p.get("yaw", 0)), h=float(p.get("h", 10)))
        else:
            raise ValueError(f"prop {p.get('id')}: unknown shape {p['shape']!r}")
        if min(fp.r if fp.shape == "circle" else min(fp.hx, fp.hy), fp.h) <= 0:
            raise ValueError(f"prop {fp.id}: sizes must be positive")
        out.append(fp)
    return raw, out


def user_footprints(objects) -> list[Footprint]:
    """Solid user objects (world.WorldObject) as footprints. Food and mates are not solid."""
    return [Footprint(f"obj{o.id}", "circle", o.x, o.y, r=SOLID_KINDS[o.kind], h=SOLID_HEIGHT_MM[o.kind], source="user")
            for o in objects if o.kind in SOLID_KINDS]


def ray_hit(fp: Footprint, ox: float, oy: float, dx: float, dy: float, inflate: float = 0.0) -> float:
    """Distance along the unit ray (ox,oy)+t(dx,dy), t >= 0, to the inflated footprint; 0 if the origin is inside; inf if missed."""
    if fp.shape == "circle":
        r = fp.r + inflate
        cx, cy = ox - fp.x, oy - fp.y
        c = cx * cx + cy * cy - r * r
        b = cx * dx + cy * dy
        if c <= 0:  # inside (the clearance zone): blocked toward the centre, free away from it
            return 0.0 if b < 0 else math.inf
        disc = b * b - c
        if b >= 0 or disc < 0:
            return math.inf
        return -b - math.sqrt(disc)
    # box: slab test in the box frame (corners not rounded by the inflation: slightly conservative)
    cs, sn = math.cos(fp.yaw), math.sin(fp.yaw)
    px, py = ox - fp.x, oy - fp.y
    lx, ly = cs * px + sn * py, -sn * px + cs * py
    ldx, ldy = cs * dx + sn * dy, -sn * dx + cs * dy
    hx, hy = fp.hx + inflate, fp.hy + inflate
    if abs(lx) <= hx and abs(ly) <= hy:  # inside: blocked unless heading out through the nearest face
        if hx - abs(lx) < hy - abs(ly):
            out = ldx * (1 if lx >= 0 else -1)
        else:
            out = ldy * (1 if ly >= 0 else -1)
        return math.inf if out > 0 else 0.0
    t0, t1 = 0.0, math.inf
    for p, d, h in ((lx, ldx, hx), (ly, ldy, hy)):
        if abs(d) < 1e-12:
            if abs(p) > h:
                return math.inf
            continue
        a, b = (-h - p) / d, (h - p) / d
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return math.inf
    return t0


def ray_angles() -> list[float]:
    return [math.radians(-FOV_DEG + i * (2 * FOV_DEG / (N_RAYS - 1))) for i in range(N_RAYS)]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class ObstacleSense:
    """Range along a fan of rays from the head, nothing else. Footprints are looked up per call (props + user solids)."""

    def __init__(self, props: list[Footprint]) -> None:
        self.props = list(props)
        self.angles = ray_angles()

    def cast(self, xy, heading: float, extra: list[Footprint] = (), exclude: set[str] = frozenset()) -> dict:
        hx, hy = xy[0] + HEAD_AHEAD_MM * math.cos(heading), xy[1] + HEAD_AHEAD_MM * math.sin(heading)
        fps = [f for f in (*self.props, *extra) if f.id not in exclude]
        # cheap cull: only footprints whose bounding circle comes within range of the head
        near = []
        for f in fps:
            br = f.r if f.shape == "circle" else math.hypot(f.hx, f.hy)
            if math.hypot(f.x - hx, f.y - hy) - br - CLEARANCE_MM <= EYE_RANGE_MM:
                near.append(f)
        dist = []
        for a in self.angles:
            ang = heading + a
            dx, dy = math.cos(ang), math.sin(ang)
            d = min((ray_hit(f, hx, hy, dx, dy, CLEARANCE_MM) for f in near), default=math.inf)
            dist.append(d if d <= EYE_RANGE_MM else math.inf)
        touch = {"left": False, "right": False}
        for side, a in (("left", math.radians(30)), ("right", math.radians(-30))):
            dx, dy = math.cos(heading + a), math.sin(heading + a)
            touch[side] = any(ray_hit(f, hx, hy, dx, dy) <= ANTENNA_MM for f in near)
        return {"angles": self.angles, "dist": dist, "touch": touch}


def _ray_at(rays: dict, bearing: float) -> float:
    """Ray distance in a bearing direction: the minimum of the two neighbouring rays (conservative)."""
    angs, ds = rays["angles"], rays["dist"]
    step = angs[1] - angs[0]
    i = (bearing - angs[0]) / step
    lo = max(0, min(len(ds) - 1, math.floor(i)))
    hi = max(0, min(len(ds) - 1, math.ceil(i)))
    return min(ds[lo], ds[hi])


class Avoider:
    """Steering filter: keeps the desired direction when it is clear, otherwise the nearest clear direction (side is sticky)."""

    def __init__(self) -> None:
        self.side = 0  # +1 = going round on the left, -1 = right, 0 = not detouring
        self._ticks = 0
        self._best = math.inf

    def reset(self) -> None:
        self.side = 0
        self._ticks = 0
        self._best = math.inf

    def filter(self, rays: dict, forward: float, turn: float, goal_bearing: float | None = None,
               goal_dist: float = math.inf) -> tuple[float, float, dict]:
        """-> (forward, turn, info). goal_bearing None: the desired bearing is taken from the turn (turn * 45 deg)."""
        want = goal_bearing if goal_bearing is not None else math.radians(turn * TURN_SCALE_DEG)
        need = min(goal_dist, LOOKAHEAD_MM)
        info = {"blocked": False, "detour_deg": 0.0, "side": 0, "front_mm": _finite(_ray_at(rays, 0.0)),
                "touch": dict(rays.get("touch", {}))}
        if forward <= 1e-6 or _ray_at(rays, want) >= need:
            self.reset()
            return forward, turn, info
        info["blocked"] = True
        # bug-style: once a side is chosen she keeps it (follows the obstacle) until the goal direction is clear again;
        # if that makes no progress toward a goal for FLIP_AFTER_TICKS, she tries the other side
        self._ticks += 1
        if goal_dist < self._best - 1.0:
            self._best, self._ticks = goal_dist, 0
        if self.side and math.isfinite(goal_dist) and (self._ticks > FLIP_AFTER_TICKS or goal_dist > self._best + FLIP_LOSS_MM):
            self.side, self._ticks, self._best = -self.side, 0, goal_dist
        cands = []
        for a, d in zip(rays["angles"], rays["dist"]):
            if d >= need:
                s = 1 if _wrap(a - want) > 0 else -1
                cands.append((abs(_wrap(a - want)), a, s))
        same = [c for c in cands if not self.side or c[2] == self.side]
        cands = same or cands
        if cands:
            _, a, s = min(cands)
            self.side = s
            new_turn = max(-1.0, min(1.0, math.degrees(a) / TURN_SCALE_DEG))
            info.update(detour_deg=round(math.degrees(_wrap(a - want)), 1), side=s)
            front = _ray_at(rays, 0.0)
            if front < STOP_MM:  # nose on the obstacle: turn on the spot-ish
                forward = min(forward, 0.25)
            return forward, new_turn, info
        # nothing clear within the look-ahead: turn toward the most open ray, crawl
        i = max(range(len(rays["dist"])), key=lambda k: (rays["dist"][k], -abs(rays["angles"][k])))
        a = rays["angles"][i]
        info.update(detour_deg=round(math.degrees(_wrap(a - want)), 1), side=0)
        return min(forward, 0.25), max(-1.0, min(1.0, math.degrees(a) / TURN_SCALE_DEG)), info


def _finite(v: float):
    return None if math.isinf(v) else round(v, 1)
