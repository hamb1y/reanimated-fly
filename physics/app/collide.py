"""MuJoCo collision for obstacles: static geoms for the kitchen props, a pool of mocap bodies for user-placed solids.

Contact filtering uses contype/conaffinity bit 2 (flygym itself uses explicit ground pairs and leaves every fly geom at
contype=conaffinity=0):
- obstacle geoms: contype 0, conaffinity 2;  selected fly geoms: contype 2, conaffinity 0.
  Fly-obstacle contacts happen ((2 & 2) != 0); fly-fly and obstacle-obstacle never do; the ground pairs are untouched.
- Fly geoms that collide: thorax, head, abdomen and coxa/femur/tibia of every leg. The tarsi do NOT touch obstacles:
  tarsus5 carries flygym's adhesion actuator, and an adhesive foot on a vertical wall would glue her to it (the walking
  controller cannot climb). So the feet can slip into a prop's footprint a little; the body cannot.
- Obstacle geoms are in group 4 (the web draws only mesh geoms, so they stay invisible there) and have priority 1, so
  their low friction is used: she slides along a wall instead of sticking.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .obstacles import Footprint

OBST_GROUP = 4
OBST_BIT = 2
POOL_SIZE = 12  # user-placed solids that can collide at once
PARK = (2000.0, 2000.0, -500.0)  # unused pool bodies sit here, far from everything
FRICTION = (0.2, 0.005, 0.0001)
FLY_COLLIDE_SUFFIXES = ("c_thorax", "c_head", "c_abdomen12", "c_abdomen3", "c_abdomen4", "c_abdomen5", "c_abdomen6",
                        "_coxa", "_trochanterfemur", "_tibia")


def _geom_kw(fp_or_r, h: float) -> dict:
    return dict(contype=0, conaffinity=OBST_BIT, group=OBST_GROUP, priority=1, friction=FRICTION,
                rgba=(0.9, 0.5, 0.2, 0.25))


def add_prop_geoms(spec: mujoco.MjSpec, props: list[Footprint]) -> list[str]:
    """Static world geoms, one per footprint. Boxes and cylinders stand on z = 0. Returns the geom names."""
    names = []
    for fp in props:
        name = f"prop/{fp.id}"
        kw = _geom_kw(fp, fp.h)
        if fp.shape == "circle":
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[fp.r, fp.h / 2, 0],
                                    pos=[fp.x, fp.y, fp.h / 2], **kw)
        else:
            c, s = np.cos(fp.yaw / 2), np.sin(fp.yaw / 2)
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[fp.hx, fp.hy, fp.h / 2],
                                    pos=[fp.x, fp.y, fp.h / 2], quat=[c, 0, 0, s], **kw)
        names.append(name)
    return names


def add_pool(spec: mujoco.MjSpec, n: int = POOL_SIZE) -> list[str]:
    """Mocap bodies with one unit cylinder each (radius and height are set at run time)."""
    names = []
    for i in range(n):
        b = spec.worldbody.add_body(name=f"obstpool{i}", mocap=True, pos=list(PARK))
        b.add_geom(name=f"obstpool{i}/geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[1.0, 1.0, 0], **_geom_kw(None, 2))
        names.append(f"obstpool{i}")
    return names


def enable_fly_contacts(m: mujoco.MjModel, fly_name: str) -> int:
    """Set the obstacle bit on the chosen fly geoms (compiled model). Returns how many."""
    n = 0
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(fly_name + "/") and name.endswith(FLY_COLLIDE_SUFFIXES):
            m.geom_contype[g] = OBST_BIT
            m.geom_conaffinity[g] = 0
            b = int(m.geom_bodyid[g])
            if hasattr(m, "body_contype"):  # MuJoCo >= 3.1 pre-filters bodies by the OR of their geoms' bits
                m.body_contype[b] |= OBST_BIT
            n += 1
    return n


class Pool:
    """Places user solids on the mocap pool. apply() must run on the sim thread (between mj_step calls)."""

    def __init__(self, m: mujoco.MjModel, n: int = POOL_SIZE) -> None:
        self.bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obstpool{i}") for i in range(n)]
        self.mocap = [int(m.body_mocapid[b]) for b in self.bodies]
        self.geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obstpool{i}/geom") for i in range(n)]
        self.wanted: list[Footprint] = []
        self.dropped = 0  # solids beyond the pool size (they are still sensed, but do not collide)

    def set(self, fps: list[Footprint]) -> None:
        self.wanted = list(fps)

    def apply(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        fps = self.wanted
        self.dropped = max(0, len(fps) - len(self.bodies))
        for i, (mid, g) in enumerate(zip(self.mocap, self.geoms)):
            if i < len(fps):
                fp = fps[i]
                m.geom_size[g, 0], m.geom_size[g, 1] = fp.r, fp.h / 2
                m.geom_rbound[g] = float(np.hypot(fp.r, fp.h / 2))
                m.geom_aabb[g] = [0, 0, 0, fp.r, fp.r, fp.h / 2]
                d.mocap_pos[mid] = [fp.x, fp.y, fp.h / 2]
            else:
                d.mocap_pos[mid] = PARK
