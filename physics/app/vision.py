"""What the fly's eyes see: an analytic ommatidial renderer (no OpenGL, no browser).

For each of the 661 ommatidia of each eye (the lattice of brain/app/retina.py and web/src/lib/eyes.ts: 4.3 deg hex
spacing inside 58 deg of the optical axis; axis 50 deg outward and 8 deg up), rays are cast from the eye centre (the
eye geoms of the MuJoCo model, in the thorax frame) against the LOGICAL world: the world objects as spheres with the
radii of world.RADIUS_MM (food shrinks with amount as in the web), the counter plane z = 0, and a bright uniform
background (window/wall/sky) above the horizon. Her own body is ignored. Each ommatidium averages 7 rays (Gaussian
acceptance, ~5 deg FWHM). Output: luminance 0..1 per ommatidium, quantised to 0..255 (the values the web draws and,
with VISION_TO_BRAIN=1, the brain receives).

Lattice coordinates (a, b) in degrees: b = up; a = camera right of an eye looking outward, i.e. toward the FRONT for
the left eye and toward the BACK for the right eye.
"""
from __future__ import annotations

import math

import numpy as np

from .world import RADIUS_MM

SPACING_DEG = 4.3
RADIUS_DEG = 58.0
AXIS_YAW_DEG = 50.0
AXIS_PITCH_DEG = 8.0
N_ROWS = 20
LUM = {"sky": 1.0, "floor": 0.5, "scary": 0.05, "mate": 0.25, "food": 0.7}
FOOD_FULL = 100.0
EYE_FALLBACK = {"left": (0.264, 0.308, 0.001), "right": (0.264, -0.308, 0.001)}  # thorax frame, mm (NMF l_eye/r_eye)
SUB_DEG = 2.2  # sub-ray offset (acceptance)
SUB_W = (0.4,) + (0.1,) * 6


def lattice() -> np.ndarray:
    """(661, 2) lattice (a, b) in degrees; same order as brain/app/retina.py and eyes.ts."""
    out = []
    dy = SPACING_DEG * 0.866
    for r in range(-N_ROWS, N_ROWS + 1):
        b = r * dy
        for c in range(-N_ROWS, N_ROWS + 1):
            a = c * SPACING_DEG + (SPACING_DEG / 2 if r & 1 else 0.0)
            if math.hypot(a, b) <= RADIUS_DEG:
                out.append((a, b))
    return np.asarray(out, dtype=np.float64)


N_OMM = len(lattice())


def quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def eye_dirs(side: str, ab: np.ndarray) -> np.ndarray:
    """Unit ray directions in the thorax frame (x forward, y left, z up) for lattice points ab (deg)."""
    s = 1.0 if side == "left" else -1.0
    p, y = math.radians(AXIS_PITCH_DEG), math.radians(AXIS_YAW_DEG)
    d = np.array([math.cos(p) * math.cos(y), s * math.cos(p) * math.sin(y), math.sin(p)])
    right = np.cross(d, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, d)
    th = np.radians(np.hypot(ab[:, 0], ab[:, 1]))
    ph = np.arctan2(ab[:, 1], ab[:, 0])
    v = np.cos(th)[:, None] * d + np.sin(th)[:, None] * (np.cos(ph)[:, None] * right + np.sin(ph)[:, None] * up)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def eye_centres(scene: dict | None) -> dict:
    """Eye centres in the thorax frame from the model's eye geoms (names *l_eye / *r_eye on the thorax body)."""
    out = dict(EYE_FALLBACK)
    if scene:
        for g in scene.get("geoms", []):
            n = str(g.get("name", "")).lower()
            if n.endswith("l_eye"):
                out["left"] = tuple(g["pos"])
            elif n.endswith("r_eye"):
                out["right"] = tuple(g["pos"])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


class EyeRenderer:
    def __init__(self, scene: dict | None = None) -> None:
        self.lat = lattice()
        offs = [(0.0, 0.0)] + [(SUB_DEG * math.cos(k * math.pi / 3), SUB_DEG * math.sin(k * math.pi / 3)) for k in range(6)]
        self.nsub = len(offs)
        self.w = np.asarray(SUB_W)
        sub = (self.lat[:, None, :] + np.asarray(offs)[None]).reshape(-1, 2)  # (661*7, 2)
        self.dirs = {s: eye_dirs(s, sub) for s in ("left", "right")}
        self.centre = eye_centres(scene)

    def render(self, thorax_pos, thorax_quat, objects) -> dict:
        """-> {'left': float[661], 'right': float[661]} luminance 0..1. objects: iterable of world.WorldObject."""
        R = quat_mat(thorax_quat)
        tp = np.asarray(thorax_pos, dtype=float)
        sph = []
        for o in objects:
            r = RADIUS_MM.get(o.kind, 3.0)
            if o.kind == "food" and o.amount is not None:
                r *= 0.3 + 0.7 * max(0.0, min(1.0, o.amount / FOOD_FULL))
            sph.append((o.x, o.y, r, r, LUM.get(o.kind, 0.3)))
        out = {}
        for side in ("left", "right"):
            org = tp + R @ self.centre[side]
            D = self.dirs[side] @ R.T  # world-frame directions
            lum = np.where(D[:, 2] < 0, LUM["floor"], LUM["sky"])
            tmin = np.where(D[:, 2] < 0, -org[2] / np.minimum(D[:, 2], -1e-9), np.inf)
            for cx, cy, cz, r, val in sph:
                oc = org - np.array([cx, cy, cz])
                b = D @ oc
                c = oc @ oc - r * r
                disc = b * b - c
                hit = disc >= 0
                t = np.where(hit, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
                inside = c < 0  # eye inside the sphere: everything is the object
                if inside:
                    t = np.zeros_like(t)
                closer = (t > 0) & (t < tmin) | (inside & np.ones_like(hit))
                lum = np.where(closer, val, lum)
                tmin = np.where(closer, t, tmin)
            out[side] = (lum.reshape(-1, self.nsub) * self.w).sum(1)
        return out


def quantise(v: np.ndarray) -> list[int]:
    return np.clip(np.rint(np.asarray(v) * 255.0), 0, 255).astype(int).tolist()
