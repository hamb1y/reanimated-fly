"""Foreleg grooming: replayed RECORDED kinematics, plus the pure logic that lets the brain decide WHEN.

Motion (recorded, not authored): app/data/groom_clip.npz holds the two front legs of a tethered real fly grooming its head
(NeuroMechFly v1 repository, DeepFly3D pose estimation + inverse kinematics, Apache 2.0, see app/data/NOTICE.md and
tools/make_groom_clip.py). Only a short loopable segment is kept. Nothing here invents leg motion; what this module adds is:

- the axis/convention mapping from the v1 model to the flygym 2.1.1 body (to_v2_qpos),
- a looping player with a short drift correction at the wrap (Clip.pose),
- a blend so the legs go from the walking controller's pose into the replay and back without a snap (Groomer),
- a hysteresis trigger that turns the brain's groom_adn1 firing rate into a yes/no (GroomTrigger).

Convention mapping (derived from the v1 SDF, neuromechfly_noLimits.sdf, and the flygym 2.1.1 model, measured in
physics/README.md): the recorded angles are in the v1 'global' convention, the same axes on both sides. v1 nests the
coxa as roll (z) -> yaw (x) -> pitch (y), so the coxa rotation is R = Rz(roll) Rx(yaw) Ry(pitch). flygym 2.1.1 uses
yaw (x) -> pitch (y) -> roll (z), R = Rx Ry Rz, and its right-leg yaw/roll axes are negated (anatomical, symmetric
convention). So the recorded rotation is re-expressed as Rx(A) Ry(B) Rz(C); the left leg gets (A, B, C), the right leg
(-A, B, -C), and the femur roll is negated on the right. Pitch joints are used as they are.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("physics.groom")

CLIP_PATH = Path(__file__).parent / "data" / "groom_clip.npz"
KEYS = ("ThC_yaw", "ThC_pitch", "ThC_roll", "CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch")
# recorded joint -> flygym (parent link, child link, axis); {s} is 'lf' or 'rf'
DOF_SPEC = {
    "ThC_yaw": ("c_thorax", "{s}_coxa", "yaw"),
    "ThC_pitch": ("c_thorax", "{s}_coxa", "pitch"),
    "ThC_roll": ("c_thorax", "{s}_coxa", "roll"),
    "CTr_pitch": ("{s}_coxa", "{s}_trochanterfemur", "pitch"),
    "CTr_roll": ("{s}_coxa", "{s}_trochanterfemur", "roll"),
    "FTi_pitch": ("{s}_trochanterfemur", "{s}_tibia", "pitch"),
    "TiTa_pitch": ("{s}_tibia", "{s}_tarsus1", "pitch"),
}
FORELEGS = ("lf", "rf")
LEGS_ALL = ("lf", "lm", "lh", "rf", "rm", "rh")

BLEND_IN_S = 0.8  # legs from walking-controller pose to the replay
BLEND_OUT_S = 0.8
SETTLE_S = 0.4  # walk command frozen this long before the forelegs leave the ground (the other legs settle to stance)
LOOP_FIX_S = 0.4  # the last 0.4 s of a loop are bent by the end-to-start jump so the wrap has no snap
# AUTHORED stance support, NOT recorded: with the forelegs off the ground the default 6-leg stance tips nose-down and the body
# collapses onto the ground (measured: thorax height 1.03 -> about 0.0 mm, pitch up to 50 deg). Moving the middle legs forward
# (coxa pitch -0.6 rad) and the hind legs back (+0.3 rad) while grooming gives a steady 4-leg stance (thorax 0.64 mm, pitch ~0).
SUPPORT: dict[tuple[str, str], float] = {
    ("lm", "ThC_pitch"): -0.6, ("rm", "ThC_pitch"): -0.6,
    ("lh", "ThC_pitch"): 0.3, ("rh", "ThC_pitch"): 0.3,
}  # (leg, joint key) -> offset in rad, blended in with the replay weight
ADHESION_OFF_W = 0.05  # foreleg adhesion is off as soon as the blend passes this


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def coxa_v1_matrix(yaw, pitch, roll):
    """Coxa rotation of the v1 (recording) model: roll (z), then yaw (x), then pitch (y). Angles in radians (arrays)."""
    return _rz(roll) @ _rx(yaw) @ _ry(pitch)


def coxa_v2_matrix(a, b, c):
    """Coxa rotation of the flygym 2.1.1 left leg: yaw (x), pitch (y), roll (z) = Rx(a) Ry(b) Rz(c)."""
    return _rx(a) @ _ry(b) @ _rz(c)


def to_v2_qpos(raw: np.ndarray, side: str) -> np.ndarray:
    """raw: (n, 7) recorded angles in KEYS order (radians). Returns (n, 7) joint angles for the flygym 2.1.1 leg."""
    raw = np.asarray(raw, dtype=float)
    m = coxa_v1_matrix(raw[:, 0], raw[:, 1], raw[:, 2])
    b = np.arcsin(np.clip(m[:, 0, 2], -1.0, 1.0))
    a = np.arctan2(-m[:, 1, 2], m[:, 2, 2])
    c = np.arctan2(-m[:, 0, 1], m[:, 0, 0])
    out = raw.copy()
    out[:, 0], out[:, 1], out[:, 2] = np.unwrap(a), b, np.unwrap(c)
    if side == "r":
        out[:, 0] *= -1.0
        out[:, 2] *= -1.0
        out[:, 4] *= -1.0
    return out


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class Clip:
    """Looping foreleg replay in flygym joint angles. q has shape (n, 2 legs [lf, rf], 7 [KEYS])."""

    q: np.ndarray
    fps: float
    meta: dict

    @classmethod
    def load(cls, path: Path = CLIP_PATH) -> "Clip":
        z = np.load(path, allow_pickle=False)
        raw = z["angles"].astype(float)  # (n, 2, 7)
        q = np.stack([to_v2_qpos(raw[:, 0], "l"), to_v2_qpos(raw[:, 1], "r")], 1)
        meta = {k: z[k].item() if z[k].ndim == 0 else z[k].tolist() for k in z.files if k != "angles"}
        return cls(q=q, fps=float(z["fps"]), meta=meta)

    @property
    def period(self) -> float:
        """Loop length in seconds (the last sample is made equal to the first by the wrap fix)."""
        return (len(self.q) - 1) / self.fps

    def pose(self, t: float) -> np.ndarray:
        """(2, 7) target joint angles at clip time t (looped). Linear interpolation of the 100 Hz samples."""
        p = self.period
        u = t % p
        x = u * self.fps
        i = min(int(x), len(self.q) - 2)
        f = x - i
        out = (1.0 - f) * self.q[i] + f * self.q[i + 1]
        fix = smoothstep((u - (p - LOOP_FIX_S)) / LOOP_FIX_S)
        if fix > 0.0:
            out = out - fix * (self.q[-1] - self.q[0])
        return out


class Groomer:
    """Blends the replayed foreleg motion over the walking controller's action. Used from the sim thread only,
    except request(), which only sets a flag."""

    def __init__(self, dof_order, clip: Clip | None = None) -> None:
        from flygym.anatomy import LEGS

        self.clip = clip if clip is not None else Clip.load()
        self.idx = np.zeros((2, 7), dtype=int)  # position in the action vector of each recorded DOF, per foreleg
        names = {
            (d.parent.name, d.child.name, d.axis.value): i for i, d in enumerate(dof_order)
        }
        for li, leg in enumerate(FORELEGS):
            for ki, key in enumerate(KEYS):
                par, chi, ax = DOF_SPEC[key]
                self.idx[li, ki] = names[(par.format(s=leg), chi.format(s=leg), ax)]
        self.adh_idx = np.array([LEGS.index(leg) for leg in FORELEGS])
        # AUTHORED (not recorded) stance support for the other four legs, see SUPPORT. (action index, delta rad)
        self.support: list[tuple[int, float]] = []
        for (leg, key), delta in SUPPORT.items():
            par, chi, ax = DOF_SPEC[key]
            self.support.append((names[(par.format(s=leg), chi.format(s=leg), ax)], float(delta)))
        self.requested = False
        self.weight = 0.0  # 0 = walking controller owns the legs, 1 = replay owns the forelegs
        self.phase = 0.0  # position in the loop, seconds
        self._since_request = 0.0
        self.cycles = 0

    # --- thread-safe surface (flag writes only) ------------------------------------------
    def request(self, on: bool) -> None:
        self.requested = bool(on)

    @property
    def engaged(self) -> bool:
        """True while the walk command must stay frozen (requested, or still blending out)."""
        return self.requested or self.weight > 0.0

    @property
    def grooming(self) -> bool:
        return self.weight > ADHESION_OFF_W

    def reset(self) -> None:
        self.requested = False
        self.weight = 0.0
        self.phase = 0.0
        self._since_request = 0.0

    # --- sim thread -----------------------------------------------------------------------
    def apply(self, angles: np.ndarray, adhesion, dt: float):
        """Return (angles, adhesion) with the forelegs blended toward the replay. Inputs are not modified."""
        if self.requested:
            self._since_request += dt
            if self._since_request >= SETTLE_S:
                self.weight = min(1.0, self.weight + dt / BLEND_IN_S)
        else:
            self._since_request = 0.0
            self.weight = max(0.0, self.weight - dt / BLEND_OUT_S)
        if self.weight <= 0.0:
            self.phase = 0.0
            return angles, adhesion
        before = self.phase
        self.phase += dt
        if int(self.phase / self.clip.period) != int(before / self.clip.period):
            self.cycles += 1
        s = smoothstep(self.weight)
        target = self.clip.pose(self.phase)
        out = np.array(angles, dtype=float)
        for li in range(2):
            ix = self.idx[li]
            out[ix] = (1.0 - s) * out[ix] + s * target[li]
        for ix, delta in self.support:
            out[ix] += s * delta
        adh = None
        if adhesion is not None:
            adh = np.asarray(adhesion, dtype=float).copy()
            adh[self.adh_idx] = adh[self.adh_idx] * (0.0 if self.weight > ADHESION_OFF_W else 1.0)
        return out, adh

    def status(self) -> dict:
        return {
            "requested": self.requested,
            "weight": round(self.weight, 3),
            "grooming": self.grooming,
            "phase_s": round(self.phase, 3),
            "cycles": self.cycles,
        }


class GroomTrigger:
    """Hysteresis on the brain's groom_adn1 firing rate (Hz, mean per neuron over the brain's 100 ms window).

    The rate is smoothed with a first-order filter (tau_s) because two neurons give a quantised, spiky rate (one spike in
    the window = 5 Hz). On when the smoothed rate stays >= on_hz for on_s; off when it stays <= off_hz for off_s.
    A missing rate (no brain, stale frames) counts as 0.
    """

    def __init__(self, on_hz: float = 2.0, off_hz: float = 0.7, on_s: float = 0.3, off_s: float = 0.8, tau_s: float = 0.4) -> None:
        self.on_hz, self.off_hz, self.on_s, self.off_s, self.tau_s = on_hz, off_hz, on_s, off_s, tau_s
        self.state = False
        self.ema = 0.0
        self._t_last: float | None = None
        self._t_cross: float | None = None

    def update(self, rate_hz: float | None, now: float) -> bool:
        r = 0.0 if rate_hz is None or not math.isfinite(rate_hz) else max(0.0, float(rate_hz))
        dt = 0.0 if self._t_last is None else max(0.0, now - self._t_last)
        self._t_last = now
        a = 1.0 - math.exp(-dt / self.tau_s)
        self.ema += a * (r - self.ema)
        want_cross = (self.ema >= self.on_hz) if not self.state else (self.ema <= self.off_hz)
        if want_cross:
            if self._t_cross is None:
                self._t_cross = now
            elif now - self._t_cross >= (self.off_s if self.state else self.on_s):
                self.state = not self.state
                self._t_cross = None
        else:
            self._t_cross = None
        return self.state

    def reset(self) -> None:
        self.state = False
        self.ema = 0.0
        self._t_last = None
        self._t_cross = None
