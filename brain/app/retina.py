"""Photoreceptor input ("the eyes"): ommatidium lattice, photoreceptor map and transduction.

The lattice is the same hexagonal lattice that physics/app/vision.py renders and web/src/lib/eyes.ts draws:
4.3 deg spacing inside a 58 deg radius around each eye's optical axis -> 661 ommatidia per eye, ordered row by row
(r = -20..20, c = -20..20). Lattice coordinates (a, b) in degrees: b = up; a = toward the FRONT for the left eye and
toward the BACK for the right eye (camera 'right' of an eye looking 50 deg outward).

retina.json (built by tools/make_retinotopy.py from the FlyWire annotations + connectome) assigns each photoreceptor
(R1-6, R7, R8) to one ommatidium of its own eye. Transduction (README "Vision"): per eye, an adaptation level A
(running mean luminance of that eye, tau ADAPT_TAU_S); each photoreceptor fires at
    rate = RATE_MAX_HZ * clip((A - I) / A * GAIN, 0, 1)
i.e. it signals DARKENING relative to the adapted mean (an OFF / contrast-decrement code). Why OFF and not
luminance: in this connectome the photoreceptor synapses are mostly predicted excitatory (acetylcholine; the
neurotransmitter classifier has no histamine class), so a spiking R cell EXCITES L1/L2/L3. Real L1/L2 depolarise
when light DIMS (histamine is inhibitory). Encoding 'darker than adapted' as R spikes makes L-cell excitation mean
what it means in the fly. A bright uniform scene therefore gives no input (silent baseline).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SPACING_DEG = 4.3
RADIUS_DEG = 58.0
N_ROWS = 20  # r, c in -N_ROWS..N_ROWS
RATE_MAX_HZ = 200.0
GAIN = 2.0  # contrast 0.5 below the adapted mean already gives the full rate
ADAPT_TAU_S = 1.0
ADAPT_MODE = "local"
MIN_CONTRAST = 0.01
EYE_CONVENTION = "groups"
SAMPLE_K = 3
ACCEPT_SIGMA_DEG = 2.2  # Gaussian acceptance, FWHM ~5 deg (Drosophila: ~4.5-5 deg)
RETINA_JSON = Path(__file__).with_name("retina.json")


def lattice() -> np.ndarray:
    """(661, 2) float64 lattice coordinates (a, b) in degrees, identical to eyes.ts lattice()."""
    out = []
    dy = SPACING_DEG * 0.866
    for r in range(-N_ROWS, N_ROWS + 1):
        b = r * dy
        for c in range(-N_ROWS, N_ROWS + 1):
            a = c * SPACING_DEG + (SPACING_DEG / 2 if r & 1 else 0.0)
            if math.hypot(a, b) > RADIUS_DEG:
                continue
            out.append((a, b))
    return np.asarray(out, dtype=np.float64)


N_OMM = len(lattice())


def load_map(ids: np.ndarray, path: Path | None = None, convention: str = EYE_CONVENTION) -> dict:
    """{'left'|'right' (BODY eye): {'idx': int64[k] tensor indices, 'ab': float[k, 2] direction in that eye's lattice
    coordinates (deg), 'type': str[k]}} plus 'meta'. Photoreceptors whose root id is not in this connectome are dropped.

    convention 'groups' (default): the body's left eye uses the photoreceptors the FlyWire annotation calls 'right'.
    This matches groups.json, whose left/right (turn_left = rojas DNa01/DNa02 '_left', looming_left, lplc2_left) are
    the annotation's 'right' neurons; the reference names predate FlyWire's left/right correction. A consistent mirror
    of inputs and outputs does not change behaviour in a (near) symmetric brain, a mixed one would.
    convention 'annotation': body left eye = annotation 'left'."""
    if convention not in ("groups", "annotation"):
        raise ValueError("convention must be 'groups' or 'annotation'")
    raw = json.load(open(path or RETINA_JSON))
    order = np.argsort(ids)
    s = ids[order]
    out = {"meta": {**raw.get("meta", {}), "convention": convention}}
    for eye in ("left", "right"):
        anat = eye if convention == "annotation" else ("right" if eye == "left" else "left")
        e = raw["eyes"][anat]
        hv = np.asarray(e["hv_deg"], dtype=np.float64).reshape(-1, 2)
        ab = np.c_[hv[:, 0] if eye == "left" else -hv[:, 0], hv[:, 1]]  # a = front for the left eye, back for the right
        rid = np.asarray(e["root_id"], dtype=np.int64)
        pos = np.searchsorted(s, rid)
        pos[pos >= len(s)] = 0
        ok = s[pos] == rid
        out[eye] = {
            "idx": order[pos[ok]].astype(np.int64),
            "ab": ab[ok],
            "anat_side": anat,
            "type": np.asarray(e["type"])[ok],
        }
    return out


def sampling(ab: np.ndarray, k: int = SAMPLE_K, sigma: float = ACCEPT_SIGMA_DEG):
    """Each photoreceptor sees a Gaussian-weighted mean of its k nearest lattice ommatidia (acceptance angle).
    -> (nbr int64[m, k], wt float64[m, k]) with rows summing to 1."""
    lat = lattice()
    d = np.linalg.norm(ab[:, None, :] - lat[None], axis=2)
    nbr = np.argsort(d, 1)[:, :k]
    dd = np.take_along_axis(d, nbr, 1)
    wt = np.exp(-0.5 * (dd / sigma) ** 2) + 1e-12
    return nbr, wt / wt.sum(1, keepdims=True)


class Transducer:
    """Ommatidium luminance (0..1 per eye) -> photoreceptor Poisson rates (Hz), with light adaptation.

    mode 'global': adaptation level = running mean luminance of the whole eye (a static dark object keeps driving).
    mode 'local' : adaptation level = running mean of each photoreceptor's own input (static scenes fade out; only
                   changes drive). Time constant tau_s for both.
    """

    def __init__(self, rmap: dict, rate_max=RATE_MAX_HZ, gain=GAIN, tau_s=ADAPT_TAU_S, mode=ADAPT_MODE,
                 subsample=1.0, types=None, seed=0):
        if mode not in ("global", "local"):
            raise ValueError("mode must be 'global' or 'local'")
        self.rate_max, self.gain, self.tau_s, self.mode = float(rate_max), float(gain), float(tau_s), mode
        self.eyes = {}
        rng = np.random.default_rng(seed)
        for eye in ("left", "right"):
            m = rmap[eye]
            keep = np.ones(len(m["idx"]), bool)
            if types is not None:
                keep &= np.isin(m["type"], list(types))
            if subsample < 1.0:
                keep &= rng.random(len(m["idx"])) < subsample
            nbr, wt = sampling(m["ab"][keep])
            self.eyes[eye] = (m["idx"][keep], nbr, wt)
        self.idx = np.concatenate([self.eyes[e][0] for e in ("left", "right")])
        self.adapt = {"left": None, "right": None}

    def reset(self):
        self.adapt = {"left": None, "right": None}

    def sample(self, lum: dict) -> dict:
        """Per-photoreceptor luminance {'left': float[m], 'right': ...}."""
        out = {}
        for eye in ("left", "right"):
            v = np.clip(np.asarray(lum[eye], dtype=np.float64), 0.0, 1.0)
            if v.shape != (N_OMM,):
                raise ValueError(f"{eye}: expected {N_OMM} ommatidium values, got {v.shape}")
            _, nbr, wt = self.eyes[eye]
            out[eye] = (v[nbr] * wt).sum(1)
        return out

    def __call__(self, lum: dict, dt_s: float) -> np.ndarray:
        """-> rates (Hz) aligned with self.idx."""
        a_k = 1.0 - math.exp(-max(dt_s, 0.0) / self.tau_s)
        out = []
        for eye, x in self.sample(lum).items():
            ref = x if self.mode == "local" else float(x.mean())
            a = self.adapt[eye]
            if a is None:
                a = np.copy(ref) if self.mode == "local" else ref
            else:
                a = a + a_k * (ref - a)
            self.adapt[eye] = a
            c = (a - x) / np.maximum(a, 0.02) * self.gain
            c[c < MIN_CONTRAST] = 0.0  # dead zone: rounding-level differences must not drive neurons
            out.append(self.rate_max * np.clip(c, 0.0, 1.0))
        return np.concatenate(out)
