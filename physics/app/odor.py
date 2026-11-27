"""Odour: a diffusing, decaying, optionally wind-blown concentration field, and the two antennae that sample it.

Food and mates are SOURCES. Each tick the field is advanced by one implicit step of

    dc/dt = D laplace(c) - k c - u . grad(c) + S(x)

solved exactly in Fourier space on a periodic grid (so any D and dt are stable: no CFL limit, and the plume reaches its
steady shape in a few seconds of simulated time instead of minutes). D is an effective, turbulence-assisted air
diffusivity (indoor air mixes far faster than molecular diffusion, ~20 mm^2/s), chosen so that a source is smellable at
the scale of the counter. Its decay length is lambda = sqrt(D/k). Two species are tracked: 'food' and 'cva'.

Sampling: each antenna reads the field with bilinear interpolation at its position (thorax frame offset, rotated by the
heading). The olfactory receptor neuron rate is a saturating (Michaelis-Menten) function of the local concentration.
The fly cannot read a bearing: only the left/right difference and the mean level exist.
"""
from __future__ import annotations

import math

import numpy as np

SPECIES = ("food", "cva")
HALF_MM = 400.0  # grid covers [-HALF, HALF) mm in x and y (periodic; the wrap is far beyond the arena)
DX_MM = 4.0
N = int(2 * HALF_MM / DX_MM)
D_MM2_S = 1500.0  # effective diffusivity
K_DECAY_S = 0.15  # decay rate -> lambda = sqrt(D/k) = 100 mm; a plume lingers ~45 s after its source is gone
SOURCE_Q = 1000.0  # emission per source at full strength (concentration units * mm^2 / s)
FOOD_FULL = 100.0  # food amount at full strength
K_HALF = 1.0  # concentration giving half the maximum ORN rate
ORN_MAX_HZ = 200.0
ANTENNA = {"left": (1.0, 0.5), "right": (1.0, -0.5)}  # (forward, left) offset from the thorax centre, mm (head is ahead of it)
MAX_DT_S = 1.0


class OdorField:
    def __init__(self, d: float = D_MM2_S, k: float = K_DECAY_S, wind=(0.0, 0.0)) -> None:
        self.d, self.k = d, k
        self.wind = (float(wind[0]), float(wind[1]))
        self.c = {s: np.zeros((N, N)) for s in SPECIES}  # c[s][iy, ix]
        kx = 2 * np.pi * np.fft.fftfreq(N, DX_MM)
        self._kx, self._ky = np.meshgrid(kx, kx)  # [iy, ix]
        self._k2 = self._kx**2 + self._ky**2
        self.sources: dict[str, list[tuple[float, float, float]]] = {s: [] for s in SPECIES}

    def set_wind(self, ux: float, uy: float) -> None:
        self.wind = (float(ux), float(uy))

    def set_sources(self, species: str, sources) -> None:
        """sources: iterable of (x_mm, y_mm, strength 0..1)."""
        self.sources[species] = [(float(x), float(y), float(s)) for x, y, s in sources if s > 0]

    def _source_grid(self, species: str) -> np.ndarray:
        s = np.zeros((N, N))
        for x, y, w in self.sources[species]:
            fx, fy = (x + HALF_MM) / DX_MM - 0.5, (y + HALF_MM) / DX_MM - 0.5  # cell-centre coordinates
            ix, iy = math.floor(fx), math.floor(fy)
            ax, ay = fx - ix, fy - iy
            for dy, wy in ((0, 1 - ay), (1, ay)):  # spread over 4 cells (bilinear), so a source can sit anywhere
                for dx, wx in ((0, 1 - ax), (1, ax)):
                    s[(iy + dy) % N, (ix + dx) % N] += SOURCE_Q * w * wy * wx / DX_MM**2
        return s

    def step(self, dt: float) -> None:
        dt = min(max(float(dt), 0.0), MAX_DT_S)
        if dt <= 0.0:
            return
        ux, uy = self.wind
        denom = 1.0 / dt + self.k + self.d * self._k2 + 1j * (ux * self._kx + uy * self._ky)  # implicit Euler in Fourier space
        for sp in SPECIES:
            rhs = self.c[sp] / dt + self._source_grid(sp)
            self.c[sp] = np.maximum(np.fft.ifft2(np.fft.fft2(rhs) / denom).real, 0.0)

    def sample(self, species: str, x: float, y: float) -> float:
        f = self.c[species]
        fx, fy = (x + HALF_MM) / DX_MM - 0.5, (y + HALF_MM) / DX_MM - 0.5
        ix, iy = math.floor(fx), math.floor(fy)
        ax, ay = fx - ix, fy - iy
        i0, i1, j0, j1 = ix % N, (ix + 1) % N, iy % N, (iy + 1) % N
        return float((1 - ay) * ((1 - ax) * f[j0, i0] + ax * f[j0, i1]) + ay * ((1 - ax) * f[j1, i0] + ax * f[j1, i1]))

    def antennae(self, species: str, xy, heading: float) -> dict:
        """Concentration at the two antennae (fly at xy, heading in rad, +left)."""
        ch, sh = math.cos(heading), math.sin(heading)
        out = {}
        for side, (fwd, lat) in ANTENNA.items():
            out[side] = self.sample(species, xy[0] + fwd * ch - lat * sh, xy[1] + fwd * sh + lat * ch)
        return out

    def snapshot(self, species: str = "food", step: int = 4) -> dict:
        """Downsampled view for the UI: the central region with the grid cell size (mm)."""
        a, b = int(N * 0.25), int(N * 0.75)  # centre +-200 mm
        sub = self.c[species][a:b:step, a:b:step]
        return {"species": species, "cell_mm": DX_MM * step, "x0_mm": -HALF_MM + a * DX_MM, "y0_mm": -HALF_MM + a * DX_MM,
                "ny": int(sub.shape[0]), "nx": int(sub.shape[1]), "max": float(sub.max()),
                "c": np.round(sub, 4).ravel().tolist()}


def orn_rate_hz(c: float) -> float:
    """Receptor neuron input rate (Hz) for a local concentration."""
    return ORN_MAX_HZ * c / (c + K_HALF) if c > 0 else 0.0
