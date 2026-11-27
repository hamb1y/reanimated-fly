"""Fidelity levels: live switch keeps the fly walking; the runner's fallback steps down levels."""

import numpy as np
import pytest

from app.flysim import FIDELITY_LEVELS, FlySim
from app.runner import FALLBACK_WINDOWS, SimRunner


class _StubSim:
    def __init__(self, level):
        self.level = level

    def set_level(self, level):
        self.level = level


class _StubRunner:
    """Only the fallback logic of SimRunner, without starting a thread."""

    fallbacks = 0
    speed_factor = 0.5
    target_speed = 1.0

    def __init__(self, level):
        self.flysim = _StubSim(level)


def test_fallback_steps_down_after_slow_windows_only():
    r = _StubRunner(0)
    n = len(FIDELITY_LEVELS)
    # one slow window is not enough
    assert SimRunner._maybe_fallback(r, FALLBACK_WINDOWS - 1) == FALLBACK_WINDOWS - 1
    assert r.flysim.level == 0
    assert SimRunner._maybe_fallback(r, FALLBACK_WINDOWS) == 0
    assert r.flysim.level == 1 and r.fallbacks == 1
    # never below the coarsest level
    for _ in range(n + 2):
        SimRunner._maybe_fallback(r, FALLBACK_WINDOWS)
    assert r.flysim.level == n - 1


def test_set_level_keeps_walking():
    fs = FlySim(0)
    assert fs.timestep == pytest.approx(FIDELITY_LEVELS[0][0])
    fs.set_level(1)
    dt, k = FIDELITY_LEVELS[1]
    assert fs.timestep == pytest.approx(dt)
    assert fs.controller.timestep == pytest.approx(k * dt)
    assert fs.controller.cpg_network.timestep == pytest.approx(k * dt)
    z0 = fs.pose()["pos"][:, 2].copy()
    fs.advance(1.0, 0.0, 200)  # 0.4 s of walking at the coarser level
    thorax = fs.body_names.index("nmf/c_thorax")
    pose = fs.pose()
    assert pose["pos"][thorax, 2] > 0.5  # still standing
    assert np.isfinite(pose["pos"]).all() and not np.allclose(pose["pos"][:, 2], z0)
