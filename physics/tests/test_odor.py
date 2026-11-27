import math
import numpy as np
from app import odor as O


def _field(**kw):
    f = O.OdorField(**kw)
    f.set_sources("food", [(0.0, 0.0, 1.0)])
    for _ in range(20):
        f.step(0.5)
    return f


def test_plume_falls_off_smoothly_and_is_monotonic():
    f = _field()
    cs = [f.sample("food", r, 0.0) for r in (4, 20, 60, 110, 170, 260)]
    assert all(a > b > 0 for a, b in zip(cs, cs[1:]))
    assert cs[0] > 3 * cs[2] > 0  # steep near the source


def test_mass_balance_steady_state():
    f = O.OdorField()
    f.set_sources("food", [(0.0, 0.0, 1.0)])
    for _ in range(300):
        f.step(1.0)  # 1/k = 7 s, so 300 s is far past steady state
    total = f.c["food"].sum() * O.DX_MM**2
    assert math.isclose(total * f.k, O.SOURCE_Q, rel_tol=0.02)  # emission = decay at steady state


def test_gradient_points_to_source_and_antennae_see_it():
    f = _field()
    # fly at (-100, 0) heading +x (toward the source): both antennae close; heading +y (source to the right): right > left
    a = f.antennae("food", (-100.0, 0.0), 0.0)
    assert abs(a["left"] - a["right"]) / (a["left"] + a["right"]) < 0.05
    b = f.antennae("food", (-100.0, 0.0), math.pi / 2)  # facing +y; source is at her right (+x is to her right)
    assert b["right"] > b["left"]


def test_wind_carries_plume_downwind():
    f = O.OdorField(wind=(60.0, 0.0))
    f.set_sources("food", [(0.0, 0.0, 1.0)])
    for _ in range(20):
        f.step(0.5)
    assert f.sample("food", 100.0, 0.0) > 3 * f.sample("food", -100.0, 0.0)


def test_removing_source_clears_the_plume_and_species_are_separate():
    f = _field()
    assert f.sample("cva", 4.0, 0.0) < 1e-9
    f.set_sources("food", [])
    for _ in range(120):
        f.step(1.0)
    assert f.sample("food", 4.0, 0.0) < 1e-3


def test_orn_rate_saturates():
    assert O.orn_rate_hz(0.0) == 0.0
    assert 0 < O.orn_rate_hz(0.01) < 5 and O.orn_rate_hz(1.0) == O.ORN_MAX_HZ / 2 and O.orn_rate_hz(1e6) < O.ORN_MAX_HZ + 1e-9
