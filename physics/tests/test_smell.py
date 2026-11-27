"""Long-range food: no hard cutoff. A hungry fly beyond FOOD_RANGE follows the odour gradient to the food."""
import math

from app.behavior import Behavior
from app.world import WorldStore


def _frame(t, x, y, th):
    return {"t": t, "pos": {0: (x, y, 0.5)}, "quat": {0: (math.cos(th / 2), 0.0, 0.0, math.sin(th / 2))}}


def _walk(food_xy, start, heading, secs=240.0, speed=12.0, dt=1 / 15):
    store = WorldStore()
    store.add("food", food_xy)
    b = Behavior(store, hunger_start=0.8)
    x, y, th, t = start[0], start[1], heading, 0.0
    modes, d0 = set(), math.hypot(food_xy[0] - x, food_xy[1] - y)
    while t < secs:
        r = b.tick(_frame(t, x, y, th), 0, 1.0)
        modes.add(r["mode"])
        th += r["turn"] * 2.0 * dt  # rad/s of turn at full command
        x += math.cos(th) * speed * r["forward"] * dt
        y += math.sin(th) * speed * r["forward"] * dt
        t += dt
        if math.hypot(food_xy[0] - x, food_xy[1] - y) < 3.0:
            return True, t, modes, d0
    return False, t, modes, d0


def test_she_finds_food_from_far_away_from_any_heading():
    for start, th in (((-100.0, 0.0), 0.0), ((-100.0, 0.0), math.pi), ((100.0, 60.0), math.pi / 2), ((0.0, -90.0), -math.pi / 2)):
        ok, t, modes, d0 = _walk((0.0, 0.0), start, th)
        assert ok, (start, th, t, modes)
        assert "smell" in modes


def test_no_smell_tracking_when_not_hungry():
    store = WorldStore()
    store.add("food", (0.0, 0.0))
    b = Behavior(store, hunger_start=0.0)
    for i in range(60):
        r = b.tick(_frame(i / 15, -150.0, 0.0, 0.0), 0, 1.0)
    assert r["mode"] == "explore"
    assert r["smell"]["food"]["left"] > 0  # she still smells it; she just does not care
