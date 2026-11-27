import math

import pytest

from app import behavior as B
from app.world import WorldStore
from tests.test_smell import _frame


def test_spider_stalks_her_but_stops_short_and_pauses_after_a_drag():
    store = WorldStore()
    store.add("scary", (80.0, 0.0))
    b = B.Behavior(store)
    t = 0.0
    for _ in range(600):  # 40 s at 15 Hz, fly standing at the origin
        b.tick(_frame(t, 0.0, 0.0, 0.0), 0, 1.0)
        t += 1 / 15
    o = store.objects()[0]
    assert math.hypot(o.x, o.y) == pytest.approx(B.PRED_STOP, abs=0.5)
    store.move(o.id, [60.0, 0.0])  # the user drags it away
    b.tick(_frame(t, 0.0, 0.0, 0.0), 0, 1.0)
    t += 1 / 15
    b.tick(_frame(t, 0.0, 0.0, 0.0), 0, 1.0)
    assert store.objects()[0].x == pytest.approx(60.0)  # paused, not snapped back toward her
    for _ in range(int(B.PRED_PAUSE_S * 15) + 30):
        t += 1 / 15
        b.tick(_frame(t, 0.0, 0.0, 0.0), 0, 1.0)
    assert store.objects()[0].x < 60.0  # then it hunts again


def test_spider_ignores_her_when_far():
    store = WorldStore()
    store.add("scary", (200.0, 0.0))
    b = B.Behavior(store)
    for i in range(60):
        b.tick(_frame(i / 15, 0.0, 0.0, 0.0), 0, 1.0)
    assert store.objects()[0].x == 200.0
