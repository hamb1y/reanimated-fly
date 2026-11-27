"""Sensing math, arbitration and eating (pure functions and Behavior on synthetic pose frames)."""

import math

import numpy as np
import pytest

from app import behavior as B
from app.world import WorldStore, angular_size, falloff, heading_from_quat, sense, wrap_pi

PERS = {"hunger": 0.8, "fear": 0.9, "lust": 0.9, "receptivity": 0.9}


def S(id, kind, dist, bearing_deg=0.0, growth=0.0, amount=None):
    return B.Sensed(id, kind, dist, math.radians(bearing_deg), growth, amount)


# --- sensing math -------------------------------------------------------------------------

def test_heading_from_quat():
    assert heading_from_quat([1, 0, 0, 0]) == pytest.approx(0.0)
    h = math.radians(90)
    assert heading_from_quat([math.cos(h / 2), 0, 0, math.sin(h / 2)]) == pytest.approx(h)


def test_sense_distance_and_bearing_sign():
    d, b = sense((0, 0), 0.0, (0, 10))
    assert d == pytest.approx(10) and b == pytest.approx(math.pi / 2)  # +y is left of +x
    d, b = sense((0, 0), 0.0, (0, -10))
    assert b == pytest.approx(-math.pi / 2)  # right
    d, b = sense((0, 0), 0.0, (-10, 0))
    assert d == pytest.approx(10) and abs(b) == pytest.approx(math.pi)
    # heading turned left 90 deg: an object on +y is now straight ahead
    _, b = sense((0, 0), math.pi / 2, (0, 10))
    assert b == pytest.approx(0.0, abs=1e-12)
    # fly position is subtracted
    d, b = sense((5, 5), 0.0, (5, 15))
    assert d == pytest.approx(10) and b == pytest.approx(math.pi / 2)


def test_wrap_and_falloff_and_angular_size():
    assert abs(abs(wrap_pi(3 * math.pi)) - math.pi) == pytest.approx(0.0, abs=1e-9)
    assert wrap_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert falloff(0, 3, 60) == 1.0 and falloff(60, 3, 60) == 0.0 and falloff(31.5, 3, 60) == pytest.approx(0.5)
    sizes = [angular_size(5, d) for d in (5, 10, 20, 40, 80)]
    assert sizes == sorted(sizes, reverse=True)


# --- arbitration -------------------------------------------------------------------------

def test_wander_is_bounded_and_edge_steers_home():
    from app.behavior import WANDER_TURN, edge_weight, wander
    assert all(abs(wander(i * 0.37)) <= WANDER_TURN + 1e-9 for i in range(400))
    from app.world import ARENA_HALF_X
    assert edge_weight(0.0, 0.0) == 0.0 and edge_weight(ARENA_HALF_X, 0.0) == 1.0 and 0.0 < edge_weight(ARENA_HALF_X - 20.0, 0.0) < 1.0


def test_nothing_sensed_is_explore():
    p = B.plan([], PERS, 0.5)
    assert p.mode == "explore" and p.drives == {} and p.forward > 0


def test_fear_overrides_lust_and_hunger():
    sensed = [S(1, "scary", 30, 0, growth=0.4), S(2, "mate", 5, 10), S(3, "food", 2, 0, amount=100)]
    p = B.plan(sensed, PERS, 0.9)
    assert p.mode == "flee"
    assert set(p.drives) <= {"looming", "lplc2", "escape_gf"}
    assert "sugar_grn" not in p.drives and "feed_mn9" not in p.drives


def test_lust_overrides_hunger_and_low_fear_is_ignored():
    sensed = [S(2, "mate", 20, 0), S(3, "food", 2, 0, amount=100), S(1, "scary", 20, 0)]
    p = B.plan(sensed, dict(PERS, fear=0.1), 0.9)  # fear below FEAR_MIN: scary ignored
    assert p.mode == "chase"
    assert "feed_mn9" not in p.drives


def test_hunger_only_when_hungry_enough():
    food = [S(3, "food", 40, 0, amount=100)]
    assert B.plan(food, PERS, 0.9).mode == "eat"
    p = B.plan(food, PERS, 0.05)  # 0.8 * 0.05 = 0.04 < HUNGER_MIN: full, ignores food
    assert p.mode == "explore"
    assert B.plan(food, dict(PERS, hunger=0.0), 0.9).mode == "explore"


def test_food_sugar_full_rate_and_feed_only_when_touching():
    far = B.plan([S(3, "food", 40, 0, amount=100)], PERS, 1.0)
    assert far.drives.get("sugar_grn", 0) > 0 and "feed_mn9" not in far.drives
    assert far.forward > 0 and far.eat_target is None
    near = B.plan([S(3, "food", 2, 0, amount=100)], PERS, 1.0)
    assert near.eat_target == 3 and near.forward == 0
    assert near.drives["feed_mn9"] == pytest.approx(B.FEED_HZ * 0.8)
    assert near.drives["sugar_grn"] == pytest.approx(B.SUGAR_HZ * 0.8)


def test_mate_uses_half_rate_sugar_and_never_feed():
    p = B.plan([S(2, "mate", 1.0, 0)], PERS, 0.5)  # within MATE_CLOSE: falloff 1
    assert p.drives == {"sugar_grn": pytest.approx(0.5 * B.SUGAR_HZ * 0.9)}
    assert "feed_mn9" not in p.drives


def test_mate_approach_turns_toward_and_slows_when_receptive():
    far = B.plan([S(2, "mate", 50, 30)], PERS, 0.5)
    assert far.mode == "chase" and far.turn == pytest.approx(30 / B.TURN_SCALE_DEG)
    assert far.forward == pytest.approx(B.APPROACH_FWD * 0.5)  # no brain: floor gain 0.5
    close_hi = B.plan([S(2, "mate", 5, 0)], dict(PERS, receptivity=0.9), 0.5)
    assert close_hi.forward == pytest.approx(B.APPROACH_FWD * 0.1 * 0.5)
    stop = B.plan([S(2, "mate", 5, 0)], dict(PERS, receptivity=1.0), 0.5)
    assert stop.forward == pytest.approx(0.0)


def test_low_receptivity_walks_away_from_mate_in_range():
    for d in (5, 50):  # close or not, a non-receptive female walks away from a male in range
        p = B.plan([S(2, "mate", d, 0)], dict(PERS, receptivity=0.0), 0.5)
        assert p.mode == "flee"
        assert abs(p.turn) == pytest.approx(1.0)  # turned fully to face away (bearing 0 -> 180)
        assert p.forward > 0
    assert B.plan([S(2, "mate", 90, 0)], dict(PERS, receptivity=0.0), 0.5).mode == "explore"  # out of range


def test_escape_gf_only_when_very_close_and_looming_only_when_approaching():
    far = B.plan([S(1, "scary", 30, 0, growth=0.5)], PERS, 0.5)
    assert far.drives["looming"] == pytest.approx(B.LOOM_HZ * 0.9)
    assert far.drives["lplc2"] == pytest.approx(B.LPLC2_HZ * 0.9)
    assert "escape_gf" not in far.drives
    close = B.plan([S(1, "scary", 5, 0, growth=0.5)], PERS, 0.5)
    assert close.drives["escape_gf"] == pytest.approx(B.ESCAPE_HZ * 0.9)
    receding = B.plan([S(1, "scary", 30, 0, growth=-0.5)], PERS, 0.5)
    assert "looming" not in receding.drives and receding.mode == "flee"


def test_flee_turns_away_from_scary():
    ahead = B.plan([S(1, "scary", 30, 0)], PERS, 0.5)
    assert abs(ahead.turn) == pytest.approx(1.0)  # dead ahead: either way is "away"
    scary_left = B.plan([S(1, "scary", 30, 30)], PERS, 0.5)
    assert scary_left.turn < 0  # object on the left: turn right (away)


def test_brain_gain_scales_forward_only():
    base = B.plan([S(3, "food", 40, 0, amount=100)], PERS, 1.0, brain_forward=None)
    full = B.plan([S(3, "food", 40, 0, amount=100)], PERS, 1.0, brain_forward=1.0)
    none = B.plan([S(3, "food", 40, 0, amount=100)], PERS, 1.0, brain_forward=0.0)
    assert none.forward == pytest.approx(B.APPROACH_FWD * B.BRAIN_FLOOR)
    assert full.forward == pytest.approx(B.APPROACH_FWD)
    assert base.forward == pytest.approx(none.forward)
    assert full.turn == none.turn  # the bearing rule is scripted, not brain
    assert full.drives == none.drives


def test_flee_latch_holds_until_well_outside_range():
    gone = B.plan([S(1, "scary", B.SCARY_RANGE + 5, 0)], PERS, 0.5)
    assert gone.mode != "flee" and gone.flee_id is None
    held = B.plan([S(1, "scary", B.SCARY_RANGE + 5, 0)], PERS, 0.5, flee_latch=1)
    assert held.mode == "flee" and held.flee_id == 1
    other = B.plan([S(1, "scary", B.SCARY_RANGE + 5, 0)], PERS, 0.5, flee_latch=7)
    assert other.mode != "flee"
    past = B.plan([S(1, "scary", B.SCARY_RANGE * B.FLEE_HOLD + 1, 0)], PERS, 0.5, flee_latch=1)
    assert past.mode != "flee"


def test_steer_is_clamped_proportional():
    assert B.steer(math.radians(22.5)) == pytest.approx(0.5)
    assert B.steer(math.radians(-300)) == pytest.approx(-1.0)


# --- behaviour over pose frames -------------------------------------------------------------

def _frame(t, xy=(0.0, 0.0), n=3, thorax=1):
    pos = np.zeros((n, 3))
    pos[thorax] = [xy[0], xy[1], 1.0]
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return {"t": t, "pos": pos, "quat": quat}


def test_eating_consumes_only_when_nearly_stopped_and_in_reach():
    store = WorldStore()
    food = store.add("food", (2.0, 0.0))  # 2 mm ahead: inside REACH
    b = B.Behavior(store)
    b.tick(_frame(0.0), 1)
    r = b.tick(_frame(1.0), 1)
    assert r["mode"] == "eat" and r["eat_target"] == food.id and r["eaten"] == pytest.approx(B.EAT_UNITS_PER_S)
    assert store.get(food.id).amount == pytest.approx(100 - B.EAT_UNITS_PER_S)
    # hunger: +0.01 per sim second, -0.004 per unit eaten
    assert b.hunger_level == pytest.approx(0.6 + 0.01 - 10 * B.HUNGER_PER_UNIT, abs=1e-9)


def test_moving_fly_does_not_eat():
    store = WorldStore()
    food = store.add("food", (20.0, 0.0))
    b = B.Behavior(store)
    b.tick(_frame(0.0), 1)
    r = b.tick(_frame(1.0, xy=(20.0, 0.0)), 1)  # arrives at 20 mm/s: in reach but too fast to feed
    assert r["mode"] == "eat" and r["speed"] > B.EAT_MAX_SPEED
    assert r["eaten"] == 0.0
    assert store.get(food.id).amount == pytest.approx(100.0)


def test_food_disappears_at_zero():
    store = WorldStore()
    food = store.add("food", (1.0, 0.0))
    b = B.Behavior(store)  # hunger 0.6, personality 0.8: hungry enough to keep eating
    t = 0.0
    b.tick(_frame(t), 1)
    for _ in range(30):
        t += 1.0
        b.tick(_frame(t), 1)
        if store.get(food.id) is None:
            break
    assert store.get(food.id) is None  # 100 units at 10/s
    assert b.hunger_level == pytest.approx(0.6 + 10 * 0.01 - 100 * 0.004)
    for _ in range(90):  # the plume lingers ~45 s after the food is gone, then she wanders again
        t += 1.0
        r = b.tick(_frame(t), 1)
    assert r["mode"] == "explore"  # nothing left to sense: wander


def test_fly_gets_full_and_stops_eating():
    store = WorldStore()
    food = store.add("food", (1.0, 0.0))
    b = B.Behavior(store, hunger_start=0.2)
    b.tick(_frame(0.0), 1)
    modes = [b.tick(_frame(float(k)), 1) for k in range(1, 6)]
    # hunger 0.2 -> 0.17 -> 0.14 -> 0.11; at 0.12 the gain (0.8 * 0.12) is below HUNGER_MIN
    # at the threshold it stops, walks off to explore, and comes back once hunger creeps up again
    assert [m["mode"] for m in modes[:4]] == ["eat", "eat", "eat", "explore"]
    assert modes[3]["eaten"] == 0.0
    assert store.get(food.id).amount == pytest.approx(60.0)  # three full ticks, then one more after hunger rose


def test_looming_growth_from_approach():
    store = WorldStore()
    store.add("scary", (40.0, 0.0))
    b = B.Behavior(store)
    b.tick(_frame(0.0), 1)
    store.move(1, (25.0, 0.0))  # object comes closer
    r = b.tick(_frame(0.5), 1)
    assert r["mode"] == "flee" and r["drives"]["looming"] > 0
    store.move(1, (40.0, 0.0))  # receding: no looming drive
    r = b.tick(_frame(1.0), 1)
    assert "looming" not in r["drives"]


def test_looming_growth_is_smoothed_not_pulsed():
    store = WorldStore()
    store.add("scary", (40.0, 0.0))
    b = B.Behavior(store)
    b.tick(_frame(0.0), 1)
    store.move(1, (30.0, 0.0))
    r_move = b.tick(_frame(0.1), 1)
    assert r_move["drives"]["looming"] > 0
    # object stops: the drive persists briefly, then decays to nothing
    r_soon = b.tick(_frame(0.4), 1)
    assert 0 < r_soon["drives"].get("looming", 0) <= r_move["drives"]["looming"] * 1.001
    r_late = b.tick(_frame(3.0), 1)
    assert "looming" not in r_late["drives"]


def test_sim_reset_is_tolerated():
    store = WorldStore()
    store.add("food", (30.0, 0.0))
    b = B.Behavior(store)
    b.tick(_frame(5.0), 1)
    r = b.tick(_frame(0.0), 1)  # time went backwards (POST /reset)
    assert r["mode"] == "eat"
    assert r["eaten"] == 0.0


def test_store_objects_and_personality():
    s = WorldStore()
    a = s.add("food", (1, 2))
    m = s.add("mate", (3, 4))
    assert (a.id, m.id) == (1, 2) and s.get(a.id).amount == 100.0
    v = s.version
    assert s.move(m.id, (5, 6)) is not None
    assert s.version > v
    assert s.move(99, (0, 0)) is None
    with pytest.raises(ValueError):
        s.add("cake", (0, 0))
    with pytest.raises(ValueError):
        s.add("food", (1000, 0))
    assert s.delete(a.id) and not s.delete(a.id)
    assert s.set_personality(fear=0.2)["fear"] == 0.2
    assert '"objects":' in s.ws_part() and '"state":' in s.ws_part()
