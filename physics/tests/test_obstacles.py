"""Kitchen props, obstacle ray sensing, steering filter, user solids, the MuJoCo block and the /props API.

Real MuJoCo sim for the block test (one test, slow-ish) and the real app for the API test. No brain.
"""

import math
import os
import time

import pytest
from fastapi.testclient import TestClient

from app import obstacles as O
from app.obstacles import Avoider, Footprint, ObstacleSense, load_props, ray_angles, ray_hit, user_footprints
from app.world import WorldStore

os.environ["BRAIN_URL"] = ""  # no brain (WS or HTTP) in these tests
os.environ["BRAIN_HTTP_URL"] = ""


def _rot_extent(fp):
    """Axis-aligned half extents of a box after its yaw."""
    c, s = abs(math.cos(fp.yaw)), abs(math.sin(fp.yaw))
    return fp.hx * c + fp.hy * s, fp.hx * s + fp.hy * c


def _origin_inside(fp, x, y):
    if fp.shape == "circle":
        return math.hypot(x - fp.x, y - fp.y) <= fp.r
    cs, sn = math.cos(fp.yaw), math.sin(fp.yaw)
    px, py = x - fp.x, y - fp.y
    lx, ly = cs * px + sn * py, -sn * px + cs * py
    return abs(lx) <= fp.hx and abs(ly) <= fp.hy


# --- kitchen props -------------------------------------------------------------------------

def test_props_load_unique_and_valid():
    raw, fps = load_props()
    assert len(fps) == len(raw["props"]) == 16
    assert len({f.id for f in fps}) == len(fps)
    assert {f.shape for f in fps} <= {"circle", "box"}


def test_props_inside_counter():
    # 3 mm tolerance: the cereal box is drawn rotated (yaw -0.12) and overhangs the counter edge by 1.7 mm in kitchen.ts too
    raw, fps = load_props()
    hx, hy = raw["counter"]["half_x"], raw["counter"]["half_y"]
    for f in fps:
        ex, ey = (f.r, f.r) if f.shape == "circle" else _rot_extent(f)
        assert abs(f.x) + ex <= hx + 3.0, f.id
        assert abs(f.y) + ey <= hy + 3.0, f.id


def test_counter_centre_is_free():
    _, fps = load_props()
    assert not any(_origin_inside(f, 0.0, 0.0) for f in fps)


# --- ray_hit: circles ----------------------------------------------------------------------

def test_ray_hit_circle():
    c = Footprint("c", "circle", 0.0, 0.0, r=2.0)
    assert ray_hit(c, -10, 0, 1, 0) == pytest.approx(8.0)
    assert ray_hit(c, -10, 0, -1, 0) == math.inf  # pointing away
    inside = Footprint("c", "circle", 0.0, 0.0, r=2.0)
    assert ray_hit(inside, 1, 0, -1, 0) == 0.0  # inside, toward the centre
    assert ray_hit(inside, 1, 0, 1, 0) == math.inf  # inside, away from the centre
    assert ray_hit(c, -10, 0, 1, 0, inflate=1.0) == pytest.approx(7.0)


# --- ray_hit: boxes ------------------------------------------------------------------------

def test_ray_hit_box():
    b = Footprint("b", "box", 0.0, 0.0, hx=2.0, hy=1.0, yaw=0.0)
    assert ray_hit(b, -10, 0, 1, 0) == pytest.approx(8.0)
    assert ray_hit(b, 0, -10, 0, 1) == pytest.approx(9.0)
    assert ray_hit(b, -10, 5, 1, 0) == math.inf  # misses
    turned = Footprint("b", "box", 0.0, 0.0, hx=2.0, hy=1.0, yaw=math.pi / 2)
    assert ray_hit(turned, -10, 0, 1, 0) == pytest.approx(9.0)  # long side now runs along y


# --- ObstacleSense.cast --------------------------------------------------------------------

def test_cast_circle_ahead_and_behind():
    sense = ObstacleSense([Footprint("c", "circle", 10.0, 0.0, r=1.0)])
    rays = sense.cast((0.0, 0.0), 0.0)
    assert len(rays["dist"]) == len(ray_angles())
    front = rays["dist"][rays["angles"].index(min(rays["angles"], key=lambda a: abs(a)))]
    assert math.isfinite(front) and front < 10.0
    for a, d in zip(rays["angles"], rays["dist"]):
        if abs(abs(a) - math.pi) < 1e-6:
            assert d == math.inf
    assert rays["touch"] == {"left": False, "right": False}


def test_cast_touch_when_close():
    sense = ObstacleSense([Footprint("c", "circle", 2.0, 0.0, r=1.0)])
    touch = sense.cast((0.0, 0.0), 0.0)["touch"]
    assert touch["left"] or touch["right"]


# --- Avoider.filter ------------------------------------------------------------------------

def test_avoider_clear_passes_through():
    rays = {"angles": ray_angles(), "dist": [math.inf] * len(ray_angles()), "touch": {}}
    fwd, turn, info = Avoider().filter(rays, 1.0, 0.3, goal_bearing=0.0, goal_dist=30.0)
    assert (fwd, turn) == (1.0, 0.3)
    assert info["blocked"] is False


def _blocked_rays():
    sense = ObstacleSense([Footprint("c", "circle", 12.0, 0.0, r=5.0)])
    return sense.cast((0.0, 0.0), 0.0)


def test_avoider_turns_away_and_keeps_side():
    rays = _blocked_rays()
    av = Avoider()
    fwd, turn, info = av.filter(rays, 1.0, 0.0, goal_bearing=0.0, goal_dist=30.0)
    assert info["blocked"] is True
    assert abs(turn) > 0.2
    side = math.copysign(1, turn)
    for _ in range(3):
        _, turn2, _ = av.filter(rays, 1.0, 0.0, goal_bearing=0.0, goal_dist=30.0)
        assert math.copysign(1, turn2) == side


def test_avoider_zero_forward_unchanged():
    fwd, turn, info = Avoider().filter(_blocked_rays(), 0.0, 0.3, goal_bearing=0.0, goal_dist=30.0)
    assert (fwd, turn) == (0.0, 0.3)
    assert info["blocked"] is False


# --- user solids ---------------------------------------------------------------------------

def test_user_footprints_only_solids():
    s = WorldStore()
    s.add("food", (0, 0))
    s.add("mate", (10, 0))
    s.add("scary", (20, 0))
    s.add("block", (30, 0))
    fps = user_footprints(s.objects())
    assert [f.source for f in fps] == ["user", "user"]
    assert {f.id for f in fps} == {"obj3", "obj4"}
    block = next(f for f in fps if f.id == "obj4")
    assert block.r == O.SOLID_KINDS["block"]


# --- MuJoCo: a block stops a walking fly ---------------------------------------------------

def test_block_stops_walking_fly_in_mujoco():
    import mujoco

    from app.flysim import FlySim

    f = FlySim()
    assert f.n_fly_collide >= 20
    assert mujoco.mj_name2id(f.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "prop/fruit_bowl") >= 0
    f.pool.set([Footprint("b", "circle", 12.0, 0.0, r=8.0, h=16.0, source="user")])
    th = f.body_index("nmf/c_thorax")
    track = []
    for _ in range(30):  # 3 s of walking straight at it, sampled every 0.1 s
        f.advance(1.0, 0.0, 50)
        track.append(f.pose()["pos"][th][:2].copy())
    assert min(math.hypot(x - 12.0, y) for x, y in track) > 8.0  # never inside the block
    assert not any(x > 12.0 and abs(y) < 8.0 for x, y in track)  # never came out on the far side through it

    free = FlySim()  # same walk, no block: she gets past it
    free.advance(1.0, 0.0, 1500)
    assert free.pose()["pos"][th][0] > 20.0


# --- API -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_props_endpoint_lists_kitchen_and_user(client):
    import json

    client.delete("/world/objects")
    body = client.get("/props").json()
    raw = json.loads(O.PROPS_PATH.read_text())
    assert len(body["props"]) == len(raw["props"])
    assert body["user"] == []
    assert client.post("/world/objects", json={"kind": "block", "pos": [30, 0]}).status_code == 201
    user = client.get("/props").json()["user"]
    assert len(user) == 1 and user[0]["pos"] == [30, 0]
    client.delete("/world/objects")


# --- closed loop (no brain): food behind a user block -------------------------------------

def test_closed_loop_goes_round_block_and_eats():
    """FlySim + Behavior at 15 Hz (no brain, gain floor 0.5): block between her and the food. ~10 s CPU."""
    from app.behavior import Behavior
    from app.flysim import FlySim

    f = FlySim()
    store = WorldStore()
    store.add("block", (20.0, 0.0))
    food = store.add("food", (42.0, 0.0))
    b = Behavior(store)
    b.obstacle_sense = ObstacleSense(f.props)
    th = f.body_index("nmf/c_thorax")
    ticks = int(round((1 / 15) / f.control_dt))
    blocked = 0
    for _ in range(int(12 * 15)):
        fr = f.pose()
        r = b.tick(fr, th, None)
        blocked += bool(r["avoid"] and r["avoid"]["blocked"])
        f.pool.set(user_footprints(store.objects()))
        f.advance(r["forward"], r["turn"], ticks)
        x, y = fr["pos"][th][:2]
        assert math.hypot(x - 20.0, y) > O.SOLID_KINDS["block"]  # never inside the block
    assert blocked > 0  # the filter did detour
    left = store.get(food.id)
    assert left is None or left.amount < 100.0  # she reached the food and ate
