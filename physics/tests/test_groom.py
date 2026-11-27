import types

import numpy as np
import pytest

from app import groom as G
from app.command import CommandState


def test_clip_loads_and_is_recorded_data():
    c = G.Clip.load()
    assert c.q.shape[1:] == (2, 7) and c.q.shape[0] > 100
    assert c.meta["license"] == "Apache-2.0" and "NeuroMechFly" in c.meta["source"]
    assert 1.0 < c.period < 3.0
    assert np.isfinite(c.q).all() and np.abs(c.q).max() < 2 * np.pi


def test_convention_mapping_reproduces_the_v1_coxa_rotation():
    rng = np.random.default_rng(0)
    raw = rng.uniform(-0.6, 0.6, size=(50, 7))
    want = G.coxa_v1_matrix(raw[:, 0], raw[:, 1], raw[:, 2])  # yaw, pitch, roll
    left = G.to_v2_qpos(raw, "l")
    got = G.coxa_v2_matrix(left[:, 0], left[:, 1], left[:, 2])
    assert np.allclose(got, want, atol=1e-9)
    np.testing.assert_allclose(left[:, 3:], raw[:, 3:])  # femur etc. unchanged on the left
    right = G.to_v2_qpos(raw, "r")
    # right axes are negated in flygym: yaw and roll flip, the global rotation is the same
    got_r = G.coxa_v2_matrix(-right[:, 0], right[:, 1], -right[:, 2])
    assert np.allclose(got_r, want, atol=1e-9)
    assert np.allclose(right[:, 4], -raw[:, 4]) and np.allclose(right[:, 5], raw[:, 5])


def test_loop_has_no_snap_at_the_wrap():
    c = G.Clip.load()
    eps = 1e-4
    jump = np.abs(c.pose(c.period - eps) - c.pose(c.period + eps)).max()
    assert jump < 0.01
    # and the largest per-sample change inside the loop is of the same order
    step = np.abs(np.diff(c.q, axis=0)).max()
    assert jump < step


def _fake_groomer():
    from flygym.anatomy import JointDOF, BodySegment, RotationAxis

    dofs = []
    for leg in G.LEGS_ALL:
        for key in G.KEYS:
            par, chi, ax = G.DOF_SPEC[key]
            dofs.append(JointDOF(BodySegment(par.format(s=leg)), BodySegment(chi.format(s=leg)), RotationAxis(ax)))
    return G.Groomer(dofs), len(dofs)


def test_groomer_blends_only_forelegs_and_releases_adhesion():
    g, n = _fake_groomer()
    base = np.zeros(n)
    adh = np.ones(6, dtype=bool)
    out, a = g.apply(base, adh, 0.002)
    assert out is base and g.weight == 0 and not g.engaged  # idle: untouched
    g.request(True)
    assert g.engaged
    t = 0.0
    while t < G.SETTLE_S + G.BLEND_IN_S + 0.05:
        out, a = g.apply(base, adh, 0.002)
        t += 0.002
    assert g.weight == 1.0 and g.grooming
    moved = np.flatnonzero(np.abs(out) > 1e-9)
    fore = set(g.idx.ravel().tolist()) | {ix for ix, _ in g.support}
    assert set(moved.tolist()) <= fore
    assert a[g.adh_idx].sum() == 0 and a.sum() == 4  # forelegs unglued, the other four keep adhesion
    assert (base == 0).all()  # input not modified


def test_groomer_blends_in_smoothly_and_out_again():
    g, n = _fake_groomer()
    base = np.zeros(n)
    g.request(True)
    prev, worst = base, 0.0
    for _ in range(int(3.0 / 0.002)):
        out, _ = g.apply(base, None, 0.002)
        worst = max(worst, np.abs(out - prev).max())
        prev = out
    # no snap: the largest per-tick change is the recorded motion's own speed (the clip is fast, ~0.37 rad per 10 ms sample),
    # not a blend jump; the blend itself adds at most a few percent on top.
    clip_tick = np.abs(np.diff(g.clip.q, axis=0)).max() * 0.002 * g.clip.fps
    assert worst < clip_tick * 1.5
    g.request(False)
    assert g.engaged  # still blending out: walk stays frozen
    for _ in range(int(1.0 / 0.002)):
        out, _ = g.apply(base, None, 0.002)
    assert g.weight == 0.0 and not g.engaged and np.abs(out).max() == 0.0


def test_trigger_hysteresis_and_missing_rate():
    tr = G.GroomTrigger()
    t = 0.0
    for _ in range(30):  # 2 s at 15 Hz of silence
        t += 1 / 15
        assert tr.update(0.0, t) is False
    on_at = None
    for _ in range(60):  # quantised spiky rate averaging ~5.6 Hz (as measured on the brain at 200 Hz jo drive)
        t += 1 / 15
        r = 10.0 if int(t * 15) % 2 else 0.0
        if tr.update(r, t) and on_at is None:
            on_at = t
    assert tr.state and on_at is not None
    # a brief dip does not turn it off
    for _ in range(3):
        t += 1 / 15
        tr.update(0.0, t)
    assert tr.state
    off_at = None
    for _ in range(90):
        t += 1 / 15
        if not tr.update(None, t):  # no brain frames count as 0 Hz
            off_at = t
            break
    assert off_at is not None


def test_commandstate_group_rates_need_a_fresh_connected_brain():
    cs = CommandState()
    cs.set_group_rates({"groom_adn1": 5.0}, 10.0)
    assert cs.group_rate("groom_adn1", 10.0) is None  # not connected
    cs.set_brain_connected(True)
    cs.set_group_rates({"groom_adn1": 5.0, "x": "bad"}, 10.0)
    assert cs.group_rate("groom_adn1", 10.5) == 5.0 and cs.group_rate("x", 10.5) is None
    assert cs.group_rate("groom_adn1", 12.0) is None  # stale
    cs.set_brain_connected(False)
    assert cs.group_rate("groom_adn1", 10.1) is None


def test_world_tick_dust_drives_jo_and_brain_rate_requests_grooming():
    from app.main import _groom_tick

    calls = []
    groom = types.SimpleNamespace(engaged=False, request=lambda on: calls.append(on))
    groom.engaged = False
    cs = CommandState()
    cs.set_brain_connected(True)
    st = types.SimpleNamespace(
        flysim=types.SimpleNamespace(groom=groom), commands=cs, trigger=G.GroomTrigger(), groom_manual=False,
        dust_hz=200.0, groom_info={"adn1_hz": None, "on_delay_s": None, "dust_t": 0.0},
    )
    drives = {"p9": 100.0}
    t = 0.0
    for _ in range(45):
        t += 1 / 15
        cs.set_group_rates({"groom_adn1": 6.0}, t)
        _groom_tick(st, t, drives)
    assert drives["jo_mechano"] == 200.0
    assert calls[-1] is True and st.groom_info["on_delay_s"] is not None and st.groom_info["on_delay_s"] < 1.5
    groom.engaged = True
    d2 = {"p9": 100.0}
    _groom_tick(st, t + 0.05, d2)
    assert "p9" not in d2  # walking neurons are not driven while she stands and grooms
