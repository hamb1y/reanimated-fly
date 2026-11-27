"""Per-neuron drive channels (model) and the eye: lattice, photoreceptor map, transduction. No connectome needed."""
import json

import numpy as np
import pytest
import torch

from app import retina
from app.model import Brain

torch.set_num_threads(1)


def toy(edges, n=4):
    pre = np.array([e[0] for e in edges], dtype=np.int64)
    post = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([e[2] for e in edges], dtype=np.float32)
    o = np.argsort(pre, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    return Brain(indptr, post[o], w[o], device="cpu", record=True)


def spiked(b, steps=200):
    b.raster = []
    b.run(steps)
    return set(i for _, idx in b.raster for i in idx.tolist())


def test_drive_channel_spikes_merges_and_clears():
    b = toy([(0, 1, 100.0)])
    b.set_drive("eye", [0, 2], [1000.0 / 0.1, 0.0])  # rate 0 is dropped
    assert set(b.drive["eye"][0].tolist()) == {0}
    s = spiked(b)
    assert {0, 1} <= s and 2 not in s
    b.stimulate([3], 1000.0 / 0.1)  # group-style stimulation and channels coexist
    assert {0, 1, 3} <= spiked(b)
    b.set_drive("eye", [], [])  # empty vector removes the channel
    assert "eye" not in b.drive and 0 not in spiked(b) and 3 in spiked(b)
    b.set_drive("eye", [0], [1000.0 / 0.1])
    b.silence([0])  # silencing wins over a channel
    assert 0 not in spiked(b)
    b.clear()
    assert b.drive == {} and spiked(b) == set()


def test_drive_channel_larger_rate_wins():
    b = toy([(0, 1, 100.0)])
    b.stimulate([0], 5.0)
    b.set_drive("a", [0], [50.0])
    assert float(b.stim_p[0]) == pytest.approx(50.0 * 0.1 / 1000.0)
    with pytest.raises(ValueError):
        b.set_drive("a", [0, 1], [1.0])


def test_lattice_matches_web_and_physics():
    lat = retina.lattice()
    assert lat.shape == (661, 2) and retina.N_OMM == 661  # eyes.ts / vision.py render the same 661 per eye
    assert lat[0][1] == pytest.approx(-15 * 4.3 * 0.866)  # first row: r = -15 (|b| <= 58)
    assert np.hypot(lat[:, 0], lat[:, 1]).max() <= 58.0 + 1e-9
    assert np.allclose(np.sort(np.unique(np.round(np.diff(np.unique(lat[:, 1])), 3))), [round(4.3 * 0.866, 3)])


def fake_map(n_per_eye=50, seed=0):
    rng = np.random.default_rng(seed)
    m = {}
    for k, eye in enumerate(("left", "right")):
        ab = rng.uniform(-40, 40, size=(n_per_eye, 2))
        m[eye] = {"idx": np.arange(n_per_eye) + k * n_per_eye, "ab": ab, "type": np.array(["R1-6"] * n_per_eye)}
    return m


def disk(center, radius, lum=0.1):
    lat = retina.lattice()
    v = np.ones(retina.N_OMM)
    v[np.hypot(lat[:, 0] - center[0], lat[:, 1] - center[1]) < radius] = lum
    return v


def test_transducer_silent_on_uniform_and_side_specific():
    m = fake_map()
    for mode in ("local", "global"):
        tr = retina.Transducer(m, mode=mode)
        u = np.full(retina.N_OMM, 0.7)
        assert tr({"left": u, "right": u}, 0.0).max() == 0.0  # no contrast, no spikes (the brain stays silent)
        r = tr({"left": disk((0, 0), 25), "right": u}, 0.066)
        left, right = r[:50], r[50:]
        inside = np.hypot(*m["left"]["ab"].T) < 20
        assert right.max() == 0.0 and (left[inside] > 50).all() and (left[np.hypot(*m["left"]["ab"].T) > 30] == 0).all()


def test_local_adaptation_fades_static_global_keeps():
    m = fake_map()
    u = np.ones(retina.N_OMM)
    res = {}
    for mode in ("local", "global"):
        tr = retina.Transducer(m, mode=mode, tau_s=0.5)
        tr({"left": u, "right": u}, 0.0)
        first = tr({"left": disk((0, 0), 20), "right": u}, 0.066).sum()
        for _ in range(60):  # 4 s of a static dark disk
            last = tr({"left": disk((0, 0), 20), "right": u}, 0.066).sum()
        res[mode] = (first, last)
    assert res["local"][1] < 0.05 * res["local"][0]
    assert res["global"][1] > 0.5 * res["global"][0]


def test_transducer_rejects_wrong_length_and_subsamples():
    m = fake_map(200)
    with pytest.raises(ValueError):
        retina.Transducer(m)({"left": np.ones(10), "right": np.ones(10)}, 0.1)
    assert 100 < len(retina.Transducer(m, subsample=0.5).idx) < 300


@pytest.mark.skipif(not retina.RETINA_JSON.exists(), reason="app/retina.json not built")
def test_retina_json_conventions_and_validation():
    raw = json.load(open(retina.RETINA_JSON))
    val = raw["meta"]["validation"]
    for side in ("left", "right"):
        v = val[side]
        assert v["chiasm_corr_dorsoventral"] > 0.9 and v["chiasm_corr_anteroposterior"] < -0.3  # outer chiasm
        assert v["dra_R7_elev_median_deg"] > v["all_R7_elev_median_deg"] + 20  # dorsal rim is up
        assert v["ommatidia_within_one_spacing_of_a_receptor"] > 0.85 * 661
    ids = np.sort(np.concatenate([np.asarray(raw["eyes"][s]["root_id"], dtype=np.int64) for s in ("left", "right")]))
    g = retina.load_map(ids, convention="groups")
    a = retina.load_map(ids, convention="annotation")
    assert g["left"]["anat_side"] == "right" and a["left"]["anat_side"] == "left"
    assert np.array_equal(np.sort(g["left"]["idx"]), np.sort(a["right"]["idx"]))
    assert len(a["left"]["idx"]) == len(raw["eyes"]["left"]["root_id"])  # exact 18-digit ids round-trip
