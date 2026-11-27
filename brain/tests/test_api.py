"""Integration tests on the real connectome (needs DATA_DIR / /data)."""
import time

import pytest
from conftest import needs_data

pytestmark = needs_data


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def advance(client, sim_ms, timeout=60):
    t0 = client.get("/health").json()["sim_ms"]
    t_end = time.time() + timeout
    while client.get("/health").json()["sim_ms"] < t0 + sim_ms:
        assert time.time() < t_end, "simulation not advancing"
        time.sleep(0.02)


def frame_stats(client, sim_ms=300):
    """Collect stream frames for roughly sim_ms of sim time -> (spike neuron set, last frame)."""
    seen, last = set(), None
    with client.websocket_connect("/stream") as ws:
        for _ in range(3):  # frames built before the preceding POST may still be cached; drop them
            first = ws.receive_json()
        t_end = first["t_ms"] + sim_ms
        last = first
        while last["t_ms"] < t_end:
            last = ws.receive_json()
            seen.update(last["spikes"])
    return seen, last


def test_health_loads_connectome(client):
    h = client.get("/health").json()
    assert h["ok"] is True
    assert h["neurons"] == 138639
    assert 14_000_000 < h["synapses"] < 16_000_000


def test_groups_contract(client):
    g = client.get("/groups").json()
    for name in ("sugar_grn", "looming", "forward_walk", "turn_left", "turn_right"):
        assert name in g and g[name]["size"] > 0
    assert g["sugar_grn"]["size"] == 21 and g["sugar_grn"]["default_rate_hz"] == 200.0


def test_params_contract_and_live_update(client):
    p = client.get("/params").json()
    for k in ("w_syn", "tau_m_ms", "v_th_mV", "v_reset_mV", "tau_syn_ms", "t_ref_ms", "poisson_rate_hz", "dt_ms"):
        assert k in p
    assert p["dt_ms"]["value"] == 0.1 and p["dt_ms"]["readonly"] is True
    for k in ("w_syn", "tau_m_ms"):
        assert {"value", "min", "max", "step"} <= set(p[k]) and p[k]["min"] < p[k]["value"] < p[k]["max"]
    r = client.put("/params", json={"tau_m_ms": 25.0})
    assert r.status_code == 200 and r.json()["tau_m_ms"] == 25.0
    assert client.get("/params").json()["tau_m_ms"]["value"] == 25.0
    assert client.put("/params", json={"dt_ms": 0.2}).status_code == 422
    assert client.put("/params", json={"bogus": 1}).status_code == 422
    assert client.put("/params", json={"tau_m_ms": -3}).status_code == 422
    client.post("/params/reset")
    assert client.get("/params").json()["tau_m_ms"]["value"] == 20.0


def test_stimulate_errors(client):
    assert client.post("/stimulate", json={"group": "nope", "rate_hz": 10}).status_code == 404
    assert client.post("/stimulate", json={"rate_hz": 10}).status_code == 422
    assert client.post("/stimulate", json={"neuron_ids": [1], "group": "sugar_grn"}).status_code == 422


def test_sugar_stimulation_propagates_and_silence_blocks(client):
    client.post("/clear")
    client.post("/reset")
    groups = client.get("/groups").json()
    assert client.post("/stimulate", json={"group": "sugar_grn", "rate_hz": 100}).json()["neurons"] == 21
    spikes, last = frame_stats(client, 500)
    assert last["group_rates"]["sugar_grn"] > 50
    assert client.get("/state").json()["stimulated_groups"] == {"sugar_grn": 100.0}
    # activity beyond the 32 directly driven neurons => spikes propagate through the connectome
    assert len(spikes) > 100, f"only {len(spikes)} distinct neurons spiked"

    # silencing the driven neurons themselves: stimulation of silenced neurons has no effect at all
    client.post("/silence", json={"group": "sugar_grn"})  # silence before reset: no drive may leak in
    client.post("/reset")
    spikes2, last2 = frame_stats(client, 300)
    assert len(spikes2) == 0 and last2["group_rates"]["sugar_grn"] == 0
    client.post("/clear")


def test_params_apply_live_to_activity(client):
    client.post("/clear")
    client.post("/reset")
    client.post("/stimulate", json={"group": "sugar_grn", "rate_hz": 100})
    base, _ = frame_stats(client, 400)
    client.put("/params", json={"w_syn": 0.0})  # no coupling (and no kick) -> silent
    client.post("/reset")
    zero, last = frame_stats(client, 300)
    assert len(zero) == 0
    client.put("/params", json={"w_syn": 0.275})
    client.post("/reset")
    back, _ = frame_stats(client, 400)
    assert len(back) > 100 and len(base) > 100
    client.post("/clear")


def test_background_poisson_drive_param(client):
    client.post("/clear")
    client.post("/reset")
    client.put("/params", json={"poisson_rate_hz": 1.0})
    spikes, _ = frame_stats(client, 300)
    client.put("/params", json={"poisson_rate_hz": 0.0})
    assert len(spikes) > 100
    client.post("/reset")


def test_stream_frame_contract(client):
    client.post("/clear")
    client.post("/reset")
    _, f = frame_stats(client, 50)
    assert set(f) >= {"t_ms", "motor", "group_rates", "spikes", "params"}
    assert 0.0 <= f["motor"]["forward"] <= 1.0 and -1.0 <= f["motor"]["turn"] <= 1.0
    assert len(f["spikes"]) <= 2000
    assert "forward_walk" in f["group_rates"] and f["params"]["dt_ms"] == 0.1


def test_motor_responds_to_forward_group(client):
    client.post("/clear")
    client.post("/reset")
    client.post("/stimulate", json={"group": "forward_walk", "rate_hz": 150})
    _, f = frame_stats(client, 1000)
    assert f["motor"]["forward"] > 0.4
    client.post("/clear")
    client.post("/stimulate", json={"group": "turn_left", "rate_hz": 150})
    _, f = frame_stats(client, 1000)
    assert f["motor"]["turn"] < -0.3
    client.post("/clear")
    client.post("/reset")


def test_drive_channel_api(client):
    client.post("/clear")
    client.post("/reset")
    ids = [720575940628857210]  # an 18-digit FlyWire root id (exact int) and a tensor index
    r = client.post("/drive", json={"channel": "t", "neuron_ids": ids + [5], "rates_hz": [200.0, 0.0]})
    assert r.status_code == 200 and r.json()["driven"] == 1
    assert client.get("/state").json()["drive_channels"] == {"t": 1}
    assert client.post("/drive", json={"neuron_ids": [1, 2], "rates_hz": [1.0]}).status_code == 422
    assert client.post("/drive", json={"neuron_ids": [1, 1], "rates_hz": [1.0, 1.0]}).status_code == 422
    assert client.post("/drive", json={"neuron_ids": [1], "rates_hz": [-1.0]}).status_code == 422
    assert client.post("/drive", json={"neuron_ids": [999999999999999999], "rates_hz": [1.0]}).status_code == 422
    client.post("/stimulate", json={"group": "sugar_grn", "rate_hz": 50})  # groups and channels are independent
    st = client.get("/state").json()
    assert st["drive_channels"] == {"t": 1} and st["stimulated_groups"] == {"sugar_grn": 50.0}
    client.post("/clear")
    assert client.get("/state").json()["drive_channels"] == {}


def test_vision_endpoint_drives_photoreceptors(client):
    from app import retina

    client.post("/clear")
    client.post("/reset")
    meta = client.get("/vision").json()
    assert meta["ommatidia_per_eye"] == retina.N_OMM and meta["photoreceptors"]["left"] > 4000
    u = [255] * retina.N_OMM
    assert client.post("/vision", json={"left": u[:10], "right": u, "max": 255}).status_code == 422
    r = client.post("/vision", json={"left": u, "right": u, "max": 255}).json()
    assert r["active"] == 0  # uniform light: no photoreceptor input
    lat = retina.lattice()
    dark = [25 if (a * a + b * b) ** 0.5 < 20 else 255 for a, b in lat]
    advance(client, 70)
    r = client.post("/vision", json={"left": dark, "right": u, "max": 255}).json()
    assert r["active"] > 100 and client.get("/state").json()["drive_channels"]["vision"] == r["active"]
    spikes, _ = frame_stats(client, 200)
    assert len(spikes) > 100  # the driven photoreceptors (and whatever follows) spike
    assert client.delete("/vision").json()["ok"] and "vision" not in client.get("/state").json()["drive_channels"]
    client.post("/clear")
    client.post("/reset")
