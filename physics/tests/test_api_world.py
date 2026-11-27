"""World, personality and speed endpoints; WS frames carry objects and state. Real sim, no brain."""

import math
import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ["BRAIN_URL"] = ""  # no brain (WS or HTTP) in these tests
os.environ["BRAIN_HTTP_URL"] = ""


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _yaw(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _wait(pred, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_world_crud_and_validation(client):
    client.delete("/world/objects")
    w = client.get("/world").json()
    assert w["objects"] == [] and w["personality"] == {"hunger": 0.8, "fear": 0.9, "lust": 0.9, "receptivity": 0.9}
    assert set(w["state"]) >= {"mode", "hunger_level", "nearest", "drives"}

    oid = client.post("/world/objects", json={"kind": "food", "pos": [10, 0]}).json()["id"]
    obj = client.get("/world").json()["objects"][0]
    assert obj == {"id": oid, "kind": "food", "pos": [10.0, 0.0], "amount": 100.0}
    assert client.put(f"/world/objects/{oid}", json={"pos": [12.5, -3]}).json()["pos"] == [12.5, -3.0]
    assert client.put("/world/objects/999999", json={"pos": [0, 0]}).status_code == 404
    assert client.post("/world/objects", json={"kind": "cake", "pos": [0, 0]}).status_code == 422
    assert client.post("/world/objects", json={"kind": "mate", "pos": [0]}).status_code == 422
    assert client.post("/world/objects", json={"kind": "mate", "pos": [5000, 0]}).status_code == 422
    assert client.delete(f"/world/objects/{oid}").status_code == 200
    assert client.delete(f"/world/objects/{oid}").status_code == 404
    client.post("/world/objects", json={"kind": "scary", "pos": [-30, 30]})
    client.post("/world/objects", json={"kind": "mate", "pos": [40, 40]})
    assert client.delete("/world/objects").json()["removed"] == 2
    assert client.get("/world").json()["objects"] == []


def test_personality_put_and_validation(client):
    r = client.put("/personality", json={"hunger": 0.3, "receptivity": 0.0})
    assert r.json()["hunger"] == 0.3 and r.json()["receptivity"] == 0.0
    assert client.get("/personality").json()["hunger"] == 0.3
    assert client.put("/personality", json={"fear": 1.5}).status_code == 422
    assert client.put("/personality", json={"lust": -0.1}).status_code == 422
    assert client.put("/personality", json={"cuteness": 0.5}).status_code == 422
    client.put("/personality", json={"hunger": 0.8, "receptivity": 0.9})


def test_speed_put_validation_and_health(client):
    assert client.put("/speed", json={"factor": 3.0}).status_code == 422
    assert client.put("/speed", json={"factor": 0.05}).status_code == 422
    r = client.put("/speed", json={"factor": 0.5}).json()
    assert r["factor"] == 0.5 and r["brain_synced"] is False  # no brain configured
    h = client.get("/health").json()
    assert h["speed"]["target"] == 0.5 and set(h["speed"]) == {"target", "achieved", "capacity"}
    client.put("/speed", json={"factor": 1.0})


def test_ws_frames_carry_objects_and_state(client):
    client.delete("/world/objects")
    client.post("/world/objects", json={"kind": "food", "pos": [200, 200]})
    with client.websocket_connect("/stream") as ws:
        f = ws.receive_json()
        assert "bodies" in f and "command" in f and "source" in f
        assert f["objects"][0]["kind"] == "food" and f["objects"][0]["amount"] == 100.0
        assert f["state"]["mode"] in ("explore", "idle", "eat", "chase", "flee")
    client.delete("/world/objects")


def test_turns_toward_food_on_the_left_and_eats(client):
    """Real sim, no brain: scripted bearing rule with gain 0.5. Food 25 mm to the fly's left."""
    client.post("/reset")
    client.delete("/world/objects")
    client.post("/world/objects", json={"kind": "food", "pos": [0, 25]})
    scene = client.get("/model").json()
    thorax = next(i for i, b in enumerate(scene["bodies"]) if b["name"] == "nmf/c_thorax")
    with client.websocket_connect("/stream") as ws:
        f0 = ws.receive_json()
        yaw0 = _yaw(f0["bodies"][thorax]["quat"])
        frames = [f0]
        while frames[-1]["t"] < f0["t"] + 3.0:
            frames.append(ws.receive_json())
        dyaw = (_yaw(frames[-1]["bodies"][thorax]["quat"]) - yaw0 + 180) % 360 - 180
        assert dyaw > 20.0, f"did not turn left toward food: {dyaw:.1f} deg"  # food is 90 deg to the left
        assert any(f["command"]["turn"] > 0 for f in frames)
        assert frames[-1]["state"]["mode"] == "eat" and frames[-1]["source"] == "world"
    client.delete("/world/objects")
