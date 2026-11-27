import math
import os

import pytest
from fastapi.testclient import TestClient

os.environ["BRAIN_URL"] = ""  # no brain in these tests; the brain client is tested separately

from app.flysim import FIDELITY_LEVELS  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _yaw_deg(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _read_frame(ws):
    return ws.receive_json()


def _advance_until(ws, pred, max_frames=5000):
    """Read stream frames until pred(frame) is true; returns the frame."""
    for _ in range(max_frames):
        f = _read_frame(ws)
        if pred(f):
            return f
    raise AssertionError("stream did not reach the wanted state")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["timestep"] == pytest.approx(FIDELITY_LEVELS[0][0])
    assert body["fidelity"]["level"] == 0
    assert body["fidelity"]["control_dt"] == pytest.approx(2e-3)
    assert body["brain"]["url"] is None
    assert body["command"]["source"] in ("hold", "manual", "world")  # world: the empty-world explorer


def test_model_scene(client):
    scene = client.get("/model").json()
    assert scene["units"] == "mm" and scene["quat_order"] == "wxyz"
    types = {g["type"] for g in scene["geoms"]}
    assert {"plane", "mesh"} <= types
    assert all(0 <= g["body"] < len(scene["bodies"]) for g in scene["geoms"])
    assert any(b["name"] == "nmf/c_thorax" for b in scene["bodies"])
    mesh_geom = next(g for g in scene["geoms"] if g["mesh_id"] is not None)
    mesh = client.get(f"/mesh/{mesh_geom['mesh_id']}").json()
    assert len(mesh["vertices"][0]) == 3 and len(mesh["faces"][0]) == 3
    assert max(max(f) for f in mesh["faces"]) < len(mesh["vertices"])
    assert client.get("/mesh/999999").status_code == 404


def test_command_validation(client):
    assert client.post("/command", json={"forward": "fast"}).status_code == 422
    r = client.post("/command", json={"forward": 3, "turn": -9})
    assert r.json() == {"source": "manual", "forward": 1.0, "turn": -1.0}
    client.post("/command", json={"release": True})


def test_walks_forward_and_turns(client):
    """Real walking: forward command moves the thorax; turn command yaws it."""
    scene = client.get("/model").json()
    thorax = next(i for i, b in enumerate(scene["bodies"]) if b["name"] == "nmf/c_thorax")

    client.post("/reset")
    with client.websocket_connect("/stream") as ws:
        f0 = _read_frame(ws)
        p0 = f0["bodies"][thorax]["pos"]
        client.post("/command", json={"forward": 1.0, "turn": 0.0})
        f1 = _advance_until(ws, lambda f: f["t"] - f0["t"] >= 0.6)
        p1 = f1["bodies"][thorax]["pos"]
        dx = p1[0] - p0[0]
        assert f1["source"] == "manual"
        assert dx > 3.0, f"forward walk too small: dx={dx:.2f} mm"
        assert f1["bodies"][thorax]["pos"][2] > 0.5  # still standing

        client.post("/command", json={"forward": 1.0, "turn": 1.0})
        h0 = _yaw_deg(f1["bodies"][thorax]["quat"])
        f2 = _advance_until(ws, lambda f: f["t"] - f1["t"] >= 0.6)
        dyaw = _yaw_deg(f2["bodies"][thorax]["quat"]) - h0
        dyaw = (dyaw + 180) % 360 - 180
        assert dyaw > 20.0, f"left turn too small: dyaw={dyaw:.1f} deg"
    client.post("/command", json={"release": True})
