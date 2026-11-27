"""POST /groom, POST /dust, mode and flags. Real sim, no brain (so dust alone cannot trigger grooming here)."""
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ["BRAIN_URL"] = ""
os.environ["BRAIN_HTTP_URL"] = ""


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _wait(pred, timeout=12.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_manual_groom_sets_mode_flag_and_releases(client):
    assert client.get("/groom").json()["requested"] is False
    r = client.post("/groom", json={"on": True}).json()
    assert r["manual"] is True and r["requested"] is True
    assert _wait(lambda: client.get("/groom").json()["grooming"])
    assert _wait(lambda: client.get("/world").json()["state"]["mode"] == "groom")
    assert _wait(lambda: client.get("/world").json()["state"]["grooming"] is True)  # state is a 15 Hz snapshot
    with client.websocket_connect("/stream") as ws:
        f = json.loads(ws.receive_text())
    assert f["grooming"] is True and "bodies" in f and "state" in f
    assert client.get("/health").json()["groom"]["weight"] > 0.05
    client.post("/groom", json={"on": False})
    assert _wait(lambda: not client.get("/groom").json()["grooming"] and not client.get("/groom").json()["requested"])
    assert _wait(lambda: client.get("/world").json()["state"]["mode"] != "groom")


def test_dust_drives_jo_mechano_and_validates(client):
    assert client.post("/dust", json={"on": True, "rate_hz": 9999}).status_code == 422
    r = client.post("/dust", json={"on": True}).json()
    assert r["dust_hz"] == 200.0
    assert _wait(lambda: client.get("/world").json()["state"]["drives"].get("jo_mechano") == 200.0)
    assert client.get("/world").json()["state"]["dust"] is True
    r = client.post("/dust", json={"on": False}).json()
    assert r["dust_hz"] == 0.0
    assert _wait(lambda: "jo_mechano" not in client.get("/world").json()["state"]["drives"])
