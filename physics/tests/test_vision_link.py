"""Renderer (app/vision.py) and the vision link to the brain (app/vision_link.py), with a mock brain."""
import json
import math

import httpx
import numpy as np

from app import vision
from app.vision_link import SCRIPTED_VISUAL_GROUPS, VisionLink, filter_drives
from app.world import WorldObject

FLY = (0.0, 0.0, 0.98)
Q = (1.0, 0.0, 0.0, 0.0)  # heading +x


def scary(d, bearing_deg):
    b = math.radians(bearing_deg)
    return WorldObject(1, "scary", d * math.cos(b), d * math.sin(b))


def dark(img, thr=0.3):
    return int((np.asarray(img) < thr).sum())


def test_lattice_and_empty_scene():
    assert vision.N_OMM == 661
    out = vision.EyeRenderer().render(FLY, Q, [])
    for side in ("left", "right"):
        v = out[side]
        assert v.shape == (661,)
        assert v.min() >= vision.LUM["floor"] - 1e-9 and v.max() <= vision.LUM["sky"] + 1e-9
    assert np.allclose(out["left"], out["right"])  # symmetric eyes, symmetric empty scene


def test_looming_object_grows_in_the_eye_on_its_side():
    r = vision.EyeRenderer()
    far = r.render(FLY, Q, [scary(60.0, 40.0)])
    near = r.render(FLY, Q, [scary(8.0, 40.0)])
    assert dark(near["left"]) > 4 * max(1, dark(far["left"]))  # the scary object fills more ommatidia when close
    assert dark(near["left"]) > dark(near["right"])  # object on the left (bearing +40 deg)
    mirrored = r.render(FLY, Q, [scary(8.0, -40.0)])
    assert dark(mirrored["right"]) == dark(near["left"])


def test_filter_drives_modes():
    d = {g: 50.0 for g in SCRIPTED_VISUAL_GROUPS} | {"sugar_grn": 10.0, "p9": 100.0}
    assert filter_drives(d, "scripted") == d
    assert filter_drives(d, "both") == d
    assert filter_drives(d, "vision") == {"sugar_grn": 10.0, "p9": 100.0}


class Brain:
    def __init__(self, fail=False):
        self.posts, self.fail = [], fail

    def handler(self, req):
        if self.fail:
            raise httpx.ConnectError("refused", request=req)
        body = json.loads(req.content) if req.content else None
        self.posts.append((req.method, req.url.path, body))
        return httpx.Response(200, json={"ok": True, "active": 123, "mean_rate_hz": 4.5})


def frame(t=1.0):
    return {"t": t, "pos": np.array([[0, 0, 0], FLY]), "quat": np.array([[1, 0, 0, 0], Q])}


def link(brain, send):
    c = httpx.Client(base_url="http://brain", transport=httpx.MockTransport(brain.handler))
    return VisionLink("http://brain", None, send=send, client=c)


def test_posts_image_when_enabled():
    b = Brain()
    v = link(b, True)
    assert v.step() is False  # nothing submitted yet
    v.submit(frame(), 1, [scary(10.0, 30.0)])
    assert v.step() is True
    (method, path, body), = b.posts
    assert (method, path) == ("POST", "/vision") and body["max"] == 255.0
    assert len(body["left"]) == len(body["right"]) == 661
    assert all(isinstance(x, int) and 0 <= x <= 255 for x in body["left"])
    assert min(body["left"]) <= round(0.05 * 255) + 30  # the dark scary object is in the left image
    s = v.status()
    assert s["posts_ok"] == 1 and s["brain_photoreceptors_active"] == 123 and v.image["t"] == 1.0


def test_render_only_when_disabled_and_failures_counted():
    b = Brain()
    v = link(b, False)
    v.submit(frame(), 1, [])
    assert v.step() and b.posts == [] and v.image is not None
    bad = link(Brain(fail=True), True)
    bad.submit(frame(), 1, [])
    bad.step()
    assert bad.posts_failed == 1 and bad.posts_ok == 0


def test_thread_renders_at_its_own_rate():
    import time

    v = link(Brain(), False)
    v.hz = 50.0
    v.start()
    try:
        for i in range(20):
            v.submit(frame(0.01 * i), 1, [])
            time.sleep(0.01)
    finally:
        v.stop()
    assert 1 <= v.renders <= 20
