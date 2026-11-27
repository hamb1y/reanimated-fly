"""Real vision into the brain: render what the fly sees (app/vision.py) and POST it to the brain's /vision.

Env:
- VISION_TO_BRAIN (default 0): 1 = POST every rendered image to the brain (POST /vision {left, right, max: 255}). The brain
  turns the 661 + 661 ommatidium luminances into photoreceptor (R1-8) Poisson rates (brain/app/retina.py). 0 = render
  only (GET /vision on physics still shows the image), nothing is sent.
- VISION_HZ (default 15): render/POST rate, wall clock. Renders happen in their own thread, never on the MuJoCo sim
  thread and never on the asyncio loop; the world loop only hands over the latest pose and a copy of the objects.
- LOOMING_DRIVE (default "scripted"): who drives the brain's visual threat neurons.
    scripted: as before, behavior.py/brainloop.py stimulate looming/LC4 + LPLC2 (+ escape_gf when very close) from the
              geometric angular-size growth. Real vision (if on) is sent in addition.
    vision:   those scripted groups are removed from the drives; the brain only gets photoreceptor input, so any LC4 /
              LPLC2 / giant-fibre activity comes from the connectome processing the image. Needs VISION_TO_BRAIN=1.
    both:     same as scripted (kept as an explicit name for experiments).
  Measured (physics/README.md "Real vision"): with the reference LIF constants the photoreceptor drive does NOT reach
  LC4/LPLC2 (0 spikes, even for a fully dark field), so "scripted" stays the default.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import httpx
import numpy as np

from .vision import EyeRenderer, quantise

log = logging.getLogger("physics.vision")

DEFAULT_HZ = 15.0
REQUEST_TIMEOUT_S = 0.5
LOOMING_MODES = ("scripted", "vision", "both")
# drives that stand in for the visual threat pathway; removed in LOOMING_DRIVE=vision
SCRIPTED_VISUAL_GROUPS = ("looming", "lplc2", "looming_left", "looming_right", "lplc2_left", "lplc2_right", "escape_gf")


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def looming_mode_from_env() -> str:
    m = os.environ.get("LOOMING_DRIVE", "scripted").strip().lower()
    if m not in LOOMING_MODES:
        log.warning("LOOMING_DRIVE=%r unknown, using 'scripted'", m)
        return "scripted"
    return m


def filter_drives(drives: dict, mode: str) -> dict:
    """Drop the scripted visual-threat groups when the looming drive comes from real vision."""
    if mode != "vision":
        return drives
    return {g: r for g, r in drives.items() if g not in SCRIPTED_VISUAL_GROUPS}


@dataclass(frozen=True)
class _Obj:  # immutable copy of a world.WorldObject for the render thread (WorldStore is loop-only)
    id: int
    kind: str
    x: float
    y: float
    amount: float | None


class VisionLink:
    def __init__(self, base_url: str, scene: dict | None = None, send: bool = False, hz: float = DEFAULT_HZ,
                 looming: str = "scripted", client: httpx.Client | None = None) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.send = bool(send and self.base_url)
        self.hz = max(0.5, float(hz))
        self.looming = looming if looming in LOOMING_MODES else "scripted"
        self.renderer = EyeRenderer(scene)
        self._client = client
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._job = None  # (sim t, thorax pos, thorax quat, objects) latest pose handed over by the world loop
        self._last_t = None
        self.image: dict | None = None  # {t, left, right} quantised 0..255, last render
        self.renders = 0
        self.render_ms = 0.0  # smoothed
        self.posts_ok = 0
        self.posts_failed = 0
        self.last_post_wall: float | None = None
        self.last_mean_rate_hz: float | None = None  # brain's reply: mean photoreceptor rate
        self.last_active: int | None = None  # brain's reply: photoreceptors with a non-zero rate
        self._times: list[float] = []
        self._warned = False
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)

    @classmethod
    def from_env(cls, base_url: str, scene: dict | None) -> "VisionLink":
        return cls(base_url, scene, send=env_flag("VISION_TO_BRAIN"), hz=float(os.environ.get("VISION_HZ", DEFAULT_HZ)),
                   looming=looming_mode_from_env())

    # --- world loop side (asyncio thread): cheap, never blocks --------------------------------
    def submit(self, frame: dict, thorax: int, objects) -> None:
        job = (float(frame["t"]), np.array(frame["pos"][thorax], dtype=float), np.array(frame["quat"][thorax], dtype=float),
               tuple(_Obj(o.id, o.kind, float(o.x), float(o.y), o.amount) for o in objects))
        with self._lock:
            self._job = job
        self._wake.set()

    def filter(self, drives: dict) -> dict:
        return filter_drives(drives, self.looming)

    # --- thread ---------------------------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2)
        if self.send:
            try:
                self.client().delete("/vision")  # leave no stale image driving the photoreceptors
            except (httpx.HTTPError, OSError):
                pass
        if self._client is not None:
            self._client.close()

    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=REQUEST_TIMEOUT_S)
        return self._client

    def _run(self) -> None:
        period = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.step()
            except Exception:  # noqa: BLE001 - keep rendering; the next pose retries
                log.exception("vision step failed")
            rest = period - (time.monotonic() - t0)
            if rest > 0:
                self._stop.wait(rest)

    def step(self) -> bool:
        """Render the latest pose once (and POST it). False when there was nothing new. Also used by the tests."""
        with self._lock:
            job, self._job = self._job, None
        if job is None:
            return False
        t, pos, quat, objs = job
        if self._last_t is not None and t == self._last_t and self.image is not None and not self.send:
            return False
        self._last_t = t
        r0 = time.perf_counter()
        lum = self.renderer.render(pos, quat, objs)
        img = {"t": round(t, 4), "left": quantise(lum["left"]), "right": quantise(lum["right"])}
        ms = (time.perf_counter() - r0) * 1000.0
        self.render_ms = ms if self.renders == 0 else 0.9 * self.render_ms + 0.1 * ms
        self.renders += 1
        now = time.monotonic()
        self._times = [x for x in self._times if now - x <= 2.0] + [now]
        self.image = img
        if self.send:
            self._post(img)
        return True

    def _post(self, img: dict) -> None:
        try:
            r = self.client().post("/vision", json={"left": img["left"], "right": img["right"], "max": 255.0})
            r.raise_for_status()
            body = r.json()
            self.last_mean_rate_hz = body.get("mean_rate_hz")
            self.last_active = body.get("active")
            self.posts_ok += 1
            self.last_post_wall = time.monotonic()
            if self._warned:
                log.info("brain /vision reachable again")
                self._warned = False
        except (httpx.HTTPError, OSError, ValueError) as exc:
            self.posts_failed += 1
            if not self._warned:
                log.warning("brain /vision POST failed (%s); vision is not delivered", type(exc).__name__)
                self._warned = True

    def status(self) -> dict:
        ts = self._times
        hz = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) >= 2 and ts[-1] > ts[0] else 0.0
        age = None if self.last_post_wall is None else round(time.monotonic() - self.last_post_wall, 3)
        return {
            "to_brain": self.send,
            "looming_drive": self.looming,
            "hz_target": self.hz,
            "hz": round(hz, 2),
            "renders": self.renders,
            "render_ms": round(self.render_ms, 2),
            "posts_ok": self.posts_ok,
            "posts_failed": self.posts_failed,
            "last_post_age_s": age,
            "brain_photoreceptors_active": self.last_active,
            "brain_photoreceptor_mean_hz": self.last_mean_rate_hz,
        }
