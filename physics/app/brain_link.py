"""Fire-and-forget HTTP link from physics to the brain (POST /stimulate, PUT /speed, GET /health, /state).

Only the groups the world drives are touched (OWNED). The link keeps `sent`, the rates the brain is
believed to hold, and sends a POST only for groups whose rate changed. A group that is no longer
driven is zeroed (rate 0 removes the stimulation in the brain). Nothing is awaited by the physics
loop: send_desired() schedules tasks and returns at once. A failed POST is retried on the next tick.
Every RECONCILE_S, `sent` is re-read from the brain, so a brain restart or a /clear is repaired.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger("physics.brain_http")
logging.getLogger("httpx").setLevel(logging.WARNING)  # the 2 s reconcile would log every request

OWNED = ("sugar_grn", "looming", "lplc2", "escape_gf", "feed_mn9", "p9", "looming_left", "looming_right", "lplc2_left", "lplc2_right", "jo_mechano",
         "orn_food_left", "orn_food_right", "orn_cva_left", "orn_cva_right")
RATE_EPS_HZ = 1.0  # rate changes smaller than this are not re-sent
RECONCILE_S = 2.0
REQUEST_TIMEOUT_S = 0.5


class BrainLink:
    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.enabled = bool(self.base_url)
        self._client = client
        self.sent: dict[str, float] = {}
        self._pending: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self.online = False
        self.speed_target: float | None = None  # brain's own speed cap, as last read
        self.sim_speed: float | None = None
        self.sim_capacity: float | None = None
        self.posts_ok = 0  # counters for /health and the README measurements
        self.posts_failed = 0
        self._warned = False

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=REQUEST_TIMEOUT_S)
        return self._client

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        if self._client is not None:
            await self._client.aclose()

    def _mark(self, ok: bool, exc: Exception | None = None) -> None:
        if ok and not self.online and self._warned:
            log.info("brain HTTP reachable again at %s", self.base_url)
        if not ok and (self.online or not self._warned):
            log.warning("brain HTTP %s unavailable (%s); world drives are not delivered",
                        self.base_url, type(exc).__name__ if exc else "error")
            self._warned = True
        self.online = ok

    def send_desired(self, rates: dict[str, float]) -> int:
        """Schedule POSTs for owned groups whose brain rate differs from `rates`. Returns the count."""
        if not self.enabled:
            return 0
        n = 0
        for g in OWNED:
            want = float(rates.get(g, 0.0))
            have = self.sent.get(g, 0.0)
            if g in self._pending:
                continue
            if abs(want - have) < RATE_EPS_HZ and (want > 0) == (have > 0):
                continue
            if want == 0.0 and have == 0.0:
                continue
            self._pending.add(g)
            task = asyncio.get_running_loop().create_task(self._stimulate(g, want))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            n += 1
        return n

    async def _stimulate(self, group: str, rate: float) -> None:
        try:
            r = await self.client().post("/stimulate", json={"group": group, "rate_hz": rate})
            r.raise_for_status()
            if rate > 0:
                self.sent[group] = rate
            else:
                self.sent.pop(group, None)
            self.posts_ok += 1
            self._mark(True)
        except (httpx.HTTPError, OSError) as exc:
            self.posts_failed += 1
            self._mark(False, exc)
        finally:
            self._pending.discard(group)

    async def reconcile(self, factor: float) -> bool:
        """Read the brain's state and speed; push our speed factor if it differs. True when in sync."""
        if not self.enabled:
            return False
        try:
            c = self.client()
            h = (await c.get("/health")).json()
            st = (await c.get("/state")).json()
            self.sim_speed = h.get("sim_speed")
            self.sim_capacity = h.get("sim_capacity")
            self.speed_target = h.get("speed_target")
            self.sent = {g: float(v) for g, v in st.get("stimulated_groups", {}).items() if g in OWNED}
            self._mark(True)
            if self.speed_target is None or abs(float(self.speed_target) - factor) > 1e-6:
                return await self.set_speed(factor)
            return True
        except (httpx.HTTPError, OSError, ValueError) as exc:
            self._mark(False, exc)
            return False

    async def set_speed(self, factor: float) -> bool:
        if not self.enabled:
            return False
        try:
            r = await self.client().put("/speed", json={"target": factor})
            r.raise_for_status()
            self.speed_target = float(factor)
            self._mark(True)
            return True
        except (httpx.HTTPError, OSError) as exc:
            self._mark(False, exc)
            return False
