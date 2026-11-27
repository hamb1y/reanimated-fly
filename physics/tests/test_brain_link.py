"""BrainLink: diffed fire-and-forget stimulation, zeroing, reconcile and speed sync (mock brain)."""

import asyncio
import json

import httpx

from app.brain_link import BrainLink


class FakeBrain:
    def __init__(self, fail=False):
        self.calls = []
        self.groups = {}
        self.speed = 1.0
        self.fail = fail

    def handler(self, req: httpx.Request) -> httpx.Response:
        if self.fail:
            raise httpx.ConnectError("refused", request=req)
        body = json.loads(req.content) if req.content else None
        self.calls.append((req.method, req.url.path, body))
        if req.url.path == "/stimulate":
            if body["rate_hz"] > 0:
                self.groups[body["group"]] = body["rate_hz"]
            else:
                self.groups.pop(body["group"], None)
            return httpx.Response(200, json={"ok": True})
        if req.url.path == "/speed":
            self.speed = body["target"]
            return httpx.Response(200, json={"ok": True})
        if req.url.path == "/health":
            return httpx.Response(200, json={"sim_speed": 1.0, "sim_capacity": 2.0, "speed_target": self.speed})
        if req.url.path == "/state":
            return httpx.Response(200, json={"stimulated_groups": dict(self.groups)})
        return httpx.Response(404)

    def posts(self):
        return [c for c in self.calls if c[1] == "/stimulate"]


def _link(fake):
    client = httpx.AsyncClient(base_url="http://brain.test", transport=httpx.MockTransport(fake.handler))
    return BrainLink("http://brain.test", client=client)


async def _settle(link):
    for _ in range(20):
        if not link._pending and not link._tasks:
            return
        await asyncio.sleep(0.01)


def test_only_changed_owned_groups_are_posted_and_zeroed_when_dropped():
    async def main():
        fake = FakeBrain()
        link = _link(fake)
        assert link.send_desired({"sugar_grn": 96.0, "forward_walk": 50.0}) == 1  # forward_walk not owned
        await _settle(link)
        assert fake.groups == {"sugar_grn": 96.0} and link.sent == {"sugar_grn": 96.0}
        assert link.send_desired({"sugar_grn": 96.4}) == 0  # within 1 Hz: no traffic
        assert link.send_desired({"sugar_grn": 150.0}) == 1
        await _settle(link)
        assert fake.groups["sugar_grn"] == 150.0
        link.send_desired({})  # no longer driven: zeroed in the brain
        await _settle(link)
        assert fake.groups == {} and link.sent == {}
        assert [p[2] for p in fake.posts()][-1] == {"group": "sugar_grn", "rate_hz": 0.0}
        assert link.send_desired({}) == 0
        await link.close()

    asyncio.run(main())


def test_unreachable_brain_marks_offline_and_retries_later():
    async def main():
        fake = FakeBrain(fail=True)
        link = _link(fake)
        link.send_desired({"looming": 200.0})
        await _settle(link)
        assert link.online is False and link.sent == {}  # not believed sent
        fake.fail = False
        link.send_desired({"looming": 200.0})  # the next tick retries
        await _settle(link)
        assert link.online is True and fake.groups == {"looming": 200.0}
        await link.close()

    asyncio.run(main())


def test_reconcile_reads_brain_state_and_syncs_speed():
    async def main():
        fake = FakeBrain()
        fake.groups = {"escape_gf": 100.0, "bitter_grn": 100.0}  # bitter_grn is not ours: left alone
        link = _link(fake)
        ok = await link.reconcile(0.5)
        assert ok and link.sent == {"escape_gf": 100.0}
        assert fake.speed == 0.5  # PUT /speed sent because the brain was at 1.0
        assert link.sim_capacity == 2.0
        assert await link.reconcile(0.5) is True
        assert len([c for c in fake.calls if c[1] == "/speed"]) == 1  # already in sync: no second PUT
        assert await link.set_speed(2.0) is True and fake.speed == 2.0
        await link.close()

    asyncio.run(main())


def test_disabled_link_does_nothing():
    async def main():
        link = BrainLink("")
        assert link.enabled is False
        assert link.send_desired({"sugar_grn": 100.0}) == 0
        assert await link.reconcile(1.0) is False
        assert await link.set_speed(1.0) is False

    asyncio.run(main())


def test_jo_mechano_is_an_owned_group():
    from app.brain_link import OWNED

    assert "jo_mechano" in OWNED
