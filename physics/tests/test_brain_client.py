import asyncio
import json
import socket
import time

import websockets

from app.brain_client import brain_loop
from app.command import CommandState


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def test_brain_frames_feed_command_and_unreachable_brain_holds():
    async def main():
        async def handler(ws):
            await ws.send(json.dumps({"t_ms": 1, "motor": {"forward": 0.7, "turn": -0.2}}))
            await asyncio.sleep(5)

        cs = CommandState()
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            task = asyncio.create_task(
                brain_loop(f"ws://127.0.0.1:{port}/stream", cs, retry_min=0.05, retry_max=0.1)
            )
            ok = await _wait_for(lambda: cs.effective(time.monotonic())[1] == "brain")
            assert ok
            motor, _ = cs.effective(time.monotonic())
            assert (motor.forward, motor.turn) == (0.7, -0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert cs.status(time.monotonic())["brain_connected"] is False

        # Nothing listening: the client keeps retrying and the fly holds still.
        cs2 = CommandState()
        task = asyncio.create_task(
            brain_loop(f"ws://127.0.0.1:{_free_port()}/stream", cs2, retry_min=0.05, retry_max=0.1)
        )
        await asyncio.sleep(0.5)
        assert cs2.effective(time.monotonic())[1] == "hold"
        assert cs2.status(time.monotonic())["brain_connected"] is False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(main())


def test_group_rates_from_the_brain_stream_are_kept():
    async def main():
        async def handler(ws):
            await ws.send(json.dumps({"t_ms": 1, "motor": {"forward": 0.1, "turn": 0.0}, "group_rates": {"groom_adn1": 5.6}}))
            await ws.send(json.dumps({"t_ms": 2, "group_rates": {"groom_adn1": 7.0}}))  # no motor: still parsed
            await asyncio.sleep(5)

        cs = CommandState()
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            task = asyncio.create_task(brain_loop(f"ws://127.0.0.1:{port}/stream", cs, retry_min=0.05, retry_max=0.1))
            assert await _wait_for(lambda: cs.group_rate("groom_adn1", time.monotonic()) == 7.0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(main())
