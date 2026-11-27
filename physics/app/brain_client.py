"""Brain WebSocket client: reads motor output from the brain and feeds CommandState.

Reconnects forever with capped backoff. While disconnected the command state
falls back to hold (zero drive), so the fly stands still without a brain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from .command import CommandState

log = logging.getLogger("physics.brain")


async def brain_loop(
    url: str,
    commands: CommandState,
    *,
    retry_min: float = 0.5,
    retry_max: float = 5.0,
) -> None:
    import websockets  # imported here so the module loads without a network stack

    delay = retry_min
    while True:
        try:
            async with websockets.connect(
                url,
                open_timeout=3,
                ping_interval=10,
                ping_timeout=10,
                max_size=2**22,
            ) as ws:
                commands.set_brain_connected(True)
                log.info("connected to brain at %s", url)
                delay = retry_min
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(msg, dict) and isinstance(msg.get("group_rates"), dict):
                        commands.set_group_rates(msg["group_rates"], time.monotonic())
                    motor = msg.get("motor") if isinstance(msg, dict) else None
                    if not isinstance(motor, dict):
                        continue
                    try:
                        commands.set_brain(
                            motor.get("forward", 0.0),
                            motor.get("turn", 0.0),
                            time.monotonic(),
                        )
                    except (TypeError, ValueError):
                        continue
        except asyncio.CancelledError:
            commands.set_brain_connected(False)
            raise
        except Exception as exc:  # noqa: BLE001 - any failure means "retry"
            log.warning("brain connection failed (%s: %s); holding still", type(exc).__name__, exc)
        commands.set_brain_connected(False)
        await asyncio.sleep(delay)
        delay = min(delay * 2, retry_max)
