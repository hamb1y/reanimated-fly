"""Background thread that steps the fly, paced to the speed target (1.0 = real time).

Speed: `target_speed` is sim seconds per wall second (the /speed factor, 0.1..2). Pacing re-bases
whenever the target changes, so a change never causes a catch-up burst or a stall.
Capacity: sim seconds per second of busy thread time over the last window, i.e. what the sim could
do flat out on this host (busy time excludes the pacing sleeps).

Fallback: when the sim cannot keep up (speed factor below FALLBACK_SPEED times the target, capped at
1.0, for FALLBACK_WINDOWS windows in a row), it moves to the next coarser fidelity level in
flysim.FIDELITY_LEVELS. It never moves back up on its own, and the fly keeps its state through the switch.
"""

from __future__ import annotations

import logging
import threading
import time

from .command import CommandState
from .flysim import FIDELITY_LEVELS, FlySim

log = logging.getLogger("physics.runner")

PUBLISH_HZ = 120  # pose snapshots per second of sim time; the legs step at ~12 Hz, so 30 Hz aliased into visible jitter
SPEED_WINDOW_S = 2.0
FALLBACK_SPEED = 0.9  # speed factor below this (of the target, capped at 1) counts as "cannot keep up"
FALLBACK_WINDOWS = 3  # consecutive slow windows (6 s) before stepping down a level


class SimRunner:
    def __init__(self, flysim: FlySim, commands: CommandState, target_speed: float = 1.0) -> None:
        self.flysim = flysim
        self.commands = commands
        self.target_speed = target_speed
        self.speed_factor = 0.0  # sim seconds per wall second, over the last window
        self.capacity: float | None = None  # sim seconds per busy second, over the last window
        self.fallbacks = 0  # how many times the fidelity was stepped down
        self._lock = threading.Lock()
        self._reset_waiters: list[threading.Event] = []
        self._frame: dict | None = None
        self._stop = threading.Event()
        self._rebase = False
        self._thread = threading.Thread(target=self._run, name="fly-sim", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def set_target(self, factor: float) -> None:
        with self._lock:
            self.target_speed = float(factor)
            self._rebase = True

    def latest(self) -> dict | None:
        """Most recent published snapshot: {t, pos, quat, motor, source}."""
        return self._frame

    def request_reset(self) -> threading.Event:
        ev = threading.Event()
        with self._lock:
            self._reset_waiters.append(ev)
        return ev

    def _publish(self, motor, source: str) -> None:
        pose = self.flysim.pose()
        pose["motor"] = motor
        pose["source"] = source
        pose["groom"] = self.flysim.groom.weight  # 0..1 blend of the replayed foreleg grooming (app/groom.py)
        self._frame = pose

    def _publish_current(self, now: float) -> None:
        motor, source = self.commands.effective(now)
        self._publish(motor, source)

    def _maybe_fallback(self, slow_windows: int) -> int:
        """Step to a coarser fidelity level after FALLBACK_WINDOWS slow windows."""
        if slow_windows < FALLBACK_WINDOWS or self.flysim.level >= len(FIDELITY_LEVELS) - 1:
            return slow_windows
        new_level = self.flysim.level + 1
        log.warning(
            "speed %.2fx (target %.2f) below %.2f for %d windows; fidelity level %d -> %d",
            self.speed_factor, self.target_speed, FALLBACK_SPEED, slow_windows, self.flysim.level, new_level,
        )
        self.flysim.set_level(new_level)
        self.fallbacks += 1
        return 0

    def _run(self) -> None:
        self._publish_current(time.monotonic())
        wall0 = win_wall = time.monotonic()
        sim0 = win_sim = self.flysim.sim_time
        last_pub_sim = sim0
        busy_win = 0.0
        slow_windows = 0
        while not self._stop.is_set():
            with self._lock:
                waiters, self._reset_waiters = self._reset_waiters, []
                rebase, self._rebase = self._rebase, False
                target = self.target_speed
            if waiters:
                self.flysim.reset()
                wall0 = win_wall = time.monotonic()
                sim0 = win_sim = self.flysim.sim_time
                self._publish_current(wall0)
                for ev in waiters:
                    ev.set()
                last_pub_sim = sim0
                busy_win = 0.0
                continue
            if rebase:
                wall0 = time.monotonic()
                sim0 = self.flysim.sim_time
                last_pub_sim = min(last_pub_sim, sim0)

            motor, source = self.commands.effective(time.monotonic())
            t_busy = time.monotonic()
            try:
                self.flysim.advance(motor.forward, motor.turn, 1)  # one controller tick (2 ms)
            except Exception:  # noqa: BLE001
                log.exception("physics step failed; stopping sim thread")
                return

            sim_t = self.flysim.sim_time
            if sim_t - last_pub_sim >= 1.0 / PUBLISH_HZ - 1e-9:
                self._publish(motor, source)
                last_pub_sim = sim_t
            now = time.monotonic()
            busy_win += now - t_busy

            ahead = (sim_t - sim0) - (now - wall0) * target
            if ahead > 0:
                time.sleep(min(ahead / target, 0.02))
                now = time.monotonic()

            if now - win_wall >= SPEED_WINDOW_S:
                self.speed_factor = (sim_t - win_sim) / (now - win_wall)
                self.capacity = (sim_t - win_sim) / busy_win if busy_win > 0 else None
                slow = self.speed_factor < FALLBACK_SPEED * min(target, 1.0)
                slow_windows = slow_windows + 1 if slow else 0
                slow_windows = self._maybe_fallback(slow_windows)
                win_wall, win_sim, busy_win = now, sim_t, 0.0
