"""Pure helpers: motor command -> per-side descending drive, and command source selection.

Motor command: forward in [0, 1], turn in [-1, 1] (positive = turn left).
The tripod CPG controller takes (left, right) drive in [0, 1.2] per side. Drive
is kept non-negative: a backward-driven side (used for spinning in place) made
the fly topple in testing, so turning is a differential of forward speed.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

MAX_DRIVE = 1.2  # amplitude the flygym tutorial controller is tuned for
BRAIN_STALE_S = 1.0  # brain motor frames older than this count as "no brain"

SOURCE_MANUAL = "manual"
SOURCE_WORLD = "world"
SOURCE_BRAIN = "brain"
SOURCE_HOLD = "hold"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clean_command(forward: float, turn: float) -> tuple[float, float]:
    """Clamp to forward [0, 1] and turn [-1, 1]; non-finite input means stand still."""
    f = float(forward)
    t = float(turn)
    if not (math.isfinite(f) and math.isfinite(t)):
        return 0.0, 0.0
    return _clamp(f, 0.0, 1.0), _clamp(t, -1.0, 1.0)


def descending_drive(forward: float, turn: float) -> tuple[float, float]:
    """Return (left, right) drive for HybridTurningController.step.

    left = forward - turn, right = forward + turn, each clipped to [0, 1] and
    scaled by MAX_DRIVE. A positive turn weakens the left side and strengthens
    the right side, which yaws the fly counter-clockwise (left, as seen from above).
    With forward 0 and a nonzero turn only one side steps (a slow pivot).
    """
    f, t = clean_command(forward, turn)
    left = _clamp(f - t, 0.0, 1.0) * MAX_DRIVE
    right = _clamp(f + t, 0.0, 1.0) * MAX_DRIVE
    return left, right


@dataclass(frozen=True)
class Motor:
    forward: float = 0.0
    turn: float = 0.0


class CommandState:
    """Thread-safe choice between manual override, brain motor output and hold.

    Priority: manual (POST /command, sticky until released) > world (scripted
    steering with brain drive, while world objects exist; see world.py/behavior.py)
    > fresh brain motor frames while the brain socket is connected > hold (zero command).
    """

    def __init__(self, stale_s: float = BRAIN_STALE_S) -> None:
        self._lock = threading.Lock()
        self._stale_s = stale_s
        self._manual: Motor | None = None
        self._brain: Motor | None = None
        self._brain_at: float | None = None
        self._connected = False
        self._world: Motor | None = None
        self._world_at: float | None = None
        self._group_rates: dict[str, float] = {}
        self._group_rates_at: float | None = None

    def set_manual(self, forward: float, turn: float) -> None:
        f, t = clean_command(forward, turn)
        with self._lock:
            self._manual = Motor(f, t)

    def release_manual(self) -> None:
        with self._lock:
            self._manual = None

    def set_world(self, forward: float, turn: float, now: float) -> None:
        f, t = clean_command(forward, turn)
        with self._lock:
            self._world = Motor(f, t)
            self._world_at = now

    def clear_world(self) -> None:
        with self._lock:
            self._world = None
            self._world_at = None

    def brain_forward(self, now: float) -> float | None:
        """Brain forward drive if a fresh brain frame exists, else None (used as the drive gain)."""
        with self._lock:
            if self._connected and self._brain is not None and self._brain_at is not None \
                    and now - self._brain_at <= self._stale_s:
                return self._brain.forward
            return None

    def brain_motor(self, now: float) -> tuple[float, float] | None:
        """(forward, turn) of the freshest brain frame (brain turn: + = right), else None."""
        with self._lock:
            if self._connected and self._brain is not None and self._brain_at is not None \
                    and now - self._brain_at <= self._stale_s:
                return (self._brain.forward, self._brain.turn)
            return None

    def set_group_rates(self, rates: dict, now: float) -> None:
        """Latest per-group firing rates (Hz) from the brain stream (e.g. groom_adn1)."""
        clean = {str(k): float(v) for k, v in rates.items() if isinstance(v, (int, float))}
        with self._lock:
            self._group_rates = clean
            self._group_rates_at = now

    def group_rate(self, name: str, now: float) -> float | None:
        """Rate of a brain group from a fresh frame (<= BRAIN_STALE_S old, socket connected), else None."""
        with self._lock:
            if self._connected and self._group_rates_at is not None and now - self._group_rates_at <= self._stale_s:
                return self._group_rates.get(name)
            return None

    def set_brain_connected(self, connected: bool) -> None:
        with self._lock:
            self._connected = connected
            self._group_rates = {}
            self._group_rates_at = None
            if not connected:
                self._brain = None
                self._brain_at = None

    def set_brain(self, forward: float, turn: float, now: float) -> None:
        f, t = clean_command(forward, turn)
        with self._lock:
            self._brain = Motor(f, t)
            self._brain_at = now

    def effective(self, now: float) -> tuple[Motor, str]:
        with self._lock:
            if self._manual is not None:
                return self._manual, SOURCE_MANUAL
            if (
                self._world is not None
                and self._world_at is not None
                and now - self._world_at <= self._stale_s
            ):
                return self._world, SOURCE_WORLD
            if (
                self._connected
                and self._brain is not None
                and self._brain_at is not None
                and now - self._brain_at <= self._stale_s
            ):
                return self._brain, SOURCE_BRAIN
            return Motor(), SOURCE_HOLD

    def status(self, now: float) -> dict:
        with self._lock:
            manual = self._manual
            age = None if self._brain_at is None else round(now - self._brain_at, 3)
            connected = self._connected
        return {
            "manual": None if manual is None else {"forward": manual.forward, "turn": manual.turn},
            "brain_connected": connected,
            "brain_last_motor_age_s": age,
        }
