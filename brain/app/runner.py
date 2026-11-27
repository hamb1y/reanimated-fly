"""Free-running simulation thread + stream frame builder (rolling rates, motor mapping)."""
import math
import threading
import time
from collections import deque

import numpy as np

from .model import DT, RATE_WINDOW_STEPS

CHUNK = 320  # steps (32 ms sim) per run() call: one device sync per ~30 Hz frame; multiple of BLOCK (20)
# Motor mapping constants (documented in README): per-neuron mean rate (Hz) of the readout groups
# that maps to full command: 200 Hz == max_rate of the DNRateDecoder in the rojas reference (brain_body_bridge.py).
MOTOR_NORM_HZ = {"forward": 200.0, "turn": 200.0}
MOTOR_SMOOTH_TAU_S = 0.15
MAX_SPIKES_PER_FRAME = 2000


class Runner:
    def __init__(self, brain, groups, speed_target=1.0):
        self.brain = brain
        self.groups = groups
        self.lock = threading.Lock()
        self.speed_target = speed_target  # sim_ms per wall_ms cap; 0 = as fast as possible
        self.stop_evt = threading.Event()
        self.thread = None
        self.samples = deque(maxlen=400)  # (wall_end, sim_ms_done, busy_s)
        self.sim_ms_total = 0.0
        self.stim_groups = {}
        self.silent_groups = []
        # frame state
        self._motor = {"forward": 0.0, "turn": 0.0}
        self._last_frame_wall = time.perf_counter()
        self._rng = np.random.default_rng(1)

    # ---- thread -------------------------------------------------------
    def start(self):
        self.thread = threading.Thread(target=self._loop, name="sim", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _loop(self):
        deadline = time.perf_counter()
        sim_ms = CHUNK * DT
        while not self.stop_evt.is_set():
            t0 = time.perf_counter()
            with self.lock:
                self.brain.run(CHUNK)  # returns after the device has finished the chunk
            now = time.perf_counter()
            busy = now - t0
            self.sim_ms_total += sim_ms
            target = self.speed_target
            if target > 0:
                deadline += sim_ms / 1000.0 / target
                if deadline < now - 0.25:  # fell behind: do not try to catch up in a burst
                    deadline = now
                if deadline > now:
                    time.sleep(deadline - now)
                    now = time.perf_counter()
            else:
                deadline = now
            self.samples.append((now, self.sim_ms_total, busy))

    def speed(self):
        """(sim_speed, sim_capacity) over the last ~2 s: sim_ms per wall_ms, and sim_ms per busy wall_ms."""
        s = list(self.samples)
        if len(s) < 3:
            return 0.0, 0.0
        now = s[-1][0]
        s = [x for x in s if now - x[0] <= 2.0] or s[-3:]
        if len(s) < 2:
            return 0.0, 0.0
        dwall = s[-1][0] - s[0][0]
        dsim = s[-1][1] - s[0][1]
        busy = sum(x[2] for x in s[1:])
        return (dsim / (dwall * 1000.0) if dwall > 0 else 0.0, dsim / (busy * 1000.0) if busy > 0 else 0.0)

    # ---- commands (all under the sim lock) ----------------------------------
    def stimulate(self, idx, rate_hz, label=None):
        with self.lock:
            if rate_hz > 0:
                self.brain.stimulate(idx, rate_hz)
                if label:
                    self.stim_groups[label] = rate_hz
            else:
                self.brain.unstimulate(idx)
                if label:
                    self.stim_groups.pop(label, None)

    def drive(self, channel, idx, rate_hz):
        """Replace a per-neuron drive channel (see BrainModel.set_drive). Returns the number of driven neurons."""
        with self.lock:
            self.brain.set_drive(channel, idx, rate_hz)
            d = self.brain.drive.get(channel)
        return 0 if d is None else int(len(d[0]))

    def silence(self, idx, label=None):
        with self.lock:
            self.brain.silence(idx)
            if label and label not in self.silent_groups:
                self.silent_groups.append(label)

    def clear(self):
        with self.lock:
            self.brain.clear()
            self.stim_groups.clear()
            self.silent_groups.clear()

    def reset(self):
        with self.lock:
            self.brain.reset()
            self._motor = {"forward": 0.0, "turn": 0.0}

    def set_params(self, upd):
        with self.lock:
            self.brain.set_params(upd)

    # ---- stream -----------------------------------------------------------------
    def frame(self):
        b = self.brain
        with self.lock:  # snapshot under the sim lock (the sim thread enqueues work under it)
            step = b.t
            win = b.window_counts()
            new = b.take_frame_counts()
            params = dict(b.params)
        win_s = min(step, RATE_WINDOW_STEPS) * DT / 1000.0
        rates = {}
        for name, g in self.groups.items():
            k = len(g["idx"])
            c = int(win[g["idx"]].sum()) if k else 0
            rates[name] = round(c / (k * win_s), 2) if win_s > 0 and k else 0.0
        fwd = rates.get("forward_walk", 0.0) / MOTOR_NORM_HZ["forward"]
        turn = (rates.get("turn_right", 0.0) - rates.get("turn_left", 0.0)) / MOTOR_NORM_HZ["turn"]
        fwd = min(max(fwd, 0.0), 1.0)
        turn = min(max(turn, -1.0), 1.0)
        now = time.perf_counter()
        a = 1.0 - math.exp(-(now - self._last_frame_wall) / MOTOR_SMOOTH_TAU_S)
        self._last_frame_wall = now
        self._motor["forward"] += a * (fwd - self._motor["forward"])
        self._motor["turn"] += a * (turn - self._motor["turn"])
        # spikes since the previous frame (one entry per spike, uniformly subsampled to MAX_SPIKES_PER_FRAME)
        nz = np.flatnonzero(new)
        sp = np.repeat(nz, new[nz])
        n_spikes = int(sp.size)  # true count, before the display subsample
        if sp.size > MAX_SPIKES_PER_FRAME:
            sp = self._rng.choice(sp, MAX_SPIKES_PER_FRAME, replace=False)
        speed, cap = self.speed()
        return {
            "t_ms": round(step * DT, 1),
            "motor": {"forward": round(self._motor["forward"], 4), "turn": round(self._motor["turn"], 4)},
            "group_rates": rates,
            "spikes": sp.tolist(),
            "n_spikes": n_spikes,
            "params": params,
            "sim_speed": round(speed, 3),
        }
