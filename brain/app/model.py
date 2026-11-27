"""Whole-brain LIF of the FlyWire connectome (reference: reference/run_pytorch.py).

Two backends with identical semantics, selected by BRAIN_DEVICE (auto|cuda|cpu):
  * cpu:  eager torch, event-driven scatter over the CSC columns of the neurons that spiked, one step at a time.
  * cuda: app/cuda_lif.py, one Triton kernel per 20-step block + a fixed-shape synaptic scatter, captured as a
          CUDA graph; no host sync inside a block.

The update order is a direct transcription of the reference AlphaLIF / TorchModel step (dt 0.1 ms, float32):

  * delay: a spike at step s reaches the synaptic input at step s+20 (delay line 19 slots + 1). Steps are
    numbered from 1. A block of BLOCK = 20 steps therefore depends only on spikes of earlier blocks.
  * conductance: g_{t+1} = g_t * (1 - dt/tauSyn) + gate * wScale * I_t; spiking neurons get g = 0.
  * refractory gate: a neuron that spiked at step s gets no synaptic input at steps s+1 .. s+22
    (reference: refrac >= refrac_steps). Stimulated neurons are exempt (reference: exc_indices, refrac 0).
  * membrane (u = v - v_rest): u += wScale*250*Bernoulli(rate*dt) (Poisson kick); u += dt/tauMem*(g - u);
    spike if u > v_th - v_rest; reset u = v_reset - v_rest.

Notes (also in README.md):
  * silencing is a mask (silenced neurons never spike, their conductance/membrane are held at 0 and they
    receive no input). The connectome itself is not edited.
  * the Poisson background (poisson_rate_hz) is an addition on top of the reference, delivered as kicks to
    random neurons with the same kick size as stimulation. The CPU backend draws a Poisson count of kicks,
    the CUDA backend a Bernoulli draw per neuron per step (same mean, at most one kick per neuron per step).
"""
import logging
from collections import deque

import numpy as np
import torch

log = logging.getLogger("brain")

DT = 0.1  # ms, fixed
V_REST = -52.0
T_DELAY = 1.8  # ms
SCALE_POISSON = 250.0
DELAY_SLOTS = 19  # int(tDelay / dt) + 1
BLOCK = DELAY_SLOTS + 1  # steps per block: one block of input only depends on spikes of earlier blocks
RATE_WINDOW_STEPS = 1000  # rolling 100 ms
RING = RATE_WINDOW_STEPS // BLOCK  # blocks kept for the rolling window
NEVER = -(1 << 40)  # "last spike" of a neuron that never spiked

DEFAULT_PARAMS = {
    "w_syn": 0.275,
    "tau_m_ms": 20.0,
    "v_th_mV": -45.0,
    "v_reset_mV": -52.0,
    "tau_syn_ms": 5.0,
    "t_ref_ms": 2.2,
    "poisson_rate_hz": 0.0,
    "dt_ms": DT,
}
PARAM_META = {
    "w_syn": {"min": 0.0, "max": 2.0, "step": 0.005, "unit": "x", "desc": "global synaptic weight scale (reference wScale)"},
    "tau_m_ms": {"min": 1.0, "max": 100.0, "step": 0.5, "unit": "ms", "desc": "membrane time constant (reference 20)"},
    "v_th_mV": {"min": -60.0, "max": -30.0, "step": 0.1, "unit": "mV", "desc": "spike threshold (reference -45)"},
    "v_reset_mV": {"min": -70.0, "max": -40.0, "step": 0.1, "unit": "mV", "desc": "reset potential (reference -52)"},
    "tau_syn_ms": {"min": 0.5, "max": 50.0, "step": 0.1, "unit": "ms", "desc": "synaptic conductance decay (reference 5)"},
    "t_ref_ms": {"min": 0.0, "max": 10.0, "step": 0.1, "unit": "ms", "desc": "refractory period, gates synaptic input (reference 2.2)"},
    "poisson_rate_hz": {"min": 0.0, "max": 20.0, "step": 0.01, "unit": "Hz", "desc": "background Poisson drive on every neuron (0 = off)"},
    "dt_ms": {"read_only": True, "unit": "ms", "desc": "integration step (fixed)"},
}


def roll_window(ring, win, frame, slot, cnt):
    """Add one block of per-neuron spike counts (cnt) to the rolling window and the frame counter.

    ring[slot] holds the block that leaves the window. Shapes and ops are fixed, so this is graph-capturable.
    """
    win.sub_(ring.index_select(0, slot)[0])
    ring.index_copy_(0, slot, cnt.unsqueeze(0))
    win.add_(cnt)
    frame.add_(cnt)
    slot.add_(1).remainder_(RING)


def resolve_device(requested):
    req = (requested or "auto").strip().lower()
    if req not in ("auto", "cuda", "cpu"):
        raise ValueError(f"BRAIN_DEVICE must be auto, cuda or cpu (got {requested!r})")
    return req


def make_brain(indptr, post, w, requested="auto", gpu_mem_mib=1024, record=False):
    """Build the Brain on the requested device. 'auto' uses CUDA when it works and falls back to CPU."""
    req = resolve_device(requested)
    if req == "cpu" or (req == "auto" and not torch.cuda.is_available()):
        return Brain(indptr, post, w, device="cpu", record=record)
    try:
        return Brain(indptr, post, w, device="cuda", gpu_mem_mib=gpu_mem_mib, record=record)
    except Exception as e:  # noqa: BLE001 - graceful fallback for 'auto' only
        if req == "cuda":
            raise
        log.warning("CUDA backend unavailable (%s); using CPU", e)
        return Brain(indptr, post, w, device="cpu", record=record)


class Brain:
    def __init__(self, indptr, post, w, seed=0, device="cpu", gpu_mem_mib=1024, record=False):
        self.n = len(indptr) - 1
        self.n_syn = int(len(post))
        self.device = device
        self.record = record
        self.gen = torch.Generator().manual_seed(seed)
        self.params = dict(DEFAULT_PARAMS)
        self.stim = {}  # neuron idx -> rate_hz (as requested; silenced ones are filtered in _sync_masks)
        self.drive = {}  # channel -> (idx int64[k], rate float64[k]): per-neuron rate vectors (set_drive, e.g. the eyes)
        self.silent = np.zeros(self.n, dtype=bool)
        self.raster = []  # (step, spiking indices) when record=True; cleared by reset() and by callers
        self.t = 0
        if device == "cuda":
            from .cuda_lif import CudaLif

            self._cuda = CudaLif(indptr, post, w, self.n, gpu_mem_mib)
            self._cpu_state = None
        elif device == "cpu":
            self.indptr = torch.from_numpy(np.asarray(indptr, dtype=np.int64))
            self.post = torch.from_numpy(np.asarray(post).astype(np.int64))
            self.w = torch.from_numpy(np.asarray(w, dtype=np.float32))
            self._cuda = None
            self.ring = torch.zeros(RING, self.n, dtype=torch.int32)
            self.win = torch.zeros(self.n, dtype=torch.int32)
            self.frame_cnt = torch.zeros(self.n, dtype=torch.int32)
            self.slot = torch.zeros(1, dtype=torch.int64)
            self._blk = torch.zeros(self.n, dtype=torch.int32)
        else:
            raise ValueError(f"unknown device {device!r}")
        self._derive()
        self._sync_masks()
        self.reset()

    # ---- parameters -------------------------------------------------
    def _derive(self):
        p = self.params
        self.k_m = DT / p["tau_m_ms"]
        self.g_decay = 1.0 - DT / p["tau_syn_ms"]
        self.thr = p["v_th_mV"] - V_REST
        self.u_reset = p["v_reset_mV"] - V_REST
        self.rs = int(round(p["t_ref_ms"] / DT))
        self.w_scale = float(p["w_syn"])
        self.kick = float(p["w_syn"]) * SCALE_POISSON
        self.p_bg = p["poisson_rate_hz"] * DT / 1000.0
        old = getattr(self, "hist", [])
        self.hist = deque(old, maxlen=max(self.rs, 0))  # spike sets of the last rs steps (CPU refractory window)
        if self._cuda is not None:
            self._cuda.set_params([self.k_m, self.g_decay, self.w_scale, self.kick, self.thr, self.u_reset, self.p_bg, self.rs])

    def set_params(self, upd):
        self.params.update(upd)
        self._derive()

    # ---- stimulation / silencing (masks; the connectome is untouched) ----
    def _sync_masks(self):
        self.sil_idx = torch.from_numpy(np.flatnonzero(self.silent).astype(np.int64))
        self.sil_mask = torch.from_numpy(self.silent.copy())
        if self.drive:  # vector channels: merged with the group/neuron stimulation, the larger rate wins per neuron
            dense = np.zeros(self.n, dtype=np.float64)
            for d_idx, d_rate in self.drive.values():
                np.maximum.at(dense, d_idx, d_rate)
            if self.stim:
                s_idx = np.fromiter(self.stim.keys(), dtype=np.int64, count=len(self.stim))
                s_rate = np.fromiter(self.stim.values(), dtype=np.float64, count=len(self.stim))
                np.maximum.at(dense, s_idx, s_rate)
            dense[self.silent] = 0.0
            idx = np.flatnonzero(dense > 0.0)
            rate = dense[idx]
        elif self.stim:
            idx = np.fromiter(self.stim.keys(), dtype=np.int64, count=len(self.stim))
            rate = np.fromiter(self.stim.values(), dtype=np.float64, count=len(self.stim))
            keep = ~self.silent[idx]
            idx, rate = idx[keep], rate[keep]
        else:
            idx, rate = np.zeros(0, np.int64), np.zeros(0, np.float64)
        self.stim_idx = torch.from_numpy(idx)
        self.stim_p = torch.from_numpy((rate * (DT / 1000.0)).astype(np.float32))
        self.stim_mask = torch.zeros(self.n, dtype=torch.bool)
        self.stim_mask[self.stim_idx] = True
        if self._cuda is not None:
            ps = np.zeros(self.n, dtype=np.float32)
            ps[idx] = (rate * (DT / 1000.0)).astype(np.float32)
            self._cuda.set_masks(ps, self.stim_mask.numpy().astype(np.int32), self.silent.astype(np.int32))

    def stimulate(self, idx, rate_hz):
        for i in np.asarray(idx).tolist():
            self.stim[i] = float(rate_hz)
        self._sync_masks()

    def unstimulate(self, idx):
        for i in np.asarray(idx).tolist():
            self.stim.pop(i, None)
        self._sync_masks()

    def set_drive(self, channel, idx, rate_hz):
        """Replace one per-neuron drive channel: idx (tensor indices) with rate_hz (same length, Hz).
        Rates <= 0 are dropped; an empty vector removes the channel. Vectorised (no per-neuron Python work)."""
        idx = np.asarray(idx, dtype=np.int64).ravel()
        rate = np.asarray(rate_hz, dtype=np.float64).ravel()
        if idx.shape != rate.shape:
            raise ValueError("idx and rate_hz must have the same length")
        keep = rate > 0.0
        if keep.any():
            self.drive[channel] = (idx[keep], rate[keep])
        else:
            self.drive.pop(channel, None)
        self._sync_masks()

    def silence(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        self.silent[idx] = True
        self._sync_masks()
        if self._cuda is not None:
            self._cuda.zero_neurons(idx)
        else:
            self.u[self.sil_idx] = 0
            self.g[self.sil_idx] = 0

    def clear(self):
        self.stim.clear()
        self.drive.clear()
        self.silent[:] = False
        self._sync_masks()

    # ---- state ------------------------------------------------------------
    def reset(self):
        """Clears membrane, conductance, pending input, spike history, the rolling window and the clock.
        Stimulation and silencing are kept (see clear())."""
        self.t = 0
        self.raster = []
        if self._cuda is not None:
            self._cuda.reset()
            return
        n = self.n
        self.u = torch.zeros(n)  # v - v_rest ; v0 == v_rest
        self.g = torch.zeros(n)  # synaptic conductance
        self.dl = torch.zeros(DELAY_SLOTS, n)  # delay line of raw recurrent inputs W @ spikes
        self.prev = torch.zeros(0, dtype=torch.int64)  # spikes of the last step
        self.hist.clear()
        self._blk.zero_()
        self.ring.zero_()
        self.win.zero_()
        self.frame_cnt.zero_()
        self.slot.zero_()

    # ---- integration --------------------------------------------------------
    def run(self, nsteps):
        """Advance nsteps (a multiple of BLOCK) and return once the device has finished them."""
        if nsteps % BLOCK:
            raise ValueError(f"nsteps must be a multiple of {BLOCK}")
        if self._cuda is not None:
            self._run_cuda(nsteps)
        else:
            self._run_cpu(nsteps)

    def _run_cuda(self, nsteps):
        c = self._cuda
        for b in range(nsteps // BLOCK):
            c.replay()
            if self.record:
                t0 = self.t + b * BLOCK
                s = c.spikes_host()
                for j in range(BLOCK):
                    idx = np.flatnonzero(s[j])
                    if idx.size:
                        self.raster.append((t0 + j + 1, idx))
        self.t += nsteps
        c.sync()

    @torch.no_grad()
    def _run_cpu(self, nsteps):
        u, g, dl = self.u, self.g, self.dl
        indptr, post, w = self.indptr, self.post, self.w
        has_sil = self.sil_idx.numel() > 0
        for _ in range(nsteps):
            self.t += 1
            t = self.t
            slot = dl[t % DELAY_SLOTS]  # holds I_{t-19} (read now, overwritten with I_t below)
            # refractory gate: spikes of steps t-1 .. t-rs block synaptic input (stimulated neurons exempt)
            self.hist.append(self.prev)
            if self.hist:
                blk = torch.cat(list(self.hist))
                if blk.numel():
                    blk = blk[~self.stim_mask[blk]]
                    slot[blk] = 0.0
            g_new = torch.add(g * self.g_decay, slot, alpha=self.w_scale)
            # Poisson kicks (stimulated neurons: Bernoulli(rate*dt) * wScale*250 mV)
            if self.stim_idx.numel():
                hit = torch.rand(self.stim_idx.numel(), generator=self.gen) < self.stim_p
                u.index_add_(0, self.stim_idx, hit.to(u.dtype) * self.kick)
            if self.p_bg > 0.0:
                nb = int(torch.poisson(torch.full((1,), self.n * self.p_bg))[0])
                if nb > 0:
                    bg = torch.randint(0, self.n, (nb,), generator=self.gen)
                    if has_sil:
                        bg = bg[~self.sil_mask[bg]]
                    u.index_add_(0, bg, torch.full((bg.numel(),), self.kick))
            # membrane: v += k*(g_old - (v - v_rest))
            u.lerp_(g, self.k_m)
            spk = torch.nonzero(u > self.thr).squeeze(1)
            if has_sil:
                spk = spk[~self.sil_mask[spk]]
                u[self.sil_idx] = 0.0
                g_new[self.sil_idx] = 0.0
            if spk.numel():
                u[spk] = self.u_reset
                g_new[spk] = 0.0  # conductance reset on spike
            # recurrent input of this step from the previous step's spikes -> delay slot t (read at t+19)
            slot.zero_()
            prev = self.prev
            if prev.numel():
                starts = indptr[prev]
                cnt = indptr[prev + 1] - starts
                tot = int(cnt.sum())
                if tot > 0:
                    off = torch.cumsum(cnt, 0) - cnt
                    gi = torch.arange(tot) + torch.repeat_interleave(starts - off, cnt)
                    slot.index_add_(0, post[gi], w[gi])
            self.g = g = g_new
            self.prev = spk
            if spk.numel():
                self._blk[spk] += 1
                if self.record:
                    self.raster.append((t, spk.numpy().copy()))
            if t % BLOCK == 0:
                roll_window(self.ring, self.win, self.frame_cnt, self.slot, self._blk)
                self._blk.zero_()

    # ---- readout for the stream -------------------------------------------------
    def window_counts(self):
        """Spike count per neuron over the last RATE_WINDOW_STEPS steps (blocks), as np.int32."""
        if self._cuda is not None:
            return self._cuda.win.cpu().numpy()
        return self.win.numpy().copy()

    def take_frame_counts(self):
        """Spike count per neuron since the previous call, as np.int32 (then reset)."""
        if self._cuda is not None:
            return self._cuda.take_frame()
        out = self.frame_cnt.numpy().copy()
        self.frame_cnt.zero_()
        return out
