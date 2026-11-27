"""CUDA backend of app.model.Brain (semantics identical to the CPU path, see model.py).

One block = BLOCK (20) steps = 2 ms of sim time. Inside a block no step reads a spike of the same block (a spike at
step s is read at s+20), so a block is three fixed-shape stages, captured once as a CUDA graph:

  1. _lif_block (Triton, one thread per neuron): loops over the 20 steps in registers: refractory gate, input
     from I[j], Poisson kicks, conductance, membrane, threshold, reset. Writes spikes S[20, n] (int8) and the
     per-neuron spike count of the block.
  2. delivery: S is compacted into a list of spiking (step, neuron) pairs (torch cumsum + _compact_kernel, no
     host sync), then _scatter_kernel adds W[:, pre] for every pair into I[20, n] (the input of the next block)
     with atomics. Synapse work is proportional to the actual spikes, not to the 15M synapses.
  Stage 1 also clears I after reading it and updates the rolling 100 ms window and the frame counter.

The host syncs once per run() call (the runner calls run() once per ~32 ms frame), plus once per frame to read
the counters. Weights are stored as CSC by presynaptic neuron (indptr int64, post int32, w float32): the CSR of W^T.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from .model import BLOCK, NEVER, RING

LIF_THREADS = 256
SCATTER_GRID = 2048
SCATTER_CHUNK = 128


@triton.jit
def _lif_block_kernel(I, S, U, G, LAST, PS, STIM, SIL, RING, WIN, FRAME, T0p, P, n, base_seed,
                      B: tl.constexpr, NRING: tl.constexpr, BLOCK_N: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = offs < n
    k_m = tl.load(P + 0)
    decay = tl.load(P + 1)
    wsc = tl.load(P + 2)
    kick = tl.load(P + 3)
    thr = tl.load(P + 4)
    ureset = tl.load(P + 5)
    pb = tl.load(P + 6)
    rs = tl.load(P + 7).to(tl.int64)
    t0 = tl.load(T0p)  # steps completed before this block
    u = tl.load(U + offs, mask=m, other=0.0)
    g = tl.load(G + offs, mask=m, other=0.0)
    last = tl.load(LAST + offs, mask=m, other=0)
    ps = tl.load(PS + offs, mask=m, other=0.0)
    stim = tl.load(STIM + offs, mask=m, other=0) != 0
    sil = tl.load(SIL + offs, mask=m, other=0) != 0
    p = ps + pb * (1.0 - ps)  # per-step probability of a Poisson kick (stimulus or background)
    cnt = tl.zeros([BLOCK_N], dtype=tl.int32)
    for j in tl.static_range(B):
        t = t0 + (j + 1)  # 1-based step number, as in the CPU path
        d = t - last
        blocked = (d >= 1) & (d <= rs) & (~stim)  # refractory gate on the synaptic input
        inp = tl.load(I + j * n + offs, mask=m, other=0.0)
        inp = tl.where(blocked, 0.0, inp)
        gn = g * decay + wsc * inp
        r = tl.rand((base_seed + t).to(tl.int32), offs)
        u = u + tl.where(r < p, kick, 0.0)
        u = u + k_m * (g - u)  # membrane uses the conductance of the previous step
        u = tl.where(sil, 0.0, u)
        spk = (u > thr) & (~sil) & m
        gn = tl.where(spk | sil, 0.0, gn)
        u = tl.where(spk, ureset, u)
        last = tl.where(spk, t, last)
        tl.store(S + j * n + offs, spk.to(tl.int8), mask=m)
        cnt += spk.to(tl.int32)
        g = gn
    tl.store(U + offs, u, mask=m)
    tl.store(G + offs, g, mask=m)
    tl.store(LAST + offs, last, mask=m)
    for jz in tl.static_range(B):  # clear the input of this block (for the next delivery). Stores after the loop:
        tl.store(I + jz * n + offs, 0.0, mask=m)  # interleaving them with the loads gave wrong results (verified)
    # rolling 100 ms window: the block leaving the ring slot is subtracted, this block's counts are added
    slot = (t0 // B) % NRING
    old = tl.load(RING + slot * n + offs, mask=m, other=0)
    tl.store(RING + slot * n + offs, cnt, mask=m)
    win = tl.load(WIN + offs, mask=m, other=0) - old + cnt
    tl.store(WIN + offs, win, mask=m)
    fr = tl.load(FRAME + offs, mask=m, other=0) + cnt
    tl.store(FRAME + offs, fr, mask=m)


@triton.jit
def _compact_kernel(FLAGS, POS, LIST, NTOT, BLOCK_N: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = offs < NTOT
    f = tl.load(FLAGS + offs, mask=m, other=0)
    pos = tl.load(POS + offs, mask=m, other=0)  # inclusive scan of the flags
    tl.store(LIST + pos - 1, offs, mask=m & (f != 0))


@triton.jit
def _scatter_kernel(LIST, NS, INDPTR, POST, WV, I, n, G: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    ns = tl.load(NS)
    for k in range(pid, ns, G):
        f = tl.load(LIST + k).to(tl.int64)  # flat (step j, presynaptic p) = j * n + p
        j = f // n
        p = f - j * n
        s0 = tl.load(INDPTR + p)
        s1 = tl.load(INDPTR + p + 1)
        for c in range(s0, s1, C):
            o = c + tl.arange(0, C)
            mm = o < s1
            q = tl.load(POST + o, mask=mm, other=0).to(tl.int64)
            v = tl.load(WV + o, mask=mm, other=0.0)
            tl.atomic_add(I + j * n + q, v, mask=mm)


class CudaLif:
    def __init__(self, indptr, post, w, n, mem_cap_mib=1024):
        self.dev = torch.device("cuda", 0)
        total = torch.cuda.get_device_properties(self.dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.95, mem_cap_mib * 2**20 / total), self.dev)
        self.n = n
        dev, f32, i32, i64 = self.dev, torch.float32, torch.int32, torch.int64
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=dev)  # noqa: E731
        self.indptr = torch.as_tensor(np.asarray(indptr, dtype=np.int64), device=dev)
        self.post = torch.as_tensor(np.asarray(post, dtype=np.int32), device=dev)
        self.wv = torch.as_tensor(np.asarray(w, dtype=np.float32), device=dev)
        self.u, self.g = z(n, f32), z(n, f32)
        self.last = torch.full((n,), NEVER, dtype=i64, device=dev)
        self.I = z((BLOCK, n), f32)  # synaptic input per step of the current block
        self.S = z((BLOCK, n), torch.int8)  # spikes of the current block
        self.ring = z((RING, n), i32)  # spike counts of the last RING blocks
        self.win = z(n, i32)  # rolling 100 ms window
        self.frame = z(n, i32)  # since the last frame
        self.ps = z(n, f32)  # stimulation kick probability per step (0 where silenced)
        self.stim = z(n, i32)  # refractory exemption
        self.sil = z(n, i32)
        self.P = z(8, f32)  # k_m, decay, w_scale, kick, thr, u_reset, p_bg, rs
        self.T0 = z(1, i64)  # steps completed
        self.pos = z(BLOCK * n, i32)  # inclusive scan of the spike flags of the block
        self.LIST = z(BLOCK * n, i32)  # compacted spiking (j * n + p)
        self.seed = 0x2F1B
        self._capture()

    def _block(self):
        _lif_block_kernel[(triton.cdiv(self.n, LIF_THREADS),)](
            self.I, self.S, self.u, self.g, self.last, self.ps, self.stim, self.sil,
            self.ring, self.win, self.frame, self.T0, self.P, self.n, self.seed,
            B=BLOCK, NRING=RING, BLOCK_N=LIF_THREADS)
        flags = self.S.view(-1)
        torch.cumsum(flags, 0, dtype=torch.int32, out=self.pos)
        _compact_kernel[(triton.cdiv(BLOCK * self.n, LIF_THREADS),)](
            flags, self.pos, self.LIST, BLOCK * self.n, BLOCK_N=LIF_THREADS)
        _scatter_kernel[(SCATTER_GRID,)](self.LIST, self.pos[BLOCK * self.n - 1:], self.indptr, self.post, self.wv,
                                         self.I, self.n, G=SCATTER_GRID, C=SCATTER_CHUNK)
        self.T0.add_(BLOCK)

    def _capture(self):
        side = torch.cuda.Stream(self.dev)
        side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(side):
            self._block()  # compiles the Triton kernels before capture
        torch.cuda.current_stream(self.dev).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._block()
        self.reset()

    # ---- host-side control (not inside the graph) ----
    def replay(self):
        self.graph.replay()

    def sync(self):
        torch.cuda.current_stream(self.dev).synchronize()

    def spikes_host(self):
        return self.S.cpu().numpy()

    def reset(self):
        for t in (self.u, self.g, self.I, self.S, self.pos, self.LIST, self.ring, self.win, self.frame, self.T0):
            t.zero_()
        self.last.fill_(NEVER)

    def set_params(self, vals):
        self.P.copy_(torch.as_tensor(np.asarray(vals, dtype=np.float32), device=self.dev))

    def set_masks(self, ps, stim_i32, sil_i32):
        self.ps.copy_(torch.as_tensor(ps, device=self.dev))
        self.stim.copy_(torch.as_tensor(stim_i32, device=self.dev))
        self.sil.copy_(torch.as_tensor(sil_i32, device=self.dev))

    def zero_neurons(self, idx):
        t = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.dev)
        self.u[t] = 0.0
        self.g[t] = 0.0

    def take_frame(self):
        out = self.frame.cpu().numpy()
        self.frame.zero_()
        return out

    def memory_mib(self):
        return torch.cuda.max_memory_reserved(self.dev) / 2**20
