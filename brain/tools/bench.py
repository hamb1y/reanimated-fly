"""Speed and memory of the brain on the real connectome (run in the image with the data mounted).

    python tools/bench.py                      # BRAIN_DEVICE from env (auto by default)
    python tools/bench.py --verify             # GPU vs CPU raster agreement under deterministic drive

Per scenario: flat-out capacity (sim ms per wall ms, the number /health reports as sim_capacity) and the wall time
of each 32 ms run() call, the unit the runner paces with (a call slower than 32 ms cannot keep real time).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app import data  # noqa: E402
from app.model import BLOCK, DT, make_brain  # noqa: E402
from app.runner import CHUNK  # noqa: E402

INPUT_GROUPS = ["sugar_grn", "bitter_grn", "looming", "lplc2", "jo_mechano", "or56a_olfactory", "p9"]
SCEN = {
    "idle": [],
    "sugar_100hz": [("sugar_grn", 100.0)],
    "heavy_338_neurons": [("sugar_grn", 100.0), ("forward_walk", 150.0), ("looming", 200.0), ("jo_mechano", 300.0)],
    "stress_all_inputs_default_rate": [(g, None) for g in INPUT_GROUPS],
}


def load(device, record=False):
    d, c = data.default_dirs()
    ids = data.load_ids(d)
    indptr, post, w = data.load_csc(d, len(ids), c)
    groups = data.load_groups(ids)
    gpu_mem = int(os.environ.get("BRAIN_GPU_MEM_MIB", "1024"))
    return make_brain(indptr, post, w, device, gpu_mem_mib=gpu_mem, record=record), groups


def apply(b, groups, spec):
    b.clear()
    b.reset()
    tot = 0
    for name, rate in spec:
        idx = groups[name]["idx"]
        b.stimulate(idx, groups[name]["default_rate_hz"] if rate is None else rate)
        tot += len(idx)
    return tot


def bench(device, sim_steps=20000, warm_steps=2000):
    b, groups = load(device)
    out = {"device": device, "threads": torch.get_num_threads(), "neurons": b.n, "synapses": b.n_syn, "scenarios": {}}
    for name, spec in SCEN.items():
        nstim = apply(b, groups, spec)
        b.run(warm_steps)
        b.take_frame_counts()
        chunk_t = []
        n_chunks = sim_steps // CHUNK
        steps = n_chunks * CHUNK
        t0 = time.perf_counter()
        for _ in range(n_chunks):
            c0 = time.perf_counter()
            b.run(CHUNK)
            chunk_t.append(time.perf_counter() - c0)
        wall = time.perf_counter() - t0
        spikes = int(b.take_frame_counts().sum())  # spikes since the warm-up was read (all of the measured run)
        ct = np.array(chunk_t) * 1000.0
        out["scenarios"][name] = {
            "stimulated_neurons": nstim,
            "sim_capacity_x": round(steps * DT / 1000.0 / wall, 2),
            "chunk32ms_mean_ms": round(float(ct.mean()), 2),
            "chunk32ms_p95_ms": round(float(np.percentile(ct, 95)), 2),
            "chunk32ms_max_ms": round(float(ct.max()), 2),
            "spikes_per_sim_s": round(spikes / (steps * DT / 1000.0)),
        }
    if device == "cuda":
        out["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        out["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out


def verify():
    """Deterministic drive (probability 1, no RNG): CUDA raster must equal the CPU raster for a while."""
    res = {}
    for dev in ("cpu", "cuda"):
        b, groups = load(dev, record=True)
        apply(b, groups, [("sugar_grn", 10000.0), ("looming", 10000.0)])
        b.run(600)
        res[dev] = b.raster
    cpu, cuda = res["cpu"], res["cuda"]
    first_diff = None
    for k in range(max(len(cpu), len(cuda))):
        a = cpu[k] if k < len(cpu) else (None, np.zeros(0, np.int64))
        c = cuda[k] if k < len(cuda) else (None, np.zeros(0, np.int64))
        if a[0] != c[0] or not np.array_equal(a[1], c[1]):
            first_diff = {"entry": k, "cpu_step": a[0], "cuda_step": c[0]}
            break
    total = sum(len(i) for _, i in cpu)
    return {"cpu_spike_events": total, "cuda_spike_events": sum(len(i) for _, i in cuda),
            "identical_raster": first_diff is None, "first_difference": first_diff}


if __name__ == "__main__":
    if "--verify" in sys.argv:
        print(json.dumps(verify()))
    else:
        dev = os.environ.get("BRAIN_DEVICE", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(json.dumps(bench(dev)))
