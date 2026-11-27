"""Step-by-step comparison of app.model.Brain against the reference TorchModel (reference/run_pytorch.py).

Poisson kicks are made deterministic by using a rate of 1/dt (probability 1 per step) on a stimulated set,
so the spike rasters of both implementations must be identical. Run:
  python tools/verify_reference.py [steps]
Needs pandas (reference imports it); not part of the container.
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "brain"))
fake = types.ModuleType("benchmark")
for k in ("T_RUN_VALUES_SEC", "N_RUN_VALUES"):
    setattr(fake, k, [])
for k in ("path_comp", "path_con", "path_wt"):
    setattr(fake, k, None)
for k in ("get_experiment", "get_spike_output_path", "print_summary_table", "save_result_csv", "spike_io_enabled"):
    setattr(fake, k, lambda *a, **kw: None)
sys.modules["benchmark"] = fake
sys.path.insert(0, str(ROOT / "reference"))
import run_pytorch as ref  # noqa: E402
from app import data, model  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
D = ROOT / "data"
ids = data.load_ids(D)
n = len(ids)
indptr, post, w = data.load_csc(D, n, None)
groups = data.load_groups(ids)
stim = np.unique(np.concatenate([groups["sugar_grn"]["idx"], groups["looming"]["idx"][:150]]))

# reference weights: W[post, pre] as sparse CSR, shape (n, n)
pre = np.repeat(np.arange(n), np.diff(indptr))
W = torch.sparse_coo_tensor(torch.from_numpy(np.stack([post.astype(np.int64), pre])), torch.from_numpy(w), (n, n)).coalesce().to_sparse_csr()
m = ref.TorchModel(1, n, ref.DT, ref.MODEL_PARAMS, W, exc_indices=stim.tolist())
c, db, s, v, rf = m.state_init()
rates = torch.zeros(1, n)
rates[:, stim] = 1000.0 / ref.DT  # probability 1 per step
steps = steps - steps % model.BLOCK  # ours advances in blocks of BLOCK steps
ref_sets = []
with torch.no_grad():
    for t in range(steps):
        c, db, s, v, rf = m(rates, c, db, s, v, rf)
        ref_sets.append(set(torch.nonzero(s[0] > 0).squeeze(1).tolist()))
b = model.Brain(indptr, post, w, record=True)
b.stimulate(stim, 1000.0 / ref.DT)
b.run(steps)
ours = {}
for t, idx in b.raster:
    ours[t] = set(idx.tolist())
bad = 0
tot = 0
for t in range(steps):
    a, bb = ref_sets[t], ours.get(t + 1, set())  # our step numbers are 1-based
    tot += len(a)
    if a != bb:
        bad += 1
        if bad <= 5:
            print("step", t + 1, "ref", len(a), "ours", len(bb), "sym diff", len(a ^ bb))
print(f"steps={steps} total ref spikes={tot} mismatching steps={bad}")
