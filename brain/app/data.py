"""Connectome loading: FlyWire (Shiu et al.) parquet -> CSC arrays + id maps + groups."""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

CONN = "2025_Connectivity_783.parquet"
COMP = "2025_Completeness_783.csv"


def load_ids(data_dir):
    """FlyWire root ids in tensor-index order (row order of the completeness csv, as in the reference)."""
    t = pacsv.read_csv(Path(data_dir) / COMP)
    return t.column(0).to_numpy().astype(np.int64)


def load_csc(data_dir, n, cache_dir=None):
    """Column-compressed (by presynaptic neuron) signed weights: W[post, pre] = 'Excitatory x Connectivity'.

    Returns indptr[int64, n+1], post[int32], w[float32]. Cached as .npy next to cache_dir if writable.
    """
    cache = Path(cache_dir) / "csc_783" if cache_dir else None
    if cache and (cache / "w.npy").exists():
        return (np.load(cache / "indptr.npy"), np.load(cache / "post.npy"), np.load(cache / "w.npy"))
    t = pq.read_table(
        Path(data_dir) / CONN,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = t.column(0).to_numpy().astype(np.int32)
    post = t.column(1).to_numpy().astype(np.int32)
    w = t.column(2).to_numpy().astype(np.float32)
    del t
    keep = w != 0
    pre, post, w = pre[keep], post[keep], w[keep]
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    if cache:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / "indptr.npy", indptr)
            np.save(cache / "post.npy", post)
            np.save(cache / "w.npy", w)
        except OSError:
            pass
    return indptr, post, w


def load_groups(ids, path=None):
    """Return {name: {role, description, idx(np.int64 sorted)}} mapped to tensor indices."""
    path = path or Path(__file__).with_name("groups.json")
    raw = json.load(open(path))
    order = np.argsort(ids)
    sorted_ids = ids[order]
    out = {}
    for name, g in raw.items():
        gid = np.asarray(g["ids"], dtype=np.int64)
        pos = np.searchsorted(sorted_ids, gid)
        pos[pos >= len(sorted_ids)] = 0
        ok = sorted_ids[pos] == gid
        idx = np.sort(order[pos[ok]])
        out[name] = {"role": g["role"], "default_rate_hz": g.get("default_rate_hz", 100.0), "description": g["description"], "idx": idx, "missing": int((~ok).sum())}
    return out


def default_dirs():
    return os.environ.get("DATA_DIR", "/data"), os.environ.get("CACHE_DIR", "/tmp/flybrain-cache")
