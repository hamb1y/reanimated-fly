#!/usr/bin/env python3
"""Per-neuron 3D positions for the FlyWire 783 connectome, in brain index order.

Brain index i = row i of data/2025_Completeness_783.csv (first column = FlyWire root id),
the same definition as brain/app/data.py load_ids().

Source: flyconnectome/flywire_annotations, supplemental_files/Supplemental_file1_neuron_annotations.tsv
(per root_id: soma_x/y/z and pos_x/y/z). Soma is used where present, else the root position (pos_*).
Table units are FAFB voxels; x and y are multiplied by 4 nm and z by 40 nm (VOXEL_NM, an assumption
about the table units, checked visually with tools/positions_check.png).

Outputs (web/static):
  neuron_positions.bin   int16 little-endian, shape [n, 3] (x, y, z interleaved), neuron i at bytes 6*i
                         normalised = value / 32767 in [-1, 1]; world = normalised * scale + offset
  neuron_positions.json  metadata: n, order, scale, offset, source, version, coverage, filled counts

Neurons with no table row: median of placed neighbours with the same super_class/cell_class/side
(then super_class/cell_class, then super_class), else the centroid of all placed neurons.

Needs: pandas, numpy; matplotlib for the check PNG (--no-plot skips it).
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
POS_DIR = DATA / "positions"
OUT_DIR = REPO / "web" / "static"
TOOLS = REPO / "tools"

COMMIT = "a83b2776d60d5764cef36b927f5f9679c16c47a2"  # flywire_annotations main (2026-09-29); latest release v3.2.0
SRC_NAME = "Supplemental_file1_neuron_annotations.tsv"
SRC_URL = f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{COMMIT}/supplemental_files/{SRC_NAME}"
COMP = "2025_Completeness_783.csv"
N_EXPECTED = 138639
VOXEL_NM = np.array([4.0, 4.0, 40.0])
Q = 32767
META = ["super_class", "cell_class", "side"]
FALLBACK = [("super_class+cell_class+side", META), ("super_class+cell_class", META[:2]), ("super_class", META[:1])]
XYZ = ["x", "y", "z"]


def fetch(path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(SRC_URL, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def brain_ids():
    ids = pd.read_csv(DATA / COMP, usecols=[0]).iloc[:, 0].to_numpy(np.int64)
    assert len(ids) == N_EXPECTED, len(ids)
    assert len(np.unique(ids)) == len(ids), "duplicate root ids in completeness csv"
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    src = POS_DIR / SRC_NAME
    fetch(src)
    ids = brain_ids()
    n = len(ids)

    cols = ["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", *META]
    ann = pd.read_csv(src, sep="\t", usecols=cols, dtype={"root_id": np.int64}, low_memory=False)
    assert not ann["root_id"].duplicated().any(), "duplicate root_id in annotation table"
    ann = ann.set_index("root_id")
    m = ann.reindex(ids)  # aligned to brain index order, NaN where absent

    soma = m[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * VOXEL_NM
    rpos = m[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
    has_soma = np.isfinite(soma).all(1)
    has_pos = np.isfinite(rpos).all(1)
    placed = has_soma | has_pos
    xyz = np.where(has_soma[:, None], soma, rpos)  # NaN rows fixed below
    in_table = np.isin(ids, ann.index.to_numpy())

    # fill unplaced neurons: median of placed neighbours, most specific group first
    df = pd.DataFrame(xyz, columns=XYZ)
    for k in META:
        df[k] = m[k].to_numpy()
    ok = df[placed]
    remaining = ~placed
    fill_by = {}
    for label, keys in FALLBACK:
        todo = np.flatnonzero(remaining)
        if todo.size == 0:
            break
        med = ok.groupby(keys, dropna=False)[XYZ].median().reset_index()
        q = df.iloc[todo][keys].reset_index(drop=True)
        q.insert(0, "_row", todo)
        j = q.merge(med, on=keys, how="left")
        got = j["x"].notna().to_numpy()
        rows = j["_row"].to_numpy()[got]
        xyz[rows] = j[XYZ].to_numpy()[got]
        remaining[rows] = False
        fill_by[label] = int(got.sum())
    centroid = xyz[placed].mean(0)
    n_centroid = int(remaining.sum())
    xyz[remaining] = centroid
    assert np.isfinite(xyz).all()

    lo, hi = xyz.min(0), xyz.max(0)
    offset = (lo + hi) / 2
    scale = float(((hi - lo) / 2).max())  # uniform: keeps aspect ratio
    norm = (xyz - offset) / scale
    q16 = np.clip(np.round(norm * Q), -Q, Q).astype("<i2")
    assert q16.shape == (n, 3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "neuron_positions.bin").write_bytes(q16.tobytes())

    n_soma = int(has_soma.sum())
    n_rpos = int((has_pos & ~has_soma).sum())
    meta = {
        "n": n,
        "order": "brain index",
        "index_def": "row order of data/2025_Completeness_783.csv (column 0 = FlyWire root id), as brain/app/data.py load_ids()",
        "file": "neuron_positions.bin",
        "dtype": "int16 little-endian, interleaved xyz, shape [n,3]",
        "decode": "normalised = value / 32767 in [-1,1]; world = normalised * scale + offset (world in voxel_nm-scaled units)",
        "scale": scale,
        "offset": [float(v) for v in offset],
        "voxel_nm": VOXEL_NM.tolist(),
        "voxel_nm_note": "assumed FAFB14 voxel size applied to table units (x,y 4 nm, z 40 nm)",
        "source": {
            "url": SRC_URL,
            "repo": "https://github.com/flyconnectome/flywire_annotations",
            "file": SRC_NAME,
            "commit": COMMIT,
            "release_latest": "v3.2.0",
            "sha256": sha256(src),
        },
        "version": "FlyWire 783 public release (FAFB14.1 space); annotations flywire_annotations v3.2.0 at the pinned commit",
        "coords": "soma_x/y/z where present, else pos_x/y/z (root position), same table",
        "coverage": {
            "total": n,
            "in_annotation_table": int(in_table.sum()),
            "soma": n_soma,
            "root_pos": n_rpos,
            "real_total": n_soma + n_rpos,
            "real_pct": round(100 * (n_soma + n_rpos) / n, 3),
            "filled": n - n_soma - n_rpos,
            "filled_pct": round(100 * (n - n_soma - n_rpos) / n, 3),
        },
        "filled_by": {**fill_by, "centroid": n_centroid},
        "filled_method": "median of placed neighbours by super_class/cell_class/side (then coarser), else centroid of placed neurons",
        "bbox_normalised": {"min": [float(v) for v in q16.min(0) / Q], "max": [float(v) for v in q16.max(0) / Q]},
        "bytes": n * 6,
    }
    (OUT_DIR / "neuron_positions.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("n", "scale", "offset", "coverage", "filled_by", "bbox_normalised")}, indent=2))

    if not args.no_plot:
        plot_check(OUT_DIR / "neuron_positions.bin", TOOLS / "positions_check.png")


def plot_check(bin_path, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.fromfile(bin_path, dtype="<i2").reshape(-1, 3) / Q  # read back what the web gets
    assert len(pts) == N_EXPECTED and np.isfinite(pts).all()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=110)
    views = [(0, 1, "x", "y", "top (x-y)"), (0, 2, "x", "z", "front (x-z)"), (1, 2, "y", "z", "side (y-z)")]
    for ax, (a, b, la, lb, title) in zip(axes, views):
        ax.scatter(pts[:, a], pts[:, b], s=0.3, c="black", alpha=0.25, linewidths=0)
        ax.set_aspect("equal")
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_title(f"FlyWire 783, {title}, {len(pts)} neurons")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print("wrote", out_png)


if __name__ == "__main__":
    main()
