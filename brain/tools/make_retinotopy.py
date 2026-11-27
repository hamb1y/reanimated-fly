"""Build app/retina.json: which ommatidium (visual direction) each photoreceptor looks through.

Run in the brain image with the data mounted:
    docker run --rm --network host -v $PWD/data:/data:ro -v $PWD/brain:/srv -w /srv flybrain-brain:cpu \
        python tools/make_retinotopy.py

Method (README "Vision" has the numbers):
1. Lamina cartridges from the CONNECTOME: each R1-6 is put in the cartridge of the L2 it synapses on most
   (neural superposition: the ~6 R1-6 of one cartridge look in the same direction). R1-6 with no L2 synapse use
   their L1/L3 targets (whose cartridge = the cartridge giving them most R1-6 input).
2. Cartridge sheet -> 2D: positions of the cartridge L2 cells (FlyWire voxels 4x4x40 nm -> um) are flattened with
   Isomap (kNN graph geodesics + classical MDS), because the lamina is a curved shell.
3. Orientation from anatomy: up = dorsal = small y (checked: dorsal-rim R7, the ones synapsing on DmDRA cells,
   are the most dorsal); front = anterior = small z. The lamina is not inverted relative to the eye.
   Cross-check: lamina vs medulla positions of the same column (cartridge L1 -> its Mi1) must show the outer
   chiasm: dorso-ventral preserved (positive correlation), antero-posterior inverted (negative). Printed below.
4. Scale: per axis (the sheet is taller than wide), then an equal-area radial remap (monotone in the distance
   from the sheet centre) so the cartridges fill the rendered 58 deg disc evenly. (Absolute angles are NOT
   measured; retinotopic order is.) Each cartridge -> nearest lattice ommatidium (within one spacing, else unused).
   The file stores, per FlyWire annotation side, root id, type and (h, v) = (front, up) in degrees; which body eye
   uses which annotation side is decided at load time (app/retina.py, EYE_CONVENTION).
5. R8: cartridge of the column cells it synapses on (L1, L3, Mi1 with known cartridge). R7: cartridge of its R8
   partner (R7<->R8 synapse in the same column) or its L3 targets. Photoreceptors without a connectivity cartridge
   (incomplete left lamina) take the mean direction of their 3 nearest same-type neighbours (own FlyWire position).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import data  # noqa: E402
from app.retina import RADIUS_DEG, SPACING_DEG, lattice  # noqa: E402

ANN = "positions/Supplemental_file1_neuron_annotations.tsv"
VOX_UM = np.array([0.004, 0.004, 0.040])


def load_ann(data_dir, ids):
    t = pacsv.read_csv(Path(data_dir) / ANN, parse_options=pacsv.ParseOptions(delimiter="\t"),
                       convert_options=pacsv.ConvertOptions(column_types={"root_id": pa.int64()},
                                                            include_columns=["root_id", "pos_x", "pos_y", "pos_z", "cell_type", "side"]))
    rid = t.column("root_id").to_numpy()
    order = np.argsort(ids)
    s = ids[order]
    p = np.searchsorted(s, rid)
    p[p >= len(s)] = 0
    ok = s[p] == rid
    idx = order[p[ok]]
    n = len(ids)
    ct = np.full(n, "", dtype=object)
    side = np.full(n, "", dtype=object)
    xyz = np.full((n, 3), np.nan)
    ct[idx] = np.asarray(t.column("cell_type").to_pylist(), dtype=object)[ok]
    side[idx] = np.asarray(t.column("side").to_pylist(), dtype=object)[ok]
    for k, c in enumerate(["pos_x", "pos_y", "pos_z"]):
        xyz[idx, k] = np.asarray([np.nan if v is None else float(v) for v in t.column(c).to_pylist()])[ok]
    ct = np.array([x or "" for x in ct], dtype=object)
    return ct, side, xyz * VOX_UM


def isomap2(P, k=10):
    n = len(P)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    G = np.full((n, n), np.inf)
    nn = np.argsort(D, 1)[:, 1:k + 1]
    for i in range(n):
        G[i, nn[i]] = D[i, nn[i]]
        G[nn[i], i] = D[i, nn[i]]
    np.fill_diagonal(G, 0.0)
    for m in range(n):
        np.minimum(G, G[:, m:m + 1] + G[m:m + 1, :], out=G)
    comp = np.isfinite(G[np.argmax(np.isfinite(G).sum(1))])
    Gc = G[np.ix_(comp, comp)] ** 2
    J = np.eye(len(Gc)) - 1.0 / len(Gc)
    B = -0.5 * J @ Gc @ J
    val, vec = np.linalg.eigh(B)
    E = vec[:, -2:] * np.sqrt(np.maximum(val[-2:], 0))
    out = np.full((n, 2), np.nan)
    out[comp] = E
    return out


def knn_fill(known_xyz, known_hv, query_xyz, k=3):
    d = np.linalg.norm(query_xyz[:, None] - known_xyz[None], axis=2)
    nn = np.argsort(d, 1)[:, :k]
    return known_hv[nn].mean(1)


def build(data_dir="/data", cache_dir=None):
    ids = data.load_ids(data_dir)
    indptr, post, w = data.load_csc(data_dir, len(ids), cache_dir)
    ct, side, xyz = load_ann(data_dir, ids)

    def out(i):
        a, b = indptr[i], indptr[i + 1]
        return post[a:b], np.abs(w[a:b])

    def vote(cells, cart_of):
        res = {}
        for c in cells:
            v = Counter()
            p, ww = out(c)
            for q, x in zip(p.tolist(), ww.tolist()):
                if q in cart_of:
                    v[cart_of[q]] += x
            if v:
                res[int(c)] = v.most_common(1)[0][0]
        return res

    def by_input(cells, src_cart):
        cs = set(int(c) for c in cells)
        acc = defaultdict(Counter)
        for r, c in src_cart.items():
            p, ww = out(r)
            for q, x in zip(p.tolist(), ww.tolist()):
                if q in cs:
                    acc[q][c] += x
        return {l: cn.most_common(1)[0][0] for l, cn in acc.items()}

    lat = lattice()
    result = {"meta": {}, "eyes": {}}
    val = {}
    for eye in ("left", "right"):
        sel = lambda t: np.flatnonzero((ct == t) & (side == eye))  # noqa: E731
        R16, R7, R8 = sel("R1-6"), sel("R7"), sel("R8")
        L1, L2, L3, Mi1 = sel("L1"), sel("L2"), sel("L3"), sel("Mi1")
        l2pos = {int(c): k for k, c in enumerate(L2)}
        rc = vote(R16, l2pos)
        l1c, l3c = by_input(L1, rc), by_input(L3, rc)
        via_l = vote([r for r in R16 if int(r) not in rc], {**l1c, **l3c})
        rc.update(via_l)
        mic = by_input(Mi1, l1c)
        r8c = vote(R8, {**l1c, **l3c, **mic})
        r7c = vote(R7, {**r8c, **l3c})
        carts = np.array(sorted(set(rc.values())))
        P = xyz[L2[carts]]
        E = isomap2(P)
        good = ~np.isnan(E).any(1)
        # orientation: v (up) along -y, h (front) along -z, fitted on the embedding
        X = np.c_[E[good], np.ones(good.sum())]
        cy = np.linalg.lstsq(X, -P[good, 1], rcond=None)[0][:2]
        uv = cy / np.linalg.norm(cy)
        uh = np.array([uv[1], -uv[0]])
        if np.corrcoef(E[good] @ uh, -P[good, 2])[0, 1] < 0:
            uh = -uh
        hv = np.c_[E @ uh, E @ uv]
        hv -= np.nanmedian(hv, 0)
        sd = np.nanstd(hv, 0)  # per-axis: the cartridge sheet is stretched to fill the round rendered field
        r95 = np.nanpercentile(np.linalg.norm(hv / sd, axis=1), 95)
        scale = RADIUS_DEG / (r95 * sd)  # deg per um, [horizontal, vertical]
        hv *= scale
        # equal-area radius (monotone, keeps the retinotopic order): every cartridge inside the field, even density
        rho = np.linalg.norm(hv, axis=1)
        okr = ~np.isnan(rho)
        rank = np.empty(okr.sum())
        rank[np.argsort(rho[okr])] = (np.arange(okr.sum()) + 0.5) / okr.sum()
        hv[okr] *= (RADIUS_DEG * np.sqrt(rank) / np.maximum(rho[okr], 1e-9))[:, None]
        cart_hv = {int(c): hv[k] for k, c in enumerate(carts) if good[k]}
        # chiasm check (lamina L2 of a cartridge vs that column's Mi1)
        pairs = [(c, m) for m, c in mic.items() if c in cart_hv]
        la = np.array([xyz[L2[c]] for c, _ in pairs])
        me = np.array([xyz[m] for _, m in pairs])
        ch_y = float(np.corrcoef(la[:, 1], me[:, 1])[0, 1])
        ch_z = float(np.corrcoef(la[:, 2], me[:, 2])[0, 1])
        # per photoreceptor direction
        rows = []
        for cells, cmap, typ in ((R16, rc, "R1-6"), (R8, r8c, "R8"), (R7, r7c, "R7")):
            known = [c for c in cells if int(c) in cmap and cmap[int(c)] in cart_hv and not np.isnan(xyz[c]).any()]
            unknown = [c for c in cells if c not in set(known)]
            khv = np.array([cart_hv[cmap[int(c)]] for c in known])
            fill = knn_fill(xyz[known], khv, xyz[unknown]) if unknown and known else np.zeros((0, 2))
            for c, h in zip(known, khv):
                rows.append((int(c), typ, h, "connectivity"))
            for c, h in zip(unknown, fill):
                if not np.isnan(h).any():
                    rows.append((int(c), typ, h, "position"))
        # dorsal-rim check: R7 that synapse on DmDRA cells should be dorsal (high v)
        dra = set(int(r) for r in R7 if any(str(ct[q]).startswith("DmDRA") for q in out(r)[0].tolist()))
        v_all = np.array([h[1] for c, t, h, _ in rows if t == "R7"])
        v_dra = np.array([h[1] for c, t, h, _ in rows if t == "R7" and c in dra])
        # lattice coverage (a = +-h, b = v; the same for either eye convention)
        ab = np.array([[h[0], h[1]] for _, _, h, _ in rows])
        d = np.linalg.norm(ab[:, None] - lat[None], axis=2)
        om = d.argmin(1)
        om[d.min(1) > SPACING_DEG] = -1
        covered = int((d.min(0) <= SPACING_DEG).sum())  # lattice points with a photoreceptor direction within one spacing
        result["eyes"][eye] = {  # keyed by the FlyWire annotation side
            "root_id": [int(ids[c]) for c, _, _, _ in rows],
            "type": [t for _, t, _, _ in rows],
            "src": [s for _, _, _, s in rows],
            "hv_deg": [[round(float(h[0]), 2), round(float(h[1]), 2)] for _, _, h, _ in rows],
        }
        nn = np.sort(np.linalg.norm(P[:, None] - P[None], axis=2), 1)[:, 1]
        val[eye] = {
            "R1-6": [len(R16), sum(1 for r in rows if r[1] == "R1-6")], "R7": [len(R7), sum(1 for r in rows if r[1] == "R7")],
            "R8": [len(R8), sum(1 for r in rows if r[1] == "R8")],
            "by_connectivity": sum(1 for r in rows if r[3] == "connectivity"), "by_position": sum(1 for r in rows if r[3] == "position"),
            "cartridges": int(len(carts)), "L2_total": int(len(L2)), "R16_per_cartridge_median": float(np.median(list(Counter(rc.values()).values()))),
            "cartridge_nn_um_median": round(float(np.median(nn)), 2), "deg_per_um_h_v": [round(float(x), 3) for x in scale],
            "implied_spacing_deg_h_v": [round(float(np.median(nn) * x), 2) for x in scale],
            "extent_um_h_v": [round(float(x), 1) for x in 2 * r95 * sd],
            "chiasm_corr_dorsoventral": round(ch_y, 3), "chiasm_corr_anteroposterior": round(ch_z, 3),
            "dra_R7": len(v_dra), "dra_R7_elev_median_deg": round(float(np.median(v_dra)), 1) if len(v_dra) else None,
            "all_R7_elev_median_deg": round(float(np.median(v_all)), 1),
            "outside_field": int((om < 0).sum()), "ommatidia_nearest_to_a_receptor": int(len(set(om[om >= 0].tolist()))),
            "ommatidia_within_one_spacing_of_a_receptor": covered,
        }
    result["meta"] = {"lattice": {"spacing_deg": SPACING_DEG, "radius_deg": RADIUS_DEG, "n": len(lat)},
                      "validation": val, "built_by": "brain/tools/make_retinotopy.py"}
    return result


if __name__ == "__main__":
    d = data.default_dirs()
    res = build(d[0], d[1])
    out = Path(__file__).resolve().parents[1] / "app" / "retina.json"
    json.dump(res, open(out, "w"), separators=(",", ":"))
    print(json.dumps(res["meta"], indent=1))
    print("wrote", out, out.stat().st_size, "bytes")
