"""Regenerate web/static/group_indices.json: member neuron indices for each named brain group.

Brain's GET /groups returns sizes and roles but no neuron ids, so the connectome view reads this file instead.
Source: brain/app/groups.json (FlyWire root ids) mapped to the brain tensor index, which is the row order of
data/2025_Completeness_783.csv (the same order as brain/app/data.py load_ids and web/static/neuron_positions.bin).

Usage (from the repo root): python3 web/tools/make_group_indices.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "static" / "group_indices.json"


def main():
    with open(ROOT / "data" / "2025_Completeness_783.csv") as f:
        rows = csv.reader(f)
        next(rows)
        ids = [int(r[0]) for r in rows]
    index = {v: i for i, v in enumerate(ids)}
    groups = json.loads((ROOT / "brain" / "app" / "groups.json").read_text())
    out, missing = {}, 0
    for name, g in groups.items():
        idx = []
        for root_id in g["ids"]:
            if root_id in index:
                idx.append(index[root_id])
            else:
                missing += 1
        out[name] = {"role": g["role"], "default_rate_hz": g["default_rate_hz"], "size": len(idx), "idx": sorted(set(idx))}
    doc = {
        "n": len(ids),
        "source": "brain/app/groups.json (FlyWire root ids) mapped to brain tensor index = row order of data/2025_Completeness_783.csv (web/tools/make_group_indices.py)",
        "groups": out,
    }
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"{len(out)} groups, {sum(len(g['idx']) for g in out.values())} members, {missing} ids not in table -> {OUT}")


if __name__ == "__main__":
    main()
