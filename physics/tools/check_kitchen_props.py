#!/usr/bin/env python3
"""Check that app/data/kitchen_props.json still matches the props drawn in web/src/lib/kitchen.ts (stdlib only).

Each prop's `kitchen_ts` is the line of kitchen.ts that places it. The check fails if that line is gone (the prop was moved
or resized in the art) or if the prop's x, y are not among the numbers on that line (named anchors such as BX are
resolved from their `const` line). Run from anywhere:  python3 physics/tools/check_kitchen_props.py [path/to/kitchen.ts]
"""
import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
props = json.loads((here.parent / "app" / "data" / "kitchen_props.json").read_text())
kts = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parents[1] / "web" / "src" / "lib" / "kitchen.ts"
src = kts.read_text()
bad = []
for p in props["props"]:
    line = p.get("kitchen_ts")
    if not line:
        bad.append(f"{p['id']}: no kitchen_ts line")
        continue
    if line not in src:
        bad.append(f"{p['id']}: line not found in kitchen.ts: {line!r}")
        continue
    nums = {float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", line)}
    x, y = (float(v) for v in p["pos"])
    if p["id"] == "sink":  # the sink is placed by its hole corners; the footprint is the centre
        ok = {-275.0, -165.0, -25.0, 45.0} <= nums and (x, y) == (-220.0, 10.0)
    else:
        ok = x in nums and y in nums
    if not ok:
        bad.append(f"{p['id']}: pos {p['pos']} not on its kitchen.ts line {line!r}")
print(f"{len(props['props'])} props checked against {kts}")
if bad:
    print("MISMATCH:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("ok")
