"""Regenerate brain/app/groups.json.

Primary source: reference/rojas/brain_body_bridge.py (MIT, rjo6615/fly-brain-interactive): FlyWire root ids of the
descending neurons (DN_NEURONS) and the stimulus sets (STIMULI) with their default rates.
Extras (LPLC2) come from the FlyWire annotation supertable (flyconnectome/flywire_annotations,
Supplemental_file1_neuron_annotations.tsv).

Usage: python tools/make_groups.py reference/rojas/brain_body_bridge.py annotations.tsv app/groups.json
"""
import ast
import json
import sys

import pandas as pd

src = ast.parse(open(sys.argv[1]).read())
ref = {}
for n in src.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("DN_NEURONS", "STIMULI"):
        ref[n.targets[0].id] = ast.literal_eval(n.value)
DN, ST = ref["DN_NEURONS"], ref["STIMULI"]
a = pd.read_csv(sys.argv[2], sep="\t", low_memory=False)


def dn(*names):
    return [DN[x] for x in names]


groups = {
    "sugar_grn": ("input", ST["sugar"]["rate"], "sugar-sensing GRNs (21 neurons, example.ipynb set from rojas reference)", ST["sugar"]["neurons"]),
    "bitter_grn": ("input", ST["bitter"]["rate"], "bitter-sensing GRNs (rojas reference)", ST["bitter"]["neurons"]),
    "looming": ("input", ST["lc4"]["rate"], "LC4 looming/escape visual projection neurons (rojas reference 'lc4')", ST["lc4"]["neurons"]),
    "lplc2": ("input", 100.0, "LPLC2 collision-detecting visual projection neurons (FlyWire annotations cell_type == LPLC2)", sorted(int(x) for x in a.loc[a.cell_type == "LPLC2", "root_id"])),
    "jo_mechano": ("input", ST["jo"]["rate"], "Johnston's organ touch/vibration afferents (rojas reference 'jo')", ST["jo"]["neurons"]),
    "or56a_olfactory": ("input", ST["or56a"]["rate"], "Or56a olfactory receptor neurons (rojas reference)", ST["or56a"]["neurons"]),
    "p9": ("input", ST["p9"]["rate"], "P9 forward-walking DNs only (2 neurons)", ST["p9"]["neurons"]),
    "forward_walk": ("readout", 100.0, "forward walking DNs: P9 left/right + P9/oDN1 left/right (rojas DN_GROUPS['forward'])", dn("P9_left", "P9_right", "P9_oDN1_left", "P9_oDN1_right")),
    "turn_left": ("readout", 100.0, "steering DNa01 + DNa02, left (rojas DN_GROUPS['turn_L'])", dn("DNa01_left", "DNa02_left")),
    "turn_right": ("readout", 100.0, "steering DNa01 + DNa02, right (rojas DN_GROUPS['turn_R'])", dn("DNa01_right", "DNa02_right")),
    "backward_walk": ("readout", 100.0, "moonwalker descending neurons MDN x4 (backward walking)", dn("MDN_1", "MDN_2", "MDN_3", "MDN_4")),
    "escape_gf": ("readout", 100.0, "giant fibres GF_1/GF_2 (escape)", dn("GF_1", "GF_2")),
    "groom_adn1": ("readout", 100.0, "antennal grooming aDN1 left/right", dn("aDN1_left", "aDN1_right")),
    "feed_mn9": ("readout", 100.0, "proboscis motor neuron MN9 left/right (feeding)", dn("MN9_left", "MN9_right")),
}
out = {k: {"role": r, "default_rate_hz": hz, "description": d, "ids": [int(x) for x in i]} for k, (r, hz, d, i) in groups.items()}
json.dump(out, open(sys.argv[3], "w"), indent=0)
print({k: len(v["ids"]) for k, v in out.items()})
