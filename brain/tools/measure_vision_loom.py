"""brain/tools/measure_vision_loom.py -- offline measurement: do LC4 / LPLC2 respond to a looming scary object seen through the real eye pipeline
(physics vision.py renderer -> brain retina.Transducer -> photoreceptor Poisson drive) vs the scripted looming drive?
Full FlyWire connectome, CPU backend, flat out (sim time). Vision updates every 66 ms of sim time (~15 Hz).

Run in the brain image with the repo mounted (physics/app is loaded by path, its package name clashes with brain/app):
    docker run --rm --network none -v $PWD/data:/data:ro -v $PWD:/repo -w /repo/brain -e PLAN='[["vision","loom",0]]' \
      --entrypoint python flybrain-brain:latest tools/measure_vision_loom.py
PLAN: list of [cond, kind, seed, (adapt), (w_syn)], cond vision|scripted, kind loom|recede|translate|none|dark.
Prints one JSON summary per run: [mean before, mean during, peak during] per group (Hz per neuron).
"""
import importlib.util, json, math, os, sys, time, types
from collections import namedtuple
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from app import data, retina
from app.model import Brain, DT

def _load_physics():
    pdir = os.environ.get("PHYSICS_APP", os.path.join(os.path.dirname(os.path.dirname(HERE)), "physics", "app"))
    pkg = types.ModuleType("pvis"); pkg.__path__ = [pdir]; sys.modules["pvis"] = pkg
    for name in ("world", "vision"):
        spec = importlib.util.spec_from_file_location("pvis." + name, os.path.join(pdir, name + ".py"))
        m = importlib.util.module_from_spec(spec); sys.modules["pvis." + name] = m; spec.loader.exec_module(m)
    return sys.modules["pvis.vision"]
_v = _load_physics()
EyeRenderer, quantise = _v.EyeRenderer, _v.quantise

torch.set_num_threads(int(os.environ.get("T", "10")))
Obj = namedtuple("Obj", "id kind x y amount")
UPD = 660  # steps per vision update (66 ms)
DTU = UPD * DT / 1000.0
FLY = (0.0, 0.0, 0.98)
QUAT = (1.0, 0.0, 0.0, 0.0)  # heading +x
BEAR = math.radians(float(os.environ.get("BEAR", "40")))  # object bearing, + = left
DATA, CACHE = data.default_dirs()
ids = data.load_ids(DATA)
indptr, post, w = data.load_csc(DATA, len(ids), CACHE)
groups = data.load_groups(ids)
rmap = retina.load_map(ids)
eye = EyeRenderer(None)
G = ["looming_left", "looming_right", "lplc2_left", "lplc2_right", "escape_gf", "turn_left", "turn_right", "forward_walk", "backward_walk"]

def xy(d, b=BEAR):
    return (FLY[0] + d * math.cos(b), FLY[1] + d * math.sin(b))

# trajectories: list of per-update object positions (None = no object). 1 s pre (static far), 1 s event, 0.5 s hold.
N_PRE, N_EV, N_POST = 15, 15, 8
def traj(kind):
    if kind == "loom":
        ds = [60.0] * N_PRE + list(np.linspace(60, 6, N_EV)) + [6.0] * N_POST
        return [xy(d) for d in ds]
    if kind == "recede":
        ds = [6.0] * N_PRE + list(np.linspace(6, 60, N_EV)) + [60.0] * N_POST
        return [xy(d) for d in ds]
    if kind == "translate":  # constant 20 mm, bearing sweeps 80 -> 10 deg (left eye)
        bs = [math.radians(80)] * N_PRE + list(np.radians(np.linspace(80, 10, N_EV))) + [math.radians(10)] * N_POST
        return [xy(20.0, b) for b in bs]
    if kind == "dark":  # diagnostic: the whole visual field goes black (upper bound of the photoreceptor drive)
        return ["DARK0"] * N_PRE + ["DARK1"] * (N_EV + N_POST)
    if kind == "none":
        return [None] * (N_PRE + N_EV + N_POST)
    raise ValueError(kind)

def scripted_profile(tr):
    """Rates the physics scripted path (behavior.py + brainloop.py) would send for this trajectory, fear 0.9."""
    out, th_prev, g = [], None, 0.0
    for p in tr:
        d = math.hypot(p[0] - FLY[0], p[1] - FLY[1])
        th = 2 * math.atan(5.0 / d)
        if th_prev is not None:
            inst = (th - th_prev) / DTU
            g += (1 - math.exp(-DTU / 0.3)) * (inst - g)
        th_prev = th
        grow = min(max(g / 0.5, 0.0), 1.0)
        near = min(max(1 - d / 60.0, 0.0), 1.0)
        thr = 0.9 * max(grow, near) if d <= 60.0 + 1e-6 else 0.0
        out.append({"looming_left": 200 * thr, "lplc2_left": 100 * thr} if thr * 200 >= 0.5 else {})
    return out

def run(cond, kind, seed, adapt="local", wsyn=None):
    b = Brain(indptr, post, w, seed=seed, device="cpu")
    if wsyn is not None:
        b.set_params({"w_syn": wsyn})
    tr = traj(kind)
    T = retina.Transducer(rmap, mode=adapt)
    prof = scripted_profile(tr) if cond == "scripted" else None
    rows = []
    for i, p in enumerate(tr):
        objs = [] if (p is None or isinstance(p, str)) else [Obj(1, "scary", p[0], p[1], None)]
        rin = 0.0
        if cond == "vision":
            lum = eye.render(FLY, QUAT, objs)
            lum = {k: np.asarray(quantise(v)) / 255.0 for k, v in lum.items()}  # what physics actually sends
            if p == "DARK1":
                lum = {k: np.zeros_like(v) for k, v in lum.items()}
            rates = T(lum, 0.0 if i == 0 else DTU)
            b.set_drive("vision", T.idx, rates)
            rin = float(rates.mean())
        elif cond == "scripted":
            for gname in ("looming_left", "lplc2_left"):
                r = prof[i].get(gname, 0.0)
                if r > 0: b.stimulate(groups[gname]["idx"], r)
                else: b.unstimulate(groups[gname]["idx"])
        b.run(UPD)
        c = b.take_frame_counts()
        row = {"i": i, "rin": round(rin, 2), "pr_rate": round(float(c[T.idx].sum()) / (len(T.idx) * DTU), 2),
               "all": round(float(c.sum()) / (len(c) * DTU), 3), "active": int((c > 0).sum())}
        for gname in G:
            k = groups[gname]["idx"]
            row[gname] = round(float(c[k].sum()) / (len(k) * DTU), 2)
            row[gname + "_n"] = int((c[k] > 0).sum())
        rows.append(row)
    return rows

def summ(rows):
    pre, ev = rows[5:N_PRE], rows[N_PRE:N_PRE + N_EV + 3]  # skip first 5 updates (adaptation onset)
    out = {}
    for k in ["pr_rate", "all", "active"] + G + [g + "_n" for g in G]:
        a = np.mean([r[k] for r in pre]); e = np.mean([r[k] for r in ev]); pk = max(r[k] for r in ev)
        out[k] = [round(a, 2), round(e, 2), round(pk, 2)]
    return out

if __name__ == "__main__":
    plan = json.loads(os.environ.get("PLAN", '[["vision","loom",0]]'))
    res = []
    for cond, kind, seed, *rest in plan:
        rest = list(rest)
        t0 = time.time()
        rows = run(cond, kind, seed, *(rest or []))
        s = summ(rows)
        res.append({"cond": cond, "kind": kind, "seed": seed, "extra": rest, "wall_s": round(time.time() - t0, 1), "summary": s, "rows": rows})
        print(json.dumps({"cond": cond, "kind": kind, "seed": seed, "wall": res[-1]["wall_s"], **{k: v for k, v in s.items() if not k.endswith("_n")}}), flush=True)
    json.dump(res, open(os.environ.get("OUT", "/tmp/measure_vision_loom.json"), "w"))
