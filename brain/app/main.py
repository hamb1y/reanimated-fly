"""flybrain brain service: FastAPI + free-running torch CPU LIF sim of the FlyWire connectome."""
import asyncio
import json
import math
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict

from . import data, retina
from .model import DEFAULT_PARAMS, DT, PARAM_META, make_brain
from .runner import MOTOR_NORM_HZ, Runner

STREAM_HZ = 30.0
state = {}


def build(data_dir=None, cache_dir=None, threads=None, speed_target=None):
    d, c = data.default_dirs()
    data_dir, cache_dir = data_dir or d, cache_dir or c
    torch.set_num_threads(threads or int(os.environ.get("TORCH_THREADS", "6")))
    ids = data.load_ids(data_dir)
    indptr, post, w = data.load_csc(data_dir, len(ids), cache_dir)
    groups = data.load_groups(ids)
    brain = make_brain(indptr, post, w, os.environ.get("BRAIN_DEVICE", "auto"),
                       gpu_mem_mib=int(os.environ.get("BRAIN_GPU_MEM_MIB", "1024")))
    st = float(os.environ.get("BRAIN_SPEED_TARGET", "1.0")) if speed_target is None else speed_target
    runner = Runner(brain, groups, st)
    order = np.argsort(ids)
    return {"ids": ids, "sorted_ids": ids[order], "order": order, "brain": brain, "groups": groups, "runner": runner,
            "vision": _build_vision(ids)}


def _build_vision(ids):
    """Photoreceptor map + transducer for POST /vision (None when app/retina.json is missing)."""
    if not retina.RETINA_JSON.exists():
        return None
    rmap = retina.load_map(ids, convention=os.environ.get("VISION_EYES", retina.EYE_CONVENTION))
    tr = retina.Transducer(rmap, mode=os.environ.get("VISION_ADAPT", retina.ADAPT_MODE),
                           subsample=float(os.environ.get("VISION_SUBSAMPLE", "1.0")))
    return {"map": rmap, "tr": tr, "last_t": None, "updates": 0, "active": 0, "mean_rate": 0.0, "wall": None}


async def _broadcaster():
    r = state["runner"]
    cond = state["cond"]
    while True:
        t0 = time.perf_counter()
        f = await asyncio.get_running_loop().run_in_executor(None, r.frame)
        state["latest"] = json.dumps(f, separators=(",", ":"))
        async with cond:
            cond.notify_all()
        await asyncio.sleep(max(0.0, 1.0 / STREAM_HZ - (time.perf_counter() - t0)))


@asynccontextmanager
async def lifespan(app):
    if "brain" not in state:
        state.update(await asyncio.get_running_loop().run_in_executor(None, build))
    state["cond"] = asyncio.Condition()
    state["latest"] = None
    state["runner"].start()
    task = asyncio.create_task(_broadcaster())
    yield
    task.cancel()
    state["runner"].stop()


app = FastAPI(title="flybrain brain", lifespan=lifespan)


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: str | None = None
    neuron_ids: list[int] | None = None
    rate_hz: float | None = None


def _resolve(t: Target):
    """-> (index array, label). neuron_ids may be tensor indices (< n) or FlyWire root ids."""
    if (t.group is None) == (t.neuron_ids is None):
        raise HTTPException(422, "give exactly one of 'group' or 'neuron_ids'")
    if t.group is not None:
        g = state["groups"].get(t.group)
        if g is None:
            raise HTTPException(404, f"unknown group '{t.group}'; see GET /groups")
        return g["idx"], t.group
    n = state["brain"].n
    arr = np.asarray(t.neuron_ids, dtype=np.int64)
    out = np.empty(len(arr), dtype=np.int64)
    big = arr >= n
    out[~big] = arr[~big]
    if big.any():
        s = state["sorted_ids"]
        pos = np.searchsorted(s, arr[big])
        pos[pos >= len(s)] = 0
        if not (s[pos] == arr[big]).all():
            raise HTTPException(422, "unknown FlyWire root id in neuron_ids")
        out[big] = state["order"][pos]
    if (out < 0).any():
        raise HTTPException(422, "negative neuron index")
    return np.unique(out), None


@app.get("/health")
def health():
    r = state["runner"]
    speed, cap = r.speed()
    return {
        "ok": True,
        "neurons": state["brain"].n,
        "synapses": state["brain"].n_syn,
        "sim_speed": round(speed, 3),
        "sim_capacity": round(cap, 3),
        "speed_target": r.speed_target,
        "sim_ms": round(state["brain"].t * DT, 1),
        "threads": torch.get_num_threads(),
        "device": state["brain"].device,
    }


@app.get("/params")
def get_params():
    """{key: {value, min, max, step, unit, desc, readonly}} (ranges as used by the web sliders)."""
    cur = state["brain"].params
    out = {}
    for k, m in PARAM_META.items():
        out[k] = {"value": cur[k], **{kk: vv for kk, vv in m.items() if kk != "read_only"}, "readonly": bool(m.get("read_only", False))}
        if m.get("read_only"):
            out[k].update(min=cur[k], max=cur[k], step=0)
    return out


@app.put("/params")
def put_params(body: dict):
    upd = {}
    for k, v in body.items():
        if k not in DEFAULT_PARAMS:
            raise HTTPException(422, f"unknown parameter '{k}'")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise HTTPException(422, f"parameter '{k}' must be a finite number")
        if k == "dt_ms":
            if abs(v - DT) > 1e-12:
                raise HTTPException(422, "dt_ms is fixed at 0.1 (read-only)")
            continue
        m = PARAM_META[k]
        if not (m["min"] <= v <= m["max"]):
            raise HTTPException(422, f"'{k}' must be within [{m['min']}, {m['max']}]")
        upd[k] = float(v)
    merged = {**state["brain"].params, **upd}
    if merged["v_th_mV"] <= merged["v_reset_mV"]:
        raise HTTPException(422, "v_th_mV must be above v_reset_mV")
    if upd:
        state["runner"].set_params(upd)
    return dict(state["brain"].params)


@app.post("/params/reset")
def reset_params():
    state["runner"].set_params({k: v for k, v in DEFAULT_PARAMS.items() if k != "dt_ms"})
    return dict(state["brain"].params)


@app.get("/groups")
def groups():
    return {
        name: {
            "size": int(len(g["idx"])),
            "role": g["role"],
            "default_rate_hz": g["default_rate_hz"],
            "description": g["description"],
        }
        for name, g in state["groups"].items()
    }


@app.get("/state")
def get_state():
    r, b = state["runner"], state["brain"]
    return {
        "stimulated_groups": dict(r.stim_groups),
        "silenced_groups": list(r.silent_groups),
        "stimulated_neurons": len(b.stim),
        "drive_channels": {c: int(len(d[0])) for c, d in list(b.drive.items())},
        "silenced_neurons": int(b.silent.sum()),
        "motor_norm_hz": MOTOR_NORM_HZ,
    }


@app.post("/stimulate")
def stimulate(t: Target):
    idx, label = _resolve(t)
    default = state["groups"][t.group]["default_rate_hz"] if t.group else 100.0
    rate = default if t.rate_hz is None else t.rate_hz
    if not (0 <= rate <= 10000):
        raise HTTPException(422, "rate_hz must be within [0, 10000] (0 removes stimulation)")
    state["runner"].stimulate(idx, rate, label)
    return {"ok": True, "neurons": int(len(idx)), "rate_hz": rate}


class DriveIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel: str = "default"
    neuron_ids: list[int]
    rates_hz: list[float]


def _ids_to_idx(arr):
    """Aligned conversion (no dedup): tensor indices (< n) pass, FlyWire root ids are looked up. 422 if unknown."""
    n = state["brain"].n
    arr = np.asarray(arr, dtype=np.int64)
    out = arr.copy()
    big = arr >= n
    if big.any():
        s = state["sorted_ids"]
        pos = np.searchsorted(s, arr[big])
        pos[pos >= len(s)] = 0
        if not (s[pos] == arr[big]).all():
            raise HTTPException(422, "unknown FlyWire root id in neuron_ids")
        out[big] = state["order"][pos]
    if (out < 0).any():
        raise HTTPException(422, "negative neuron index")
    return out


@app.post("/drive")
def drive(d: DriveIn):
    """Per-neuron rate vector on a named channel (replaces that channel; empty lists remove it). Kept apart from the
    group stimulation: /stimulate never touches a channel and vice versa; /clear drops both."""
    if len(d.neuron_ids) != len(d.rates_hz):
        raise HTTPException(422, "neuron_ids and rates_hz must have the same length")
    rate = np.asarray(d.rates_hz, dtype=np.float64)
    if rate.size and (not np.isfinite(rate).all() or rate.min() < 0 or rate.max() > 10000):
        raise HTTPException(422, "rates_hz must be within [0, 10000]")
    idx = _ids_to_idx(d.neuron_ids)
    if len(np.unique(idx)) != len(idx):
        raise HTTPException(422, "duplicate neuron in neuron_ids")
    k = state["runner"].drive(d.channel, idx, rate)
    return {"ok": True, "channel": d.channel, "driven": k}


class VisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: list[float]
    right: list[float]
    max: float = 1.0  # value that means full brightness (physics sends 0..255)


def _vision():
    v = state.get("vision")
    if v is None:
        raise HTTPException(404, "no photoreceptor map (app/retina.json); build it with tools/make_retinotopy.py")
    return v


@app.get("/vision")
def get_vision():
    v = _vision()
    tr = v["tr"]
    return {
        "ommatidia_per_eye": retina.N_OMM,
        "lattice": {"spacing_deg": retina.SPACING_DEG, "radius_deg": retina.RADIUS_DEG},
        "photoreceptors": {e: int(len(tr.eyes[e][0])) for e in ("left", "right")},
        "convention": v["map"]["meta"].get("convention"),
        "adapt_mode": tr.mode, "adapt_tau_s": tr.tau_s, "rate_max_hz": tr.rate_max, "gain": tr.gain,
        "updates": v["updates"], "active": v["active"], "mean_rate_hz": round(v["mean_rate"], 2),
        "validation": v["map"]["meta"].get("validation"),
    }


@app.post("/vision")
def post_vision(body: VisionIn):
    """One image per eye (661 ommatidium luminances each) -> photoreceptor Poisson rates (channel 'vision')."""
    v = _vision()
    if not (body.max > 0):
        raise HTTPException(422, "max must be > 0")
    if len(body.left) != retina.N_OMM or len(body.right) != retina.N_OMM:
        raise HTTPException(422, f"left and right need {retina.N_OMM} values each")
    lum = {"left": np.asarray(body.left, dtype=np.float64) / body.max, "right": np.asarray(body.right, dtype=np.float64) / body.max}
    if not (np.isfinite(lum["left"]).all() and np.isfinite(lum["right"]).all()):
        raise HTTPException(422, "values must be finite")
    t = state["brain"].t
    dt = 0.0 if v["last_t"] is None else min(max((t - v["last_t"]) * DT / 1000.0, 0.0), 0.5)  # adaptation runs on sim time
    v["last_t"] = t
    rates = v["tr"](lum, dt)
    k = state["runner"].drive("vision", v["tr"].idx, rates)
    v["updates"] += 1
    v["active"] = k
    v["mean_rate"] = float(rates.mean()) if rates.size else 0.0
    return {"ok": True, "active": k, "mean_rate_hz": round(v["mean_rate"], 2)}


@app.delete("/vision")
def delete_vision():
    v = _vision()
    state["runner"].drive("vision", [], [])
    v["tr"].reset()
    v["last_t"] = None
    v["active"] = 0
    return {"ok": True}


@app.post("/silence")
def silence(t: Target):
    idx, label = _resolve(t)
    state["runner"].silence(idx, label)
    return {"ok": True, "neurons": int(len(idx))}


@app.post("/clear")
def clear():
    state["runner"].clear()
    v = state.get("vision")
    if v is not None:  # the vision channel went with the clear; restart adaptation too
        v["tr"].reset()
        v["last_t"] = None
        v["active"] = 0
    return {"ok": True}


@app.post("/reset")
def reset():
    state["runner"].reset()
    return {"ok": True}


class Speed(BaseModel):
    target: float


@app.put("/speed")
def set_speed(s: Speed):
    if s.target < 0:
        raise HTTPException(422, "target must be >= 0 (0 = as fast as possible)")
    state["runner"].speed_target = s.target
    return {"ok": True, "speed_target": s.target}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    cond = state["cond"]
    last = None
    try:
        while True:
            try:
                async with cond:
                    await asyncio.wait_for(cond.wait_for(lambda: state["latest"] is not last), 5)
            except asyncio.TimeoutError:
                continue
            last = state["latest"]
            await ws.send_text(last)
    except (WebSocketDisconnect, RuntimeError):
        pass
