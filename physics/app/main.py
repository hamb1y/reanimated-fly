"""physics service (port 5032): MuJoCo NeuroMechFly v2 walker, world objects and personality, over HTTP and WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import brainloop
from . import obstacles as obst
from .behavior import Behavior
from .brain_client import brain_loop
from .brain_link import RECONCILE_S, BrainLink
from .command import CommandState
from .flysim import FlySim
from .groom import GroomTrigger
from .runner import SimRunner
from .world import WorldStore
from .vision_link import VisionLink

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("physics")

DEFAULT_BRAIN_URL = "ws://brain:5031/stream"  # motor frames (WebSocket)
DEFAULT_BRAIN_HTTP_URL = "http://127.0.0.1:5031"  # stimulation and speed (HTTP)
STREAM_HZ = 120  # matches runner.PUBLISH_HZ
WORLD_HZ = 15.0  # sensing and arbitration rate
BRAIN_IN_LOOP = os.environ.get("BRAIN_IN_LOOP", "1") != "0"  # speed and fleeing go through the brain (see brainloop.py)
RESET_TIMEOUT_S = 10.0
THORAX = "nmf/c_thorax"
SPEED_MIN, SPEED_MAX = 0.1, 2.0

Vec2 = Annotated[list[float], Field(min_length=2, max_length=2)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class CommandBody(BaseModel):
    forward: float = 0.0
    turn: float = 0.0
    release: bool = False  # true: drop the manual override and return to brain/hold


class ObjectIn(BaseModel):
    kind: Literal["food", "mate", "scary", "block"]
    pos: Vec2


class ObjectMove(BaseModel):
    pos: Vec2


class PersonalityIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hunger: Optional[Unit] = None
    fear: Optional[Unit] = None
    lust: Optional[Unit] = None
    receptivity: Optional[Unit] = None


class GroomIn(BaseModel):
    on: bool


class DustIn(BaseModel):
    on: bool
    rate_hz: float = Field(default=200.0, ge=1.0, le=500.0)  # drive on the Johnston's-organ group (measured: 200 Hz)


class SpeedIn(BaseModel):
    factor: float = Field(ge=SPEED_MIN, le=SPEED_MAX)


@asynccontextmanager
async def lifespan(app: FastAPI):
    flysim = await asyncio.to_thread(FlySim, int(os.environ.get("PHYSICS_FIDELITY_LEVEL", "0")))
    commands = CommandState()
    runner = SimRunner(flysim, commands)
    runner.start()
    store = WorldStore()
    behavior = Behavior(store)
    behavior.obstacle_sense = obst.ObstacleSense(flysim.props)  # kitchen props + user solids, ray sensing (app/obstacles.py)
    http_url = os.environ.get("BRAIN_HTTP_URL", DEFAULT_BRAIN_HTTP_URL).strip()
    link = BrainLink(http_url)
    vision = VisionLink.from_env(http_url, flysim.scene())  # real eyes -> brain /vision (app/vision_link.py)
    vision.start()
    brain_url = os.environ.get("BRAIN_URL", DEFAULT_BRAIN_URL).strip()
    brain_task: Optional[asyncio.Task] = None
    if brain_url:
        brain_task = asyncio.create_task(brain_loop(brain_url, commands))
        log.info("brain client enabled: %s", brain_url)
    else:
        log.info("brain client disabled (BRAIN_URL empty)")
    world_task = asyncio.create_task(world_loop(app))
    app.state.flysim = flysim
    app.state.commands = commands
    app.state.runner = runner
    app.state.store = store
    app.state.behavior = behavior
    app.state.link = link
    app.state.vision = vision
    app.state.thorax = flysim.body_index(THORAX)
    app.state.trigger = GroomTrigger()  # brain groom_adn1 rate -> yes/no (app/groom.py)
    app.state.groom_manual = False  # POST /groom {on:true}
    app.state.dust_hz = 0.0  # POST /dust: rate on jo_mechano while dust is on
    app.state.groom_info = {"adn1_hz": None, "on_delay_s": None, "dust_t": None}
    app.state.brain_url = brain_url or None
    try:
        yield
    finally:
        world_task.cancel()
        if brain_task is not None:
            brain_task.cancel()
        await link.close()
        await asyncio.to_thread(vision.stop)
        runner.stop()


app = FastAPI(title="flybrain physics", version="0.2.0", lifespan=lifespan)


async def world_loop(app: FastAPI) -> None:
    """15 Hz: sense, arbitrate, eat, steer (world command), and drive the brain (fire-and-forget)."""
    st = app.state
    period = 1.0 / WORLD_HZ
    last_reconcile = -1e9
    background: set[asyncio.Task] = set()
    while True:
        t0 = time.monotonic()
        try:
            frame = st.runner.latest()
            if frame is not None:
                has = len(st.store) > 0
                bf = st.commands.brain_forward(t0)
                in_loop = BRAIN_IN_LOOP and bf is not None
                r = st.behavior.tick(frame, st.thorax, 1.0 if in_loop else bf)  # in the loop, the request is not pre-scaled
                if BRAIN_IN_LOOP:
                    r = brainloop.apply(r, st.commands.brain_motor(t0))
                if r.get("steering") == "brain":  # the brain's flee turn also has to respect obstacles
                    r = st.behavior.refilter(r)
                st.flysim.pool.set(obst.user_footprints(st.store.objects()))  # user solids -> MuJoCo mocap pool
                st.commands.set_world(r["forward"], r["turn"], t0)  # empty world: she explores (wander), not idle
                drives = st.vision.filter(dict(r["drives"]))  # LOOMING_DRIVE=vision drops the scripted threat groups
                st.vision.submit(frame, st.thorax, st.store.objects())  # rendered and POSTed in the vision thread
                mode = r["mode"]
                grooming = _groom_tick(st, t0, drives)
                if grooming:
                    mode = "groom"
                st.store.set_state({
                    "mode": mode,
                    "hunger_level": r["hunger_level"],
                    "nearest": r["nearest"] if has else None,
                    "drives": drives,
                    "steering": r.get("steering", "rule"),
                    "grooming": st.flysim.groom.grooming,
                    "dust": st.dust_hz > 0,
                    "smell": r.get("smell"),
                    "avoid": r.get("avoid"),
                })
                st.link.send_desired(drives)
            if t0 - last_reconcile >= RECONCILE_S:
                last_reconcile = t0
                task = asyncio.create_task(st.link.reconcile(st.runner.target_speed))
                background.add(task)
                task.add_done_callback(background.discard)
        except Exception:  # noqa: BLE001 - keep the loop alive; the next tick retries
            log.exception("world tick failed")
        await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))


def _groom_tick(st, now: float, drives: dict) -> bool:
    """Dust drives the brain's Johnston's organ group; the brain's aDN1 rate (through GroomTrigger) decides when to groom.

    Adds jo_mechano to `drives` (in place). While grooming the walking neurons (p9) are not driven: the walk command is
    frozen in the body. Returns True while grooming is requested or still blending out.
    """
    groom = st.flysim.groom
    if st.dust_hz > 0:
        drives["jo_mechano"] = st.dust_hz
    rate = st.commands.group_rate("groom_adn1", now)
    was = st.trigger.state
    auto = st.trigger.update(rate, now)
    info = st.groom_info
    info["adn1_hz"] = rate
    if auto and not was and info["dust_t"] is not None:
        info["on_delay_s"] = round(now - info["dust_t"], 3)  # dust on -> grooming requested
    groom.request(st.groom_manual or auto)
    if groom.engaged:
        drives.pop("p9", None)
    return groom.engaged


def _capacity(st) -> float | None:
    vals = [v for v in (st.runner.capacity, st.link.sim_capacity) if v is not None]
    return round(min(vals), 3) if vals else None


@app.get("/health")
async def health():
    runner: SimRunner = app.state.runner
    commands: CommandState = app.state.commands
    link: BrainLink = app.state.link
    now = time.monotonic()
    motor, source = commands.effective(now)
    frame = runner.latest()
    return {
        "ok": True,
        "sim_time": None if frame is None else round(frame["t"], 5),
        "timestep": app.state.flysim.timestep,
        "speed_factor": round(runner.speed_factor, 3),
        "speed": {
            "target": round(runner.target_speed, 3),
            "achieved": round(runner.speed_factor, 3),
            "capacity": _capacity(app.state),
        },
        "fidelity": {
            "level": app.state.flysim.level,
            "timestep": app.state.flysim.timestep,
            "control_dt": round(app.state.flysim.control_dt, 6),
            "fallbacks": runner.fallbacks,
        },
        "command": {"source": source, "forward": motor.forward, "turn": motor.turn},
        "brain": {"url": app.state.brain_url, **commands.status(now)},
        "brain_http": {
            "url": link.base_url or None,
            "online": link.online,
            "sim_speed": link.sim_speed,
            "sim_capacity": link.sim_capacity,
            "speed_target": link.speed_target,
            "posts_ok": link.posts_ok,
            "posts_failed": link.posts_failed,
        },
        "world": {"objects": len(app.state.store), "mode": app.state.store.state["mode"]},
        "groom": _groom_status(app.state),
        "vision": app.state.vision.status(),
    }


@app.get("/vision")
async def get_vision():
    """Last rendered eye image (661 ommatidia per eye, 0..255) and the vision link status."""
    v = app.state.vision
    return {**v.status(), "image": v.image}


@app.get("/props")
async def get_props():
    """Obstacle footprints: the kitchen props (app/data/kitchen_props.json, the shared source) and the user-placed solids."""
    raw = app.state.flysim.props_raw
    return {**raw, "user": [f.as_dict() for f in obst.user_footprints(app.state.store.objects())],
            "pool": {"size": len(app.state.flysim.pool.bodies), "dropped": app.state.flysim.pool.dropped}}


@app.get("/model")
async def model():
    return JSONResponse(app.state.flysim.scene())


@app.get("/mesh/{mesh_id}")
async def mesh(mesh_id: int):
    data = app.state.flysim.mesh(mesh_id)
    if data is None:
        raise HTTPException(status_code=404, detail="no such mesh")
    return JSONResponse(data)


@app.post("/command")
async def command(body: CommandBody):
    commands: CommandState = app.state.commands
    if body.release:
        commands.release_manual()
    else:
        commands.set_manual(body.forward, body.turn)
    motor, source = commands.effective(time.monotonic())
    return {"source": source, "forward": motor.forward, "turn": motor.turn}


@app.post("/reset")
async def reset():
    runner: SimRunner = app.state.runner
    done = runner.request_reset()
    ok = await asyncio.to_thread(done.wait, RESET_TIMEOUT_S)
    if not ok:
        raise HTTPException(status_code=503, detail="reset timed out")
    app.state.behavior.reset_history()
    return {"ok": True}


# --- world and personality (ADDENDUM C) ----------------------------------------------------

@app.get("/world")  # also reports the arena (counter) size, mm
async def get_world():
    return app.state.store.snapshot()


@app.get("/odor")
async def get_odor(species: str = "food"):
    """Downsampled odour concentration (central +-200 mm) for the UI, plus the wind and what the antennae read."""
    if species not in odor_species():
        raise HTTPException(422, f"species must be one of {odor_species()}")
    b = app.state.behavior
    return {**b.odor.snapshot(species), "wind": list(b.odor.wind), "antennae": b.smell.get(species)}


class WindIn(BaseModel):
    x: float = Field(0.0, ge=-300.0, le=300.0)
    y: float = Field(0.0, ge=-300.0, le=300.0)


@app.post("/odor/wind")
async def post_wind(body: WindIn):
    app.state.behavior.odor.set_wind(body.x, body.y)
    return {"wind": [body.x, body.y]}


def odor_species():
    from .odor import SPECIES
    return SPECIES


@app.post("/world/objects", status_code=201)
async def add_object(body: ObjectIn):
    try:
        obj = app.state.store.add(body.kind, body.pos)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"id": obj.id}


@app.put("/world/objects/{oid}")
async def move_object(oid: int, body: ObjectMove):
    try:
        obj = app.state.store.move(oid, body.pos)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if obj is None:
        raise HTTPException(status_code=404, detail="no such object")
    return obj.as_dict()


@app.delete("/world/objects/{oid}")
async def delete_object(oid: int):
    if not app.state.store.delete(oid):
        raise HTTPException(status_code=404, detail="no such object")
    return {"ok": True}


@app.delete("/world/objects")
async def clear_objects():
    return {"ok": True, "removed": app.state.store.clear()}


@app.get("/personality")
async def get_personality():
    return dict(app.state.store.personality)


@app.put("/personality")
async def put_personality(body: PersonalityIn):
    return app.state.store.set_personality(**body.model_dump(exclude_none=True))


def _groom_status(st) -> dict:
    return {
        **st.flysim.groom.status(),
        "manual": st.groom_manual,
        "dust_hz": st.dust_hz,
        "trigger": st.trigger.state,
        "adn1_hz": st.groom_info["adn1_hz"],
        "adn1_smoothed_hz": round(st.trigger.ema, 3),
        "on_delay_s": st.groom_info["on_delay_s"],
    }


@app.get("/groom")
async def get_groom():
    return _groom_status(app.state)


@app.post("/groom")
async def post_groom(body: GroomIn):
    """Manual grooming. on=true forces it (replayed foreleg kinematics); on=false releases the manual request, and the
    brain-driven trigger (dust) can still start it."""
    app.state.groom_manual = body.on
    app.state.flysim.groom.request(body.on or app.state.trigger.state)
    return _groom_status(app.state)


@app.post("/dust")
async def post_dust(body: DustIn):
    """Dust on the antennae: physics drives the brain's jo_mechano group; the brain's groom_adn1 rate then starts grooming."""
    st = app.state
    if body.on:
        if st.dust_hz <= 0:
            st.groom_info["dust_t"] = time.monotonic()
            st.groom_info["on_delay_s"] = None
        st.dust_hz = body.rate_hz
    else:
        st.dust_hz = 0.0
        st.groom_info["dust_t"] = None
    return _groom_status(st)


@app.put("/speed")
async def put_speed(body: SpeedIn):
    """Global time scale: physics clock here, brain PUT /speed forwarded to the brain."""
    st = app.state
    st.runner.set_target(body.factor)
    synced = await st.link.set_speed(body.factor)
    return {"factor": body.factor, "capacity": _capacity(st), "brain_synced": synced}


# --- body and brain command ----------------------------------------------------------------

def _encode_frame(frame: dict, nbody: int) -> str:
    pos = np.round(frame["pos"][:nbody], 5).tolist()
    quat = np.round(frame["quat"][:nbody], 6).tolist()
    motor = frame["motor"]
    bodies = [{"id": i, "pos": p, "quat": q} for i, (p, q) in enumerate(zip(pos, quat))]
    return json.dumps(
        {
            "t": round(frame["t"], 5),
            "bodies": bodies,
            "command": {"forward": motor.forward, "turn": motor.turn},
            "source": frame["source"],
            "grooming": frame.get("groom", 0.0) > 0.05,
        },
        separators=(",", ":"),
    )


class _FrameEncoder:
    """Encodes each pose frame once; every WebSocket client shares the same string."""

    def __init__(self) -> None:
        self._frame: dict | None = None
        self._text: str | None = None

    def get(self, frame: dict, nbody: int) -> str:
        if frame is not self._frame:
            self._text = _encode_frame(frame, nbody)
            self._frame = frame
        return self._text


_encoder = _FrameEncoder()


@app.websocket("/stream")
async def stream(ws: WebSocket):
    """Pose frames (30 Hz sim time) with the world part appended: {..., objects, state}. Sent when either changes."""
    await ws.accept()
    st = app.state
    runner: SimRunner = st.runner
    store: WorldStore = st.store
    nbody = st.flysim.sim.mj_model.nbody
    last_frame = None
    last_version = -1
    try:
        while True:
            frame = runner.latest()
            version = store.version
            if frame is not None and (frame is not last_frame or version != last_version):
                last_frame, last_version = frame, version
                pose = _encoder.get(frame, nbody)
                await ws.send_text(pose[:-1] + "," + store.ws_part()[1:])
            await asyncio.sleep(0.5 / STREAM_HZ)  # poll at 2x so each new pose frame is sent once
    except Exception:  # client went away (disconnect or closed send)
        return
