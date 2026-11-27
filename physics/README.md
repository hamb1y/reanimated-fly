# physics (port 5032)

MuJoCo fly body: NeuroMechFly v2 via the official `flygym==2.1.1` (`flygym.compose`
world/fly, controllers from `flygym_demo.complex_terrain`). Headless (`MUJOCO_GL=egl`), CPU only.
Python 3.12, FastAPI + uvicorn. Image is about 1.3 GB, mostly flygym's numerics deps.

## Run

    docker build --network host -t flybrain-physics .
    docker run --rm -p 127.0.0.1:5032:5032 -e BRAIN_URL= flybrain-physics

Env:
- `BRAIN_URL` (default `ws://brain:5031/stream`). Motor frames over WebSocket. Empty string disables the brain client.
- `BRAIN_HTTP_URL` (default `http://127.0.0.1:5031`). Stimulation (`POST /stimulate`) and speed (`PUT /speed`) of the brain. Empty string disables it. With `network_mode: host` the default is right.
- `LOG_LEVEL` (default INFO).
- `PHYSICS_FIDELITY_LEVEL` (default `0`): starting fidelity level, see "Speed and fidelity".
- `VISION_TO_BRAIN` (default `0`), `VISION_HZ` (default `15`), `LOOMING_DRIVE` (default `scripted`): rendered vision to the brain, see "Real vision".

Tests run inside the image:

    docker run --rm --network host -e BRAIN_URL= --entrypoint python flybrain-physics -m pytest -q

## API

- `GET /health` -> `{ok, sim_time, timestep, speed_factor, command:{source,forward,turn}, brain:{url,manual,brain_connected,brain_last_motor_age_s}, fidelity:{level,timestep,control_dt,fallbacks}}`.
  `speed_factor` is sim seconds per wall second (measured over the last 2 s). `fidelity` is new in this
  version (additive); `fallbacks` counts automatic step-downs.
- `GET /model` -> scene JSON, static. `units: "mm"`, `quat_order: "wxyz"` (MuJoCo order).
  - `bodies[]`: `{id, name, parent}`, ids are indices into the pose stream.
  - `geoms[]`: `{id, name, body, type, size, pos, quat, rgba, group, mesh_id}`. `pos`/`quat` are relative to the body. `type` is `plane` or `mesh` here. A plane's `size` of 0 means infinite. `mesh_id` is null for non-mesh geoms.
  - `meshes[]`: counts only. Fetch geometry from `/mesh/{id}`.
- `GET /mesh/{id}` -> `{id, vertices: [[x,y,z]...], faces: [[a,b,c]...]}` (mesh-local, as MuJoCo stores it). 404 for unknown ids.
- `WS /stream` -> JSON text frames at 30 Hz, one per new pose frame, sent only when the sim advanced:
  `{t, bodies:[{id, pos:[x,y,z], quat:[w,x,y,z]}...], command:{forward,turn}, source}`.
  `t` is sim time in seconds. `bodies` is world pose for every body, ordered by id. Compose with a geom's `pos`/`quat` for the mesh transform. three.js wants `[x,y,z,w]`, so reorder.
  Each frame is encoded once and shared by all clients.
- `POST /command` body `{forward, turn}` sets a manual override (clamped: forward 0..1, turn -1..1). Body `{"release": true}` drops it. The override is sticky until released, and while it is set, brain motor frames are ignored.
- `POST /reset` re-creates the initial standing pose at t=0 (waits up to 10 s).

Added for the world and time scale (ADDENDUM C; all additive):
- `GET /health` gains `speed: {target, achieved, capacity}` (target = /speed factor; achieved = measured sim s per wall s; capacity = sim s per busy second, min'd with the brain's `sim_capacity`), `brain_http: {url, online, sim_speed, sim_capacity, speed_target, posts_ok, posts_failed}`, and `world: {objects, mode}`.
- `GET /world` -> `{objects:[{id, kind:'food'|'mate'|'scary', pos:[x,y], amount? (food)}], personality:{hunger,fear,lust,receptivity}, state:{mode, hunger_level, nearest:{kind,dist,bearing_deg}|null, drives:{group: rate_hz}}}`. `mode` is idle | explore | eat | chase | flee.
- `POST /world/objects` `{kind, pos:[x,y]}` -> `{id}` (201). Food starts at 100 units. Positions are mm, MuJoCo z-up, within +-450 mm (422 otherwise).
- `PUT /world/objects/{id}` `{pos}` -> the object (404 if unknown). Meant to be called ~20/s while dragging.
- `DELETE /world/objects/{id}` (404 if unknown); `DELETE /world/objects` -> `{ok, removed}`.
- `GET /personality`, `PUT /personality` `{hunger?, fear?, lust?, receptivity?}` each 0..1 (anything else 422). Defaults 0.8 / 0.9 / 0.9 / 0.9.
- `PUT /speed` `{factor}` with factor 0.1..2.0 (422 otherwise) -> `{factor, capacity, brain_synced}`. Sets the physics clock and forwards `PUT /speed {target}` to the brain.
- `WS /stream` frames gain `objects` and `state` (same shapes as `GET /world`). A frame is sent when the pose or the world changes. The pose fields are unchanged.

## Command priority and brain link

1. manual override (`POST /command`)
2. world command (scripted steering with a brain drive gain), only while world objects exist and the last world tick is <= 1 s old
3. brain motor frames, only while the brain socket is connected and the last frame is <= 1 s old. With no objects this is the original brain-driven walking, unchanged.
4. hold (zero command). This is the state with no brain and no manual command.

The brain client reconnects forever with exponential backoff (0.5 s to 5 s). Losing the socket drops the brain command immediately, so the fly stops.

## World, personality and time (ADDENDUM C)

Objects, personality and sensing live in `app/world.py` (store, geometry), `app/behavior.py` (arbitration, eating, hunger)
and `app/brain_link.py` (HTTP). The 15 Hz loop is in `app/main.py` (`world_loop`). Physics never renders.

### What comes from the brain and what is scripted

- **Brain, stimulation only.** Physics owns five groups and sends `POST /stimulate` for them: `sugar_grn`, `looming`,
  `lplc2`, `escape_gf`, `feed_mn9`. Other groups (for example one the web UI sets) are never touched. A group that is no longer
  driven is set to rate 0.
- **Brain, motor output as a gain only.** The forward value of the brain's own motor output (WS `/stream`, 0..1) scales the forward
  command: `gain = 0.5 + 0.5 * brain_forward`, and `gain = 0.5` with no brain. The brain does not set the turn for food or mate steering (scripted); while fleeing, the turn comes from the brain (physics/app/brainloop.py).
- **Scripted.** Everything else: which object is attended, the bearing rule (turn), the flee and approach decisions, eating, hunger,
  receptivity, and the stand-in mate drive. The brain does not navigate. Measured: the brain's motor forward was 0 in every world
  run below, so all walking there came from the 0.5 floor.

### Sensing (geometric, no vision)

The decisions below use geometry only. Real rendered vision for the brain is optional, see "Real vision".


- Position and heading come from the body pose: `nmf/c_thorax`, heading from the quaternion (+x is forward, as in the walking test).
- Distance (mm) is thorax to object on the ground plane. Bearing is the angle from heading to the object, in degrees, **positive = left**
  (the same sign as the motor `turn`, so a bearing rule is `turn = clamp(bearing / 45 deg, -1, 1)`).
- Scary objects (radius 5 mm): apparent angular size `theta = 2 atan(r / d)`. Growth `dtheta/dt` in simulation time, smoothed
  with a 0.3 s time constant. Only growth counts: receding gives 0. Looming and lplc2 follow growth/0.5 rad/s, clipped to 0..1.

### Arbitration (priority fear > lust > hunger)

| branch | trigger | command | stimulation (brain groups) |
|---|---|---|---|
| fear | scary within 60 mm, fear >= 0.2 | forward 1.0 x gain, turn away | looming = 200 x fear x growth; lplc2 = 100 x fear x growth; escape_gf = 100 x fear inside 10 mm |
| lust | mate within 80 mm, lust >= 0.2 | approach (1.0 x gain, turn toward). Receptivity >= 0.25 and within 12 mm: forward x (1 - receptivity), so it slows and stops. Receptivity < 0.25: walks away while in range | sugar_grn = 100 x lust x falloff (half rate). feed_mn9 is never driven |
| hunger | food within 60 mm, hunger x hunger_level >= 0.1 | approach (1.0 x gain). Within 3 mm of the thorax: stop, eat, turn to face it | sugar_grn = 200 x hunger x hunger_level x falloff; feed_mn9 = 100 x hunger x hunger_level when touching |
| none | objects present, none in range | forward 0.5 x gain, straight | none |

- Fleeing latches: once fleeing, the fly keeps fleeing until the scary object is beyond 90 mm (no flip-flop at the range edge).
- "Falloff" is linear: 1 at the near edge, 0 at the far edge. Drives under 0.5 Hz are not sent.
- The dials scale the stimulation (`hunger`, `fear`, `lust`) and set the thresholds above.

### Eating and hunger (simulation time)

- Eating needs the thorax within 3 mm of the food and speed under 2 mm/s (the fly is nearly stopped). The food loses 10 units/s.
  Food is removed at 0 units. Each unit eaten lowers hunger_level by 0.004. Hunger rises 0.01 per sim second, starting at 0.6.
- When hunger x dial drops below 0.1 the fly is full: it stops feeding, walks off to explore, and comes back when hungry again.

### Mate (stand-in, labelled as such)

FlyWire is a female connectome and the NeuroMechFly body is female, and there is no courtship circuit here. The mate drive is a stand-in:
the sugar_grn group at half rate, feed_mn9 never, and scripted pursuit or walking away. There is no explicit behaviour in physics.
Song and wing-extension cues are visual and belong to the web side.

### Time scale and brain sync

- `PUT /speed` changes the physics pacing target and forwards `PUT /speed {target}` to the brain. The physics checks the brain's
  `speed_target` and `sim_capacity` every 2 s (`GET /health`, `GET /state`) and re-sends the speed if they differ. This is the
  sync.
- `speed.capacity` = min(physics busy-time capacity, brain `sim_capacity`). The UI should cap the slider at capacity. Above it
  the brain and physics run at different speeds.

### Brain link (fire-and-forget)

- The 15 Hz loop computes the desired rates and hands them to `BrainLink.send_desired`. That schedules a POST only for groups
  whose rate changed by 1 Hz or more, or started or stopped. It awaits nothing, so the sim thread and the loop never block on HTTP.
  A POST has a 0.5 s timeout. A failed one is retried on the next tick. `posts_ok` and `posts_failed` are in `/health`.
- Every 2 s the link re-reads `GET /state` and believes the brain's stimulated rates, which repairs a brain restart or a `/clear`.

### Command rate and load

Pose frames still come from the sim thread. The world loop and the HTTP calls run on the asyncio loop, and the sim thread only
reads the command (a locked read, as before).

## Real vision (app/vision.py, app/vision_link.py)

The decisions in "Sensing" above stay geometric. In addition, physics can show the brain what she actually sees.

- `app/vision.py` renders both eyes analytically (661 ommatidia per eye, the lattice of brain/app/retina.py): rays from the eye
  geoms against the world objects as spheres (scary 0.05, mate 0.25, food 0.7 luminance), the counter (0.5) and a bright
  background (1.0). Her own body is ignored. Measured: 0.4-1.7 ms per render (both eyes) in the live container.
- `app/vision_link.py` runs it in its own thread at `VISION_HZ` (wall clock), never on the MuJoCo thread or the asyncio loop.
  The world loop only hands over the newest thorax pose and a copy of the objects. With `VISION_TO_BRAIN=1` every image is
  sent as `POST /vision {left, right, max: 255}` to the brain, which drives the photoreceptors R1-8 (see brain/README.md "Vision").
- `GET /vision` -> `{to_brain, looming_drive, hz_target, hz, renders, render_ms, posts_ok, posts_failed, last_post_age_s,
  brain_photoreceptors_active, brain_photoreceptor_mean_hz, image:{t, left[661], right[661]}}`. `GET /health` gains the same
  status as `vision` (without the image).

Env:
- `VISION_TO_BRAIN` (default `0`): 1 = POST the images to the brain. 0 = render only.
- `VISION_HZ` (default `15`).
- `LOOMING_DRIVE` (default `scripted`): `scripted` keeps the scripted threat stimulation (looming/LC4, lplc2, escape_gf from the
  geometric growth, see "Arbitration" and brainloop.py). `vision` removes those groups from the drives, so LC4 / LPLC2 / giant fibre
  can only be driven by the connectome processing the photoreceptor input (use with `VISION_TO_BRAIN=1`). The fear branch's
  decision to flee is a rule in both modes. `both` = `scripted`.

### Measured: does real vision drive the looming neurons? (2026-11-27)

No, not with the reference LIF constants. **`scripted` stays the default**, `vision` is for experiments.

Offline, full connectome, CPU, brain/tools/measure_vision_loom.py (fly still, heading +x; scary sphere r = 5 mm at bearing +40 deg
(left eye); 1 s static at 60 mm, then 60 -> 6 mm in 1 s, 0.5 s hold; images at 15 Hz quantised as physics sends them; brain
transduction as in POST /vision, local adaptation). Mean rate per neuron during approach + 0.2 s, Hz, 3 seeds (all three equal to
the shown precision):

| input | photoreceptors | LC4 left (looming_left) | LPLC2 left | escape_gf | turn_right / left |
|---|---|---|---|---|---|
| vision, loom | 17.1 (peak 54; ~3,100 of 10,582 active) | 0 | 0 | 0 | 0 / 0 |
| vision, recede / translate at 20 mm / empty | 0 / 5.0 / 0 | 0 | 0 | 0 | 0 |
| vision, whole field goes black (upper bound: all 10,582 R at 200 Hz) | 200 | 0 | 0 | 0 | 0 / 0 |
| scripted, loom (what physics sends today) | - | 113 (imposed) | 57 (imposed) | 98 (connectome response) | 24-28 / 0 |
| scripted, recede / translate | - | 69 / 120 | 35 / 61 | 70 / 118 | |

Why: LC4 and LPLC2 are 3 excitatory synapses from R1-8 in this graph. With the loom, 730 of 14,152 first-order neurons
(lamina/medulla) and 254 of 67,267 second-order neurons spiked; nothing at hop 3 or later. Diagnostic only (not a setting we use):
at w_syn 0.5 (1.8x reference) the black field gives LC4 0.6 Hz and the loom 0 Hz; at w_syn 1.0 (3.6x) the loom gives LC4 0.5 Hz
while 20,000 neurons light up and turn_right dominates whatever side the object is on, i.e. not a looming response.
The scripted path is not selective either: its "near" term drives the threat for a receding or sideways object too.

Live (2026-11-27, separate brain :5041 on CPU 6 threads + physics :5042 from this code, VISION_TO_BRAIN=1, 3 drags of a scary object
from 60 mm toward her at 54 mm/s): POST /vision at 8.7-15 Hz (posts_ok 549, failed 1), render 0.4-1.7 ms, up to all 10,582
photoreceptors at 200 Hz when the object reached her eye. `LOOMING_DRIVE=vision`: LC4, LPLC2 and escape_gf 0 Hz in every brain frame
(she still fled, by rule). `LOOMING_DRIVE=scripted`: LC4 left 90 Hz mean (peak 180), LPLC2 left 45, escape_gf 188 (peak 295; in this
mode escape_gf is also stimulated directly within 10 mm).

## Locomotion mapping (decisions)

- The tripod CPG walking controller (`HybridTurningController`, tutorial 4d) takes a
  `(left, right)` drive. `command.descending_drive`: `left = clamp(forward - turn, 0, 1) * 1.2`,
  `right = clamp(forward + turn, 0, 1) * 1.2`. Positive turn is left.
- Drive is non-negative on purpose. A negative drive, which walks one side backwards and spins
  in place, toppled the fly in testing (body height fell to 0.59 mm). With this mapping the
  minimum height stayed above 0.8 mm in every run below.
- Consequence: no spin on the spot. Forward 0 with nonzero turn gives a slow one-sided pivot.
- Stance: zero command means zero CPG amplitude. The fly holds its default standing pose.

## Speed and fidelity

The physics was about 0.24x real time on 8 cores (one busy thread). It is now real time with headroom
on one core of this desktop. Level 0 is the default and the one that gives real time.

Levels are `(physics timestep, physics steps per controller tick)` in `app/flysim.py`:

| level | timestep | controller tick | role |
|---|---|---|---|
| 0 | 0.2 ms | 2 ms (10 steps) | default, real time |
| 1 | 0.25 ms | 2 ms (8 steps) | fallback |

What changed from the original 0.1 ms per-step setup:
- **Controller every 2 ms instead of every 0.1 ms.** The CPG and retraction/stumbling rates are
  integrated with the controller's own timestep, so the gait keeps its speed. Joint targets are
  held between ticks (zero-order hold), which is what MuJoCo position actuators see anyway.
  The controller's persistence counter now counts 2 ms ticks, not 0.1 ms steps.
- **Physics timestep 0.2 ms instead of 0.1 ms.** Half the MuJoCo steps. Checked for stability (below).
- **Raw `mj_step` loop.** flygym's `Simulation.step` is just `mj_step`, so the same physics runs without the per-step Python wrapper.
- **One thread.** The sim runs in one thread. The `Dockerfile` sets `OMP_NUM_THREADS=1` and the BLAS/MKL/NumExpr thread counts to 1.
- **Pose path.** The sim publishes at 30 Hz of sim time. The WS stream encodes each frame once (vectorised) and every client shares that string.
- **Time scale.** `PUT /speed` sets the pacing target (sim seconds per wall second). Pacing is re-based on each change, so no catch-up burst. Capacity is measured on busy thread time, so it does not depend on the pacing.
- **Fallback.** If the speed factor stays below 0.9 times the target (target capped at 1.0) for 3 consecutive 2 s windows, the runner steps down one level (`set_level`, live, the fly keeps its state). A user-chosen 0.5x is therefore not mistaken for a slow CPU. It never steps back up by itself. `/health` shows the level and the fallback count.

Rejected settings (measured):
- Timestep 0.5 ms with 1 ms ticks: in 1 s of forward walk, yaw went to 86 deg and lateral drift to 16 mm.
- Timestep 1 ms: the fly fell (min height -0.66 mm, end height 0.08 mm).
- Controller at 4 ms or 1 ms ticks: no real speed gain beyond 2 ms. MuJoCo is the floor.
- MuJoCo options (solver iterations 15/30/100, energy flag off, CG solver): no measurable gain in noisy runs. Iteration limits did not change the trajectory, so the solver converges early. CG was slower. `noslip_iterations=0` gave about 20% less physics time and slightly different drift. Not adopted, because it changes friction behaviour.

## Measurements

Host: 24-thread Zen 5 desktop, shared with other workloads (1-minute load average 14 to 50 during the runs, so each number is about +/-20%). CPU-time table: one thread pinned to a lightly loaded core, flygym 2.1.1, MuJoCo 3.16, forward 1.0 walk. "CPU-s per sim-s" is thread CPU time / sim time, the number that predicts real time on another core.

### CPU, one thread (offline)

| config | timestep | controller tick | CPU-s per sim-s | sim speed on one core |
|---|---|---|---|---|
| original (per-step controller) | 0.1 ms | 0.1 ms | 4.5 | 0.22x |
| 0.1 ms, 1 ms ticks | 0.1 ms | 1 ms | 0.93 | 1.07x |
| **level 0 (default)** | **0.2 ms** | **2 ms** | **0.54** | **1.85x** |
| level 1 (fallback) | 0.25 ms | 2 ms | 0.46 | 2.16x |

### Live container, 8-CPU quota (`docker run --cpus 8`, BRAIN_URL empty, 25 s forward walk)

| build | speed_factor | stream rate | sim thread busy (top thread) | notes |
|---|---|---|---|---|
| before (original image) | 0.24x | 29.5 fps | 99.6% (one thread) | cannot reach real time |
| after, level 0 | 1.00x (paced) | 29.6 fps | 61% | capacity about 1.6x here |
| after, CPU quota 0.4 (fallback test) | 0.63x at level 1 | 19 fps | 39% | fell to level 1 after 6 s, as designed |

Only one thread is busy in every run (the next thread is at 0%).

### Fidelity checks (same controller, unchanged gait code)

- Level 0 over 20 s with a mixed command (forward 1 for 6 s; forward 1 + turn 0.5 for 4 s; forward 0.5 + turn -0.5 for 4 s; forward 1 for 6 s): no fall, minimum thorax height 0.75 mm. The 0.1 ms setting with 1 ms ticks gave 0.73 mm on the same run. Forward segments moved 93 mm and 90 mm, as commanded.
- Level 0 zero command: 1 s drift 0.7 mm, height stays at 1.03 mm.
- Level 1: 4 s forward walk, minimum height 0.91 mm, 65 mm forward.
- Straight-line walking wanders in yaw by about +/-10 deg over a few seconds (the original setting was 9 deg after 1 s). This comes from the controller and was not fixed here.

### GPU (flygym.warp GPUSimulation, one world, `flygym[warp]==2.1.1`)

- Result: not viable. Raw `mjw.step` cost was 5.9 to 8.1 ms per physics step on the shared RTX 5070 Ti, which another process held at 100% utilisation. That is 0.017x to 0.025x real time with no controller work.
- VRAM used by the single-fly process: about 0.29 GiB (within the 1 GiB limit).
- Not measured on an idle GPU. A single fly is launch-bound (hundreds of small kernels per step), and real time needs about 100 us per 0.1 ms step, so an idle GPU would also be unlikely to reach real time. The laptop's RTX 3050 Ti (4 GiB) would be slower than this card.
- Not added to `requirements.txt`.

### Laptop (not measured here)

A laptop (8 Zen 3 cores, about 1.3 to 1.5x slower per core than this desktop with throttling) is estimated to give about 1.2x on level 0 and about 1.4x on level 1. Level 0 has thin headroom there, which is what the fallback is for. Check it on the laptop with `GET /health`: `speed_factor` should stay at 1.0 with `fidelity.level` 0.

## Deviations from SPEC

- Timestep is 0.2 ms, not flygym's 0.1 ms, and the controller ticks every 2 ms. Both are needed for real time on 8 Zen 3 cores. Checked above.
- Quaternions are MuJoCo `wxyz`, as above. The web side must reorder them for three.js.
- `/mesh/{id}` returns the full mesh (up to ~55k vertices total across all meshes). Fetch once per mesh and cache.
- `/health` has the new `fidelity` object (additive). The SPEC.md contract is unchanged otherwise.

## Verification

Tests (46 in the image): `test_world.py` (sensing math, arbitration incl. the fear > lust > hunger order, mate sugar at half rate and no feed,
looming from growth, escape_gf inside 10 mm, eating and hunger, the flee latch), `test_brain_link.py` (diffed POSTs, zeroing, unreachable
brain, reconcile, speed sync, against a mock transport), `test_api_world.py` (CRUD and validation, personality, speed, WS frames, a real
turn toward food), plus the original command, brain client, API, fidelity tests.

    docker run --rm --network none -e BRAIN_URL= -e BRAIN_HTTP_URL= --entrypoint python flybrain-physics -m pytest -q
    # 46 passed

Offline: `docker run --network none` serves `GET /model` (70 geoms, 69 mesh geoms), `GET /mesh/0` (950 vertices), `/health`,
`POST /world/objects`. The meshes are in the image; nothing is downloaded at run time.

### End-to-end against the live brain (port 15032, brain on 5031)

Run with `BRAIN_URL=ws://127.0.0.1:5031/stream` and `BRAIN_HTTP_URL=http://127.0.0.1:5031`, on a standard host with the brain on CPU.
The host was loaded (the brain container used about 5.7 cores), so the numbers are ~20% noisy.

| scenario | measured |
|---|---|
| food 30 mm ahead, 90 deg left of the fly | fly turns to heading about 93 deg in about 1 s; thorax reaches 2.4 mm from food at t = 4 s; 100 units eaten by t = 14 s at 10 units/s; hunger 0.945 -> 0.685; food removed at 0 |
| scary 40 mm ahead | fly turns to face away in < 1 s (yaw about 165 deg in 1 s); distance 39 -> 61 mm at 3.5 s, -> 76 mm at 8 s (about 9 mm/s); then explores |
| scary dragged toward a still fly (about 20 mm/s) | looming 16-27 Hz and lplc2 8-13 Hz while approaching (smoothed); looming about 110 Hz at 5.7 mm; escape_gf 90 Hz below 10 mm |
| mate 60 mm ahead, receptivity 0.9 | approaches (about 9 mm/s), 60 -> 11 mm at t = 6 s; holds at 9.5-11 mm (forward 0.05); sugar_grn 26 Hz at 60 mm rising to 90 Hz close; feed_mn9 not driven |
| mate 60 mm ahead, receptivity 0.0 | turns to face away within 0.1 s; leaves the 80 mm range at about 3 s; then explores |
| `PUT /speed` 0.5 / 2.0 / 1.0 | brain and physics achieved 0.50 / 2.00 and 1.76 (capacity 1.76; brain at 2.0) / 1.00 |
| POST rate, food + mate + scary present | 4.9 POST/s on average over 10 s (only on rate changes; ceiling 75/s = 5 groups x 15 Hz); physics achieved 1.00x; brain sim_speed 0.98-1.02 |
| brain capacity on CPU | 1.5-1.8x idle (physics capacity during the same runs: 1.0-1.5x); about 0.93-1.0x with 50-70 Hz sugar_grn plus a moving fly, on this loaded host. Not measured on the GPU or on the laptop |

## Obstacles and collision (kitchen props + user blocks)

Files: `app/data/kitchen_props.json` (shared footprints), `app/obstacles.py` (footprints, ray sensing, steering filter),
`app/collide.py` (MuJoCo geoms), hooks in `flysim.py`, `behavior.py`, `main.py`.

- **Single source of truth.** `app/data/kitchen_props.json` lists the 16 counter props as ground footprints (circle or
  rotated box, height). Physics builds its geoms from it and serves it at `GET /props` (web: `PROPS_URL` in
  `web/src/lib/kitchen.ts`, through the `/api/physics` proxy). The art in kitchen.ts stays procedural; each prop's
  `kitchen_ts` is the line that places it, and `python3 physics/tools/check_kitchen_props.py` (stdlib, host) fails if
  the two drift. The footprints are authored simplifications read off kitchen.ts, not computed from the meshes. The sink
  is a low solid block (the physics floor has no hole, so she cannot fall in).
- **Collision (real MuJoCo contact).** One static geom per prop (group 4, invisible in the web, which draws mesh geoms only)
  and a pool of 12 mocap cylinders for user solids. Fly thorax, head, abdomen and coxa/femur/tibia collide with them
  (contype/conaffinity bit 2; flygym's ground pairs untouched). Tarsi do not: they carry the adhesion actuators and would
  glue her to walls. Low friction, so she slides along a wall.
- **User solids.** `POST /world/objects {kind:"block", pos}` adds a solid block (radius 8 mm, 16 mm high). Scary objects are
  solid too (radius 5 mm). Food and mates are not solid (she eats food; the mate is a pursued fly). More than 12 solids:
  the rest are still sensed but do not collide (`GET /props` `pool.dropped`). The web has no Block token yet.
- **Sensing, no map.** `ObstacleSense` casts 37 rays (10 deg apart, all round) from her head on the ground plane and
  returns only the distance along each ray, cut at 30 mm, against footprints inflated by 2.5 mm (half her leg span),
  plus two 2.5 mm "antenna" touch rays at +-30 deg. This is a geometric stand-in for parallax/looming range cues and
  antennal touch, not rendered vision and not a brain output. The goal object is excluded from its own sensing.
- **Steering (scripted).** `Avoider.filter` runs after the arbitration on every mode: if the direction she wants (goal
  bearing for eat/chase/flee, turn x 45 deg for explore/smell) is clear for min(goal distance, 14 mm) it changes nothing;
  otherwise it picks the clear ray closest to that direction on a sticky side (bug-style wall following) and flips side
  after 8 s without progress or once the detour has taken her 12 mm farther from the goal. Nose on a wall with nothing
  clear: crawl (forward <= 0.25) and turn to the most open ray. When the brain sets the flee turn (brainloop.py) the
  filter is applied again to the brain's turn. `state.avoid` in `GET /world` and the WS frames:
  `{blocked, detour_deg, side, front_mm, touch}`.
- **Arena.** With the props solid, the explorer uses the whole counter (`ARENA_HALF_X/Y` 285/185 mm, was 125/95).

Measured (this desktop; offline = FlySim + Behavior in-process, no brain; live = the running stack, brain on CPU at
0.39-0.45x real time):

| check | result |
|---|---|
| live, manual forward 1.0 straight at the fruit bowl (avoidance bypassed: contact only) | hits the rim at t = 7 s and slides round it; thorax never inside a footprint, closest 0.38 mm outside the r = 54 mm rim |
| offline, forward 1.0 straight at the cutting board | stopped at its edge (y about -67..-72, edge about -73) |
| live, block at (20, 0) between her and food at (42, 0) | detours on one side, reaches the food (2-3 mm) at t = 4 s; never inside the block |
| live, food 6 mm from the bowl | reaches it at t = 9 s; avoidance never needed to intervene |
| live, 90 s explore over the counter | 0 frames with the thorax inside any footprint; closest 3.6 mm (kettle) |
| offline, food 7.5 mm behind the jam jar (the jars + tray form a wall) | first side chosen leads away; flips side after 12 mm loss, goes round the jam jar, eats at t = 30 s |
| offline, food 7 mm behind the fruit bowl, start on the far side | follows the bowl rim (smell, then eat), eats at t = 28 s |
| cost | ~0.51-0.58 CPU-s per sim-s for FlySim with the geoms (0.51 before, same run-to-run noise); sensing is 37 analytic rays at 15 Hz |

Not working / limits:
- Live, she reaches food (near props or in the open) but hovers 2-3 mm around it without eating: with the brain in the loop on
  CPU (0.4x) the brain's forward output does not fall to zero, so she never stays under 2 mm/s. This happens with no props
  at all (food at (42, 0), no block: food 100 -> 100 in 30 s), so it is not caused by the obstacle code. Offline (no brain)
  she eats next to props.
- Local planner only: a goal fully enclosed by props, or in a deep concave pocket, can make her alternate sides.
- Feet (tarsi) can poke into a footprint; the body cannot. Box footprints are not corner-rounded in sensing (slightly conservative).
