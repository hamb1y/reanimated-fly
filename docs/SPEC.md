# flybrain MVP spec

Three containers, one `docker compose up`. Everything on localhost. Brain: CUDA if available (BRAIN_DEVICE=auto|cuda|cpu, CPU fallback, ~8x real time on GPU), physics: CPU only. install.sh runs the brain on the GPU when Docker can give a container the GPU, otherwise on the CPU (docker-compose.gpu.yml / docker-compose.cpu.yml).

```
brain :5031  --WS motor--> physics :5032 --WS pose--> web :5030 (SvelteKit, three.js)
   ^------------------- HTTP params/stim (proxied by web) ---------------'
```

## brain (:5031) — owner: brain/
Whole-brain LIF of the FlyWire connectome (Shiu et al., ~138k neurons, ~5M synapses), data in ../data (2025_Connectivity_783.parquet, 2025_Completeness_783.csv, sez_neurons.pickle). Reference implementation: reference/run_pytorch.py (LIF constants and update rule come from there; keep the model faithful). Python, FastAPI + uvicorn, torch CPU build. Free-running sim thread (dt 0.1 ms) targeting real time or as fast as CPU allows; report the achieved speed factor.
- GET /health -> {ok, neurons, synapses, sim_speed (sim_ms per wall_ms)}
- GET /params -> current; PUT /params (partial JSON). Hyperparams: w_syn (global synaptic weight scale), tau_m_ms, v_th_mV, v_reset_mV, tau_syn_ms, t_ref_ms, poisson_rate_hz (default drive), dt_ms (fixed 0.1, read-only). Changes apply live without restart.
- GET /groups -> named neuron groups available for stimulation/readout (at minimum: sugar-sensing GRNs, looming/visual, plus the descending-neuron groups used as motor readout: forward-walk (e.g. DNp09/oviDN-like groups — pick from sez_neurons / known walking DNs, document the choice in brain/README.md), turn-left, turn-right)
- POST /stimulate {group | neuron_ids, rate_hz} ; POST /silence {group | neuron_ids} ; POST /clear ; POST /reset
- WS /stream (30 Hz JSON): {t_ms, motor:{forward:0..1, turn:-1..1}, group_rates:{name:hz}, spikes:[[neuron_idx,...] recent window, capped 2000], params}
- Motor mapping: rolling 100 ms firing rate of readout groups, normalised by a documented constant, smoothed.

## physics (:5032) — owner: physics/
MuJoCo fly body via NeuroMechFly v2 (flygym, pip). Headless, no rendering, CPU. Python, FastAPI.
- Connects to ws://brain:5031/stream (env BRAIN_URL), uses motor.forward/turn as a descending command into a flygym walking controller (CPG/tripod gait from flygym examples; forward scales gait frequency/amplitude, turn biases left/right stride). Reconnect forever. If brain unreachable: hold still.
- GET /health ; GET /model -> JSON scene description for three.js: for every MuJoCo geom {id, body, type, size, pos, quat (relative to body), rgba, mesh_id?}; if meshes are used also GET /mesh/{id} (glTF or simple {vertices,faces}). Static geometry (floor) included.
- WS /stream (30 Hz): {t, bodies:[{id,pos:[3],quat:[4]}...] (world poses, ordered by body id), command:{forward,turn}}
- POST /reset ; POST /command {forward,turn} manual override (so physics is testable without brain). Physics steps at real time.

## web (:5030) — owner: web/
SvelteKit (adapter-node, TypeScript), three.js. Server side bridges: browser talks only to :5030. /api/brain/* -> HTTP proxy to brain; /ws/brain and /ws/physics -> WS proxies (env BRAIN_URL, PHYSICS_URL).
UI: left panel sliders for every hyperparam in GET /params (built from the response, debounced PUT), stimulate/silence controls from GET /groups, reset buttons, connection status dots, sim speed; main pane 3D three.js scene built from physics GET /model and updated from the pose stream, orbit controls; a small live motor-output (forward/turn) bar and spike-count/rate sparkline. Dark, clean, no AI-slop gradients.

## compose / repo
docker-compose.yml at root with three services (build contexts brain/ physics/ web/), ports 5030/5031/5032 bound to 127.0.0.1, brain mounts ../data read-only, healthchecks, restart unless-stopped, internal DNS names brain/physics/web. README.md with run instructions. Each service has its own Dockerfile, README.md, and tests that run in-container. Do NOT edit files outside your own directory except as stated; compose + root README are the orchestrator's.


## ADDENDUM 2026-11-26 — "world" + personality (demo; built after the real-time work)
Goal: coloured 3D world; user drags Food / Mate / Scary objects around live; fly reacts. Fly personality = "starving, lovesick coward".
- World objects live in physics (new world.py): POST/PUT/DELETE /world/objects {id,kind:food|mate|scary,pos}, WS stream includes objects. Web drags them (three.js) and sends positions.
- Sensing is GEOMETRIC (no flygym vision: too slow for real time): per object compute distance, bearing (left/right of heading), and for scary the apparent angular size growth (looming). Physics sends stimulation to the brain at ~15 Hz via POST /stimulate with distance-falloff rates:
  food -> sugar_grn (+ feed_mn9 when touching); scary -> looming + lplc2 (real loom detectors) -> escape_gf; mate -> STAND-IN (see below).
- Steering: brain gives drive (go/stop/speed/turn gain), a simple bearing controller gives direction (turn toward food/mate, away from scary). Label in UI/README what is brain vs scripted. Brain should not be claimed to do navigation by itself.
- Priority: fear > lust > hunger. Scary in range overrides everything (flee, back off). Mate in range suppresses hunger (single-minded). Hunger rises over time, drops while eating (food object shrinks), so it can get full.
- Personality dials in the UI (the "hyperparameters" of this demo): hunger, fear, lust (0..1) = gains on the stimulation groups + arbitration thresholds. Presets: Coward, Starving, Lovesick, Chill.
- Mate: FlyWire is a FEMALE connectome and the NeuroMechFly body is female; P1/courtship circuits are male-specific, so there is no real courtship circuit here. Mate drive is a stand-in (visual-object neurons + scripted pursuit + a visual 'song' ripple), clearly labelled as such. Keep it PG: pursuit, circling, wing-extension/song cue; no explicit behaviour.
- The player fly stays FEMALE (matches FlyWire + NMF body); Mate object = a MALE fly (reuse NMF mesh, different tint); optional autonomous wander/run-away.
- Look: warm kitchen/fruit-bowl scene (counter, banana, apple, light, soft shadows, sky), cheap three.js only. UI: black page, white rounded panels (small radius, padded), draggable panels, plain-language labels with technical names small.

### Addendum B (user ideas, same day)
- Mate stand-in drive: Mate object stimulates sugar_grn at a LOWER rate (approach) with feeding output (feed_mn9) suppressed for mate; 'receptivity' dial 0..1: at max she slows/stops when the male is close, at 0 she walks away. Food keeps full sugar + feeding.
- Connectome view: 3D point cloud (three.js Points, all 138,639 neurons) coloured/brightened by recent spikes with per-point decay, using the existing WS /stream 'spikes' list. NEEDS neuron 3D positions: not in data/. Find FlyWire public annotation tables with soma/position coordinates, match to flywire_id <-> neuron index (see brain/app/data.py), ship as a compact binary (float16/int16 xyz) in web/static. Camera orbit; optional group highlight (sugar, looming, motor DNs).
- Global time scale 0.1x..2x: one clock. Web sends the factor to brain (PUT /speed) AND physics (new PUT /speed). Cap the slider at min(2, measured capacity from brain /health.sim_capacity and physics headroom); show capacity. Slow-mo must keep brain and body in sync.

## ADDENDUM C — exact API for world + personality + time (physics owns it; web consumes it)
Units/axes: same as GET /model (see its 'units', MuJoCo z-up). Object positions are [x,y] on the ground plane, same units as pose frames.
physics (:5032) new:
- GET /world -> {objects:[{id,kind:'food'|'mate'|'scary',pos:[x,y],amount?,heading?}], personality:{hunger,fear,lust,receptivity}, state:{mode:'idle'|'eat'|'chase'|'flee'|'explore', hunger_level:0..1, nearest:{kind,dist,bearing_deg}|null, drives:{<brain group>:rate_hz}}}
- POST /world/objects {kind,pos:[x,y]} -> {id}; PUT /world/objects/{id} {pos:[x,y]} (called live while dragging, ~20/s); DELETE /world/objects/{id}; DELETE /world/objects (all)
- GET/PUT /personality {hunger?,fear?,lust?,receptivity?} each 0..1 (defaults: hunger 0.8 'starving', fear 0.9 'coward', lust 0.9, receptivity 0.9)
- PUT /speed {factor:0.1..2.0} -> {factor, capacity}; GET /health gains speed:{target,achieved,capacity}. Physics and brain must stay in sync: physics applies the factor to its own clock AND forwards it to brain PUT /speed.
- WS /stream frames gain: objects (same shape as GET /world objects, incl. food amount), state (same as GET /world state). Pose frame shape otherwise unchanged.
Behaviour (spec in ADDENDUM + B): geometric sensing at ~15 Hz -> POST brain /stimulate per group (rate 0 or /clear for groups no longer driven); steering by bearing rule; priority fear > lust > hunger; eating decrements food amount while the fly is within reach and nearly stopped (food disappears at 0, hunger_level drops); receptivity: when mate close, high = slow/stop, low = walk away. Mate drive uses sugar_grn at ~half rate with feed_mn9 NOT driven. Scary: looming + lplc2 rates from apparent angular-size growth, escape_gf when very close.
web (:5030) shows all of it; physics never renders.
