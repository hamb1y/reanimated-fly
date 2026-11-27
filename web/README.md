# flybrain web (:5030)

SvelteKit (adapter-node, TypeScript) + three.js. The browser only talks to this server.

- /api/brain/* -> $BRAIN_URL/*, /api/physics/* -> $PHYSICS_URL/* (HTTP proxy, any method; 502 JSON if upstream is down)
- /ws/brain -> $BRAIN_URL/stream, /ws/physics -> $PHYSICS_URL/stream (WS proxies in server.js)
- Defaults: BRAIN_URL=http://brain:5031, PHYSICS_URL=http://physics:5032, PORT=5030, HOST=0.0.0.0

## Screen
- Black page, white text. One rounded view (16px radius) on the left, about 3/4 of the width: the 3D kitchen.
- Right column, plain words first, technical names small: The fly (How hungry, How scared, How lovesick, How receptive,
  with presets Coward, Starving, Lovesick, Chill), Brain and time (Brain strength = w_syn, Time speed capped by the
  reported capacity, shown as "max on this computer"), Controls (Reset fly, Clear world, Show connectome, Export run (CSV)),
  What the fly sees (two eye views), and a collapsed "Advanced (for scientists)" with every brain parameter and group.
- Floating panels inside the view, movable by their title bar: Place things (Food, Mate, Scary tokens; Follow, Close up,
  Kitchen), What the fly is doing (status from the physics stream state.mode, nearest object, motor bars, spikes sparkline,
  selected object with Delete), Connectome (toggle).
- Tokens: click one, then click the view; or drag it into the view. Drag any placed object on the table (ray to the z=0
  plane), PUT /world/objects/{id} throttled to 20/s with a final send on release. Select an object and press Delete.
- Esc clears the armed token and the selection.
- Tab hidden: the view pauses (browsers stop requestAnimationFrame) and a banner says so. The body keeps running.

## 3D
- Fly: the real NeuroMechFly mesh from GET /model and /mesh/{id} (69 mesh geoms, about 55k vertices, 115k triangles).
  The model ships every geom grey, so parts are coloured by MuJoCo name: body brown, abdomen amber, legs tan, eyes red,
  wings translucent. Geoms are merged per body and colour (about 50 meshes total). Body poses come from the stream.
- Mate: the same rig, tinted blue and scaled 0.92, posed from the player rig. Its heading is atan2 toward the player
  (the model faces +x, checked against the head geom position).
- Food: a red sphere whose scale follows amount (0.3 to 1.0). Scary: a red spider, merged geometry, scaled up as it
  gets closer. Selection ring: a flat ring under the selected object.
- Kitchen (procedural, no assets): wood counter top (large plane at z = 0, so the fly never walks off the wood), cream
  cabinet, mint wall with a window, fruit bowl with banana, apple and orange, cup, cutting board, plant pot. One warm
  directional light with one 1024 shadow map, a hemisphere fill.
- Render on demand: a frame is drawn when the pose, the objects, the camera or the selection changes (about 30 per
  second, the physics stream rate). pixelRatio is capped at 1.5. Measured in the dev build: 110 draw calls and about
  230k triangles per render (shadow pass included). GPU frame time on the target laptop is not measured here.

## Connectome view
- 138,639 neurons from neuron_positions.bin (int16 xyz, decoded with neuron_positions.json: normalised * scale + offset,
  centred and fitted to a unit box), additive points, brain index order.
- Spikes come from the brain stream: each spike sets a timestamp; the shader decays the flash (tau 0.35 s), so the CPU
  only writes one float per spike. The view orbits slowly and renders while flashing or orbiting.
- Group chips colour neurons of a group (sugar grn, looming, forward walk, turn left/right, feed mn9, ...) and show the
  live group rate from the brain stream.
- Decision: brain GET /groups has no neuron ids (only size, role, rate, description). The group membership is therefore
  generated into static/group_indices.json by tools/make_group_indices.py from brain/app/groups.json (FlyWire root ids
  mapped to brain tensor index via data/2025_Completeness_783.csv). 14 groups, 645 members, no misses. The brain could
  make this live with a GET /groups/{name}/neurons (or an "indices" field in /groups); then the static file can go.

## Physics and brain API used (ADDENDUM C)
- GET /api/physics/world, POST /world/objects, PUT/DELETE /world/objects/{id}, DELETE /world/objects
- GET/PUT /api/physics/personality, PUT /api/physics/speed {factor} -> {factor, capacity}
- WS /ws/physics frames: bodies, command, objects, state (mode, hunger_level, nearest, drives)
- Speed: the slider shows min(2, physics capacity, brain sim_capacity). The web talks to physics only; physics forwards to brain.

## Export run (CSV)
- 4 rows per second while the page is open (up to 60000 rows): wall_s, sim_t_s, speed_factor, mode, hunger_level,
  command_forward/turn, motor_forward/turn, fly_x/y/z_mm, nearest kind/mm/bearing, the four personality values, drives,
  and one rate_hz_<group> column per brain group.

## Mock (mock/server.mjs)
- npm run mock: brain on :5031, physics on :5032 (MOCK_BRAIN_PORT, MOCK_PHYSICS_PORT, MOCK_HOST). Same shapes as the real
  services. The steering, eating and mate rules are simplified stand-ins, not the physics.
- MOCK_FLY_DIR=<dir>: serve a snapshot of the real /model (model.json), /mesh/<id> (mesh_<id>.json) and one /stream
  frame (pose.json, sets the rest pose). Without it a box-shaped stand-in fly is served (used by the tests).
- MOCK_CAPACITY (default 1.6) sets the reported max speed, so the cap is visible.

## Tests
- npm test: starts the mock and the built server, checks the page labels, the production bundle has no debug handle,
  proxies and PUT roundtrips, both WS bridges, the world object lifecycle, personality and speed, and the static assets.
- The debug handle window.__world exists only in non-production builds (import.meta.env.MODE !== production).

## Run
npm install && npm run build
BRAIN_URL=http://127.0.0.1:5031 PHYSICS_URL=http://127.0.0.1:5032 HOST=127.0.0.1 npm start
docker build --network host -t flybrain-web .   # --network host: docker bridge DNS is broken on this machine
docker run --rm --network host --entrypoint npm flybrain-web test

Screenshots in shots/: 10-idle, 11-objects-placed, 12-dragging, 13-flee, 14-connectome, 15-kitchen, 16-tab-hidden-hint.
