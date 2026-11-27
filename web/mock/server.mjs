// Mock brain (:5031) and physics (:5032) for developing and testing the web UI.
// Shapes follow brain/README.md, physics/README.md, SPEC.md and ADDENDUM C. It is a stand-in, not a simulation:
// the steering/eating/mate rules are simplified, but every endpoint the web uses exists with the real shape.
// MOCK_FLY_DIR (optional): a snapshot of the real physics model.json, mesh_<id>.json and pose.json (one /stream frame),
// so the mock serves the real NeuroMechFly mesh. Without it a box-shaped stand-in fly is served.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer, WebSocket } from 'ws';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BRAIN_PORT = Number(process.env.MOCK_BRAIN_PORT ?? 5031);
const PHYS_PORT = Number(process.env.MOCK_PHYSICS_PORT ?? 5032);
const HOST = process.env.MOCK_HOST ?? '127.0.0.1';
const FLY_DIR = process.env.MOCK_FLY_DIR ?? '';
const CAPACITY = Number(process.env.MOCK_CAPACITY ?? 1.6); // pretend this computer tops out here
const NEURONS = 138639;
const readJson = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
// name -> {role, default_rate_hz, size, idx}; static/ is copied to build/client by the build (the image has no static/ dir)
const GI_PATH = [path.join(HERE, '../static/group_indices.json'), path.join(HERE, '../build/client/group_indices.json')].find((p) => fs.existsSync(p));
const GROUPS = readJson(GI_PATH).groups;

function json(res, code, body) { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(body)); }
async function readBody(req) { let s = ''; for await (const c of req) s += c; try { return s ? JSON.parse(s) : {}; } catch { return null; } }
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const wrap = (b) => Math.atan2(Math.sin(b), Math.cos(b));

// ---------------- brain ----------------
const P = {
  w_syn: { value: 0.275, min: 0, max: 2, step: 0.005, unit: '', desc: 'global synaptic weight scale', readonly: false },
  tau_m_ms: { value: 20, min: 1, max: 60, step: 0.5, unit: 'ms', desc: 'membrane time constant', readonly: false },
  v_th_mV: { value: -45, min: -70, max: -20, step: 0.5, unit: 'mV', desc: 'spike threshold', readonly: false },
  v_reset_mV: { value: -52, min: -90, max: -30, step: 0.5, unit: 'mV', desc: 'reset potential', readonly: false },
  tau_syn_ms: { value: 5, min: 0.5, max: 30, step: 0.5, unit: 'ms', desc: 'synaptic time constant', readonly: false },
  t_ref_ms: { value: 2.2, min: 0, max: 10, step: 0.1, unit: 'ms', desc: 'refractory period', readonly: false },
  poisson_rate_hz: { value: 20, min: 0, max: 200, step: 1, unit: 'Hz', desc: 'background kicks per neuron', readonly: false },
  dt_ms: { value: 0.1, min: 0.1, max: 0.1, step: 0.1, unit: 'ms', desc: 'time step (fixed)', readonly: true }
};
const stim = new Map(); // group -> rate_hz
const silenced = new Set();
let tMs = 0, brainSpeed = 1, motor = { forward: 0, turn: 0 };

const brainHttp = http.createServer(async (req, res) => {
  const u = new URL(req.url, 'http://x');
  const m = req.method;
  if (u.pathname === '/health') return json(res, 200, { ok: true, neurons: NEURONS, synapses: 15091983, sim_speed: brainSpeed * 0.99, sim_capacity: 2.04, speed_target: brainSpeed, sim_ms: tMs, threads: 8, device: 'cpu' });
  if (u.pathname === '/params' && m === 'GET') return json(res, 200, P);
  if (u.pathname === '/params' && m === 'PUT') {
    const b = await readBody(req);
    if (!b) return json(res, 400, { detail: 'bad json' });
    for (const [k, v] of Object.entries(b)) {
      if (!(k in P) || P[k].readonly) return json(res, 422, { detail: 'unknown or read-only ' + k });
      if (typeof v !== 'number' || v < P[k].min || v > P[k].max) return json(res, 422, { detail: 'out of range ' + k });
    }
    for (const [k, v] of Object.entries(b)) P[k].value = v;
    return json(res, 200, Object.fromEntries(Object.entries(P).map(([k, p]) => [k, p.value])));
  }
  if (u.pathname === '/groups') return json(res, 200, Object.fromEntries(Object.entries(GROUPS).map(([k, g]) => [k, { size: g.size, role: g.role, default_rate_hz: g.default_rate_hz, description: 'mock group' }])));
  if (u.pathname === '/stimulate' && m === 'POST') {
    const b = (await readBody(req)) ?? {};
    const g = GROUPS[b.group];
    if (!g) return json(res, 404, { detail: "unknown group '" + b.group + "'; see GET /groups" });
    const rate = b.rate_hz ?? g.default_rate_hz;
    if (!(rate >= 0 && rate <= 10000)) return json(res, 422, { detail: 'rate_hz out of range' });
    if (rate === 0) stim.delete(b.group); else { stim.set(b.group, rate); silenced.delete(b.group); }
    return json(res, 200, { ok: true, neurons: g.size, rate_hz: rate });
  }
  if (u.pathname === '/silence' && m === 'POST') {
    const b = (await readBody(req)) ?? {};
    if (!GROUPS[b.group]) return json(res, 404, { detail: 'unknown group' });
    silenced.add(b.group); stim.delete(b.group);
    return json(res, 200, { ok: true, neurons: GROUPS[b.group].size });
  }
  if (u.pathname === '/clear' && m === 'POST') { stim.clear(); silenced.clear(); return json(res, 200, { ok: true }); }
  if (u.pathname === '/reset' && m === 'POST') { tMs = 0; motor = { forward: 0, turn: 0 }; return json(res, 200, { ok: true }); }
  if (u.pathname === '/speed' && m === 'PUT') {
    const b = (await readBody(req)) ?? {};
    if (typeof b.target !== 'number' || b.target < 0) return json(res, 422, { detail: 'target' });
    brainSpeed = b.target;
    return json(res, 200, { target: brainSpeed });
  }
  json(res, 404, { detail: 'not found' });
});
const brainWss = new WebSocketServer({ server: brainHttp, path: '/stream' });

function brainTick() {
  const dt = 0.033 * (brainSpeed || 1);
  tMs += dt * 1000;
  const drive = (P.poisson_rate_hz.value / 20) * (P.w_syn.value / 0.275);
  const r = (g) => stim.get(g) ?? 0;
  const sugar = r('sugar_grn') > 0, loom = r('looming') > 0 || r('lplc2') > 0;
  const tf = clamp((sugar ? 0.55 : 0) + (r('forward_walk') / 200) * 0.5 + (r('escape_gf') > 0 ? 0.9 : 0), 0, 1) * (silenced.has('forward_walk') ? 0 : 1);
  const tt = clamp((r('turn_left') > 0 ? -0.6 : 0) + (r('turn_right') > 0 ? 0.6 : 0) + (loom ? 0.3 : 0), -1, 1);
  motor.forward += (tf - motor.forward) * 0.15;
  motor.turn += (tt - motor.turn) * 0.15;
  const group_rates = {};
  for (const g of Object.keys(GROUPS)) group_rates[g] = +((stim.get(g) ?? 0) * (0.6 + 0.4 * Math.random()) + 2 * drive * Math.random()).toFixed(2);
  const spikes = [];
  for (let i = 0; i < 120 + Math.round(200 * drive * Math.random()); i++) spikes.push(Math.floor(Math.random() * NEURONS));
  for (const [g, rate] of stim) {
    const idx = GROUPS[g].idx, k = Math.min(idx.length, Math.round(idx.length * Math.min(1, rate / 200) * 0.7 + 1));
    for (let i = 0; i < k; i++) spikes.push(idx[Math.floor(Math.random() * idx.length)]);
  }
  return { t_ms: tMs, motor, group_rates, spikes: spikes.slice(0, 2000), params: Object.fromEntries(Object.entries(P).map(([k, p]) => [k, p.value])), sim_speed: brainSpeed };
}
setInterval(() => { const m = JSON.stringify(brainTick()); for (const c of brainWss.clients) if (c.readyState === 1) c.send(m); }, 33);
brainHttp.listen(BRAIN_PORT, HOST, () => console.log('mock brain on', HOST, BRAIN_PORT));

// ---------------- physics ----------------
const fixture = (f) => { const p = path.join(FLY_DIR, f); return FLY_DIR && fs.existsSync(p) ? p : null; };
let MODEL, REST = null;
if (fixture('model.json')) {
  MODEL = readJson(fixture('model.json'));
  REST = fixture('pose.json') ? readJson(fixture('pose.json')).bodies : null;
} else {
  MODEL = {
    units: 'mm', quat_order: 'wxyz', fly: 'stand-in', timestep: 0.0002,
    bodies: [{ id: 0, name: 'world', parent: 0 }, { id: 1, name: 'nmf/c_thorax', parent: 0 }],
    geoms: [
      { id: 0, name: 'ground_plane', body: 0, type: 'plane', size: [1000, 1000, 1], pos: [0, 0, 0], quat: [1, 0, 0, 0], rgba: [0.5, 0.5, 0.5, 1], group: 0, mesh_id: null },
      { id: 1, name: 'nmf/c_thorax', body: 1, type: 'mesh', size: [1, 1, 1], pos: [0, 0, 0], quat: [1, 0, 0, 0], rgba: [0.5, 0.5, 0.5, 1], group: 2, mesh_id: 0 },
      { id: 2, name: 'nmf/c_head', body: 1, type: 'mesh', size: [1, 1, 1], pos: [0.9, 0, 0.1], quat: [1, 0, 0, 0], rgba: [0.5, 0.5, 0.5, 1], group: 2, mesh_id: 1 },
      { id: 3, name: 'nmf/c_abdomen4', body: 1, type: 'mesh', size: [1, 1, 1], pos: [-1.6, 0, -0.1], quat: [1, 0, 0, 0], rgba: [0.5, 0.5, 0.5, 1], group: 2, mesh_id: 2 }
    ],
    meshes: 3
  };
}
function fallbackMesh(id) {
  const s = [[0.5, 0.25, 0.25], [0.2, 0.2, 0.2], [0.9, 0.25, 0.22]][id % 3];
  const v = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]].map(([x, y, z]) => [x * s[0], y * s[1], z * s[2]]);
  const f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]];
  return { id, vertices: v, faces: f };
}
const qmul = (a, b) => [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2], a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
const qconj = (q) => [q[0], -q[1], -q[2], -q[3]];
const qrot = (q, v) => { const p = qmul(qmul(q, [0, v[0], v[1], v[2]]), qconj(q)); return [p[1], p[2], p[3]]; };
const qz = (a) => [Math.cos(a / 2), 0, 0, Math.sin(a / 2)];
// rest pose of each body relative to the thorax (body 1), from the real snapshot when available
const REL = new Map();
if (REST) {
  const T = REST.find((b) => b.id === 1);
  for (const b of REST) {
    const rp = qrot(qconj(T.quat), [b.pos[0] - T.pos[0], b.pos[1] - T.pos[1], b.pos[2] - T.pos[2]]);
    REL.set(b.id, { pos: rp, quat: qmul(qconj(T.quat), b.quat) });
  }
}
const BODY_IDS = REST ? REST.map((b) => b.id) : [0, 1];

const world = { objects: [], next: 1 };
let odorWind = { x: 0, y: 0 };
const pers = { hunger: 0.8, fear: 0.9, lust: 0.9, receptivity: 0.9 };
const body = { x: 0, y: 0, z: 1.02, yaw: 0 };
let speedF = 1, command = { forward: 0, turn: 0 }, brainMotor = { forward: 0, turn: 0 }, source = 'brain';
let state = { mode: 'idle', hunger_level: 0.8, nearest: null, drives: {} };
let simT = 0, lastBrainAt = 0;
const lastSent = new Map();

function brainCall(p, b) { return fetch('http://127.0.0.1:' + BRAIN_PORT + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b) }).catch(() => {}); }
const near = (list, x, y) => list.map((o) => ({ o, d: Math.hypot(o.pos[0] - x, o.pos[1] - y), b: wrap(Math.atan2(o.pos[1] - y, o.pos[0] - x) - body.yaw) }))
  .sort((a, b) => a.d - b.d)[0];

function physTick() {
  const dt = (1 / 30) * speedF;
  simT += dt;
  const drives = {};
  const setDrive = (g, rate) => { drives[g] = Math.max(drives[g] ?? 0, rate); };
  const sc = near(world.objects.filter((o) => o.kind === 'scary'), body.x, body.y);
  const ma = near(world.objects.filter((o) => o.kind === 'mate'), body.x, body.y);
  const fo = near(world.objects.filter((o) => o.kind === 'food'), body.x, body.y);
  const all = near(world.objects, body.x, body.y);
  let nearest = all ? { kind: all.o.kind, dist: all.d, bearing_deg: all.b * 180 / Math.PI } : null;
  // priority: fear > lust > hunger
  let mode = 'explore', turn = 0, fwd = 0.6, eat = false;
  if (sc && sc.d < 60) {
    const f = ((60 - sc.d) / 60) * pers.fear;
    mode = 'flee'; fwd = 1; turn = -Math.sign(sc.b || 1) * 1.2;
    setDrive('looming', 200 * f); setDrive('lplc2', 100 * f);
    if (sc.d < 20) setDrive('escape_gf', 150);
  } else if (ma && ma.d < 120 && pers.lust > 0.3) {
    setDrive('sugar_grn', 100 * pers.lust); // stand-in drive: half-rate approach, no feeding output
    if (pers.receptivity >= 0.5) {
      if (ma.d < 25) { mode = 'idle'; fwd = 0; } else { mode = 'chase'; fwd = 0.5; turn = clamp(ma.b * 1.5, -1, 1); }
    } else { mode = 'flee'; fwd = 1; turn = -Math.sign(ma.b || 1); }
  } else if (fo && fo.d < 120 && state.hunger_level > 0.3) {
    setDrive('sugar_grn', 200 * state.hunger_level);
    if (fo.d < 10) { mode = 'eat'; fwd = 0; eat = true; setDrive('feed_mn9', 60); }
    else { mode = 'chase'; fwd = 0.7; turn = clamp(fo.b * 1.5, -1, 1); }
  }
  if (mode === 'explore') turn = 0.3 * Math.sin(simT * 0.7);
  if (Math.hypot(body.x, body.y) > 220) turn = clamp(wrap(Math.atan2(-body.y, -body.x) - body.yaw) * 1.5, -1, 1); // stay near the counter
  state.hunger_level = clamp(state.hunger_level + (eat ? -0.25 : 0.01) * dt, 0, 1);
  if (eat && fo) {
    fo.o.amount = (fo.o.amount ?? 1) - 0.3 * dt;
    if (fo.o.amount <= 0) world.objects = world.objects.filter((o) => o !== fo.o);
  }
  // scripted steering plus the brain's drive
  command = { forward: fwd * (0.4 + 0.6 * brainMotor.forward), turn: clamp(turn + 0.3 * brainMotor.turn, -1, 1) };
  body.yaw += command.turn * 1.5 * dt;
  body.x += Math.cos(body.yaw) * command.forward * 30 * dt;
  body.y += Math.sin(body.yaw) * command.forward * 30 * dt;
  // brain stimulation: send only the groups whose rate changed; groups no longer driven go to 0
  for (const g of new Set([...Object.keys(drives), ...lastSent.keys()])) {
    const rate = Math.round((drives[g] ?? 0) * 10) / 10;
    if (lastSent.get(g) !== rate) { lastSent.set(g, rate); brainCall('/stimulate', { group: g, rate_hz: rate }); }
  }
  state = {
    mode, hunger_level: +state.hunger_level.toFixed(3),
    nearest: nearest ? { kind: nearest.kind, dist: +nearest.dist.toFixed(1), bearing_deg: +nearest.bearing_deg.toFixed(1) } : null,
    drives: Object.fromEntries(Object.entries(drives).map(([k, v]) => [k, +v.toFixed(1)]))
  };
  // every body = thorax pose composed with its rest offset (real snapshot), or thorax only
  const q = qz(body.yaw);
  const bodies = BODY_IDS.map((id) => {
    if (id === 0) return { id, pos: [0, 0, 0], quat: [1, 0, 0, 0] };
    const r = REL.get(id);
    if (!r) return { id, pos: [body.x, body.y, body.z], quat: q };
    const rp = qrot(q, r.pos);
    return { id, pos: [body.x + rp[0], body.y + rp[1], body.z + rp[2]], quat: qmul(q, r.quat) };
  });
  return { t: simT, bodies, command, source, objects: world.objects.map((o) => ({ ...o })), state };
}

function physConnectBrain() {
  const ws = new WebSocket('ws://127.0.0.1:' + BRAIN_PORT + '/stream');
  ws.on('message', (d) => { try { const m = JSON.parse(d); if (m.motor) { brainMotor = m.motor; lastBrainAt = Date.now(); } } catch {} });
  ws.on('close', () => setTimeout(physConnectBrain, 1000));
  ws.on('error', () => {});
}
setTimeout(physConnectBrain, 300);

const physHttp = http.createServer(async (req, res) => {
  const u = new URL(req.url, 'http://x');
  const m = req.method;
  const parts = u.pathname.split('/').filter(Boolean);
  if (u.pathname === '/health') return json(res, 200, { ok: true, sim_time: simT, speed_factor: speedF, speed: { target: speedF, achieved: speedF, capacity: CAPACITY }, command: { source, ...command }, brain: { brain_connected: Date.now() - lastBrainAt < 2000 } });
  if (u.pathname === '/model') return json(res, 200, MODEL);
  if (parts[0] === 'mesh' && parts[1] !== undefined && m === 'GET') {
    const id = Number(parts[1]);
    const f = fixture('mesh_' + id + '.json');
    if (f) return json(res, 200, readJson(f));
    if (FLY_DIR) return json(res, 404, { detail: 'no such mesh' });
    return json(res, 200, fallbackMesh(id));
  }
  if (u.pathname === '/reset' && m === 'POST') { Object.assign(body, { x: 0, y: 0, yaw: 0 }); state.hunger_level = 0.8; return json(res, 200, { ok: true }); }
  if (u.pathname === '/command' && m === 'POST') { source = 'manual'; return json(res, 200, { ok: true }); }
  // Odour: MOCK data (a Gaussian plume around the first food object, wind shifts it) in the same shape as the real physics /odor.
  if (u.pathname === '/odor' && m === 'GET') {
    const sp = u.searchParams.get('species') ?? 'food';
    if (!['food', 'cva'].includes(sp)) return json(res, 422, { detail: 'species must be one of food, cva' });
    const nx = 25, ny = 25, cell = 16, x0 = -200, y0 = -200, src = world.objects.find((o) => o.kind === (sp === 'food' ? 'food' : 'mate'));
    const c = []; let max = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = src ? Math.round(1e4 * Math.exp(-((x0 + i * cell - src.pos[0] - odorWind.x * 0.2) ** 2 + (y0 + j * cell - src.pos[1] - odorWind.y * 0.2) ** 2) / (2 * 40 * 40))) / 1e4 : 0;
      c.push(v); max = Math.max(max, v);
    }
    return json(res, 200, { species: sp, cell_mm: cell, x0_mm: x0, y0_mm: y0, nx, ny, max, c, wind: [odorWind.x, odorWind.y], antennae: { left: 0, right: 0 } });
  }
  if (u.pathname === '/odor/wind' && m === 'POST') {
    const b = (await readBody(req)) ?? {};
    if (!(Math.abs(b.x ?? 0) <= 300 && Math.abs(b.y ?? 0) <= 300)) return json(res, 422, { detail: 'wind out of range' });
    odorWind = { x: +(b.x ?? 0), y: +(b.y ?? 0) };
    return json(res, 200, { wind: [odorWind.x, odorWind.y] });
  }
  if (u.pathname === '/world' && m === 'GET') return json(res, 200, { objects: world.objects, personality: pers, state });
  if (u.pathname === '/world/objects' && m === 'POST') {
    const b = (await readBody(req)) ?? {};
    if (!['food', 'mate', 'scary'].includes(b.kind) || !Array.isArray(b.pos) || b.pos.length !== 2) return json(res, 422, { detail: 'kind/pos' });
    const o = { id: 'o' + world.next++, kind: b.kind, pos: [+b.pos[0], +b.pos[1]] };
    if (b.kind === 'food') o.amount = 1;
    world.objects.push(o);
    return json(res, 200, { id: o.id });
  }
  if (u.pathname === '/world/objects' && m === 'DELETE') { world.objects = []; return json(res, 200, { ok: true }); }
  if (parts[0] === 'world' && parts[1] === 'objects' && parts[2]) {
    const o = world.objects.find((x) => x.id === parts[2]);
    if (m === 'DELETE') { world.objects = world.objects.filter((x) => x.id !== parts[2]); return json(res, 200, { ok: true }); }
    if (!o) return json(res, 404, { detail: 'no such object' });
    if (m === 'PUT') {
      const b = (await readBody(req)) ?? {};
      if (!Array.isArray(b.pos) || b.pos.length !== 2) return json(res, 422, { detail: 'pos' });
      o.pos = [+b.pos[0], +b.pos[1]];
      return json(res, 200, o);
    }
  }
  if (u.pathname === '/personality' && m === 'GET') return json(res, 200, pers);
  if (u.pathname === '/personality' && m === 'PUT') {
    const b = (await readBody(req)) ?? {};
    for (const [k, v] of Object.entries(b)) if (!(k in pers) || typeof v !== 'number' || v < 0 || v > 1) return json(res, 422, { detail: 'personality ' + k });
    Object.assign(pers, b);
    return json(res, 200, pers);
  }
  if (u.pathname === '/speed' && m === 'PUT') {
    const b = (await readBody(req)) ?? {};
    if (typeof b.factor !== 'number' || b.factor < 0.1 || b.factor > 2) return json(res, 422, { detail: 'factor 0.1..2.0' });
    speedF = b.factor;
    brainCall('/speed', { target: speedF }).then(() => {});
    return json(res, 200, { factor: speedF, capacity: CAPACITY });
  }
  json(res, 404, { detail: 'not found' });
});
const physWss = new WebSocketServer({ server: physHttp, path: '/stream' });
setInterval(() => { const s = JSON.stringify(physTick()); for (const c of physWss.clients) if (c.readyState === 1) c.send(s); }, 33);
physHttp.listen(PHYS_PORT, HOST, () => console.log('mock physics on', HOST, PHYS_PORT, FLY_DIR ? 'with real fly snapshot' : 'stand-in fly'));
