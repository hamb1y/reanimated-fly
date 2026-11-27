// Smoke test: mock brain+physics, real built web server, checks page, proxies, WS bridges, world API, static assets.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { WebSocket } from 'ws';

const BP = 15031, PP = 15032, WP = 15030, procs = [];
const base = 'http://127.0.0.1:' + WP;
const run = (file, env) => { const p = spawn('node', [file], { env: { ...process.env, ...env }, stdio: 'ignore' }); procs.push(p); };
const wsFirst = (url) => new Promise((res, rej) => { const w = new WebSocket(url); const t = setTimeout(() => rej(new Error('timeout ' + url)), 5000); w.on('message', (d) => { clearTimeout(t); w.close(); res(JSON.parse(d)); }); w.on('error', rej); });
const get = (p) => fetch(base + p).then((r) => r.json());
const send = (p, method, body) => fetch(base + p, { method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
const sendJson = (p, method, body) => send(p, method, body).then((r) => r.json());

before(async () => {
  run('mock/server.mjs', { MOCK_BRAIN_PORT: BP, MOCK_PHYSICS_PORT: PP });
  run('server.js', { PORT: WP, HOST: '127.0.0.1', BRAIN_URL: 'http://127.0.0.1:' + BP, PHYSICS_URL: 'http://127.0.0.1:' + PP });
  await new Promise((r) => setTimeout(r, 1500));
});
after(() => procs.forEach((p) => p.kill()));

test('page is served with plain labels', async () => {
  const r = await fetch(base + '/'); assert.equal(r.status, 200);
  const html = await r.text();
  assert.ok(html.includes('flybrain'), 'title');
  for (const label of ['How hungry', 'How scared', 'How lovesick', 'How receptive', 'Export run']) assert.ok(html.includes(label), label);
});

test('production bundle has no debug world handle', () => {
  const dir = path.join(process.cwd(), 'build', 'client');
  const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
  for (const f of walk(dir).filter((f) => f.endsWith('.js'))) assert.ok(!fs.readFileSync(f, 'utf8').includes('__world'), f);
});

test('GET /api/brain/params + PUT roundtrip (brain shape)', async () => {
  const p = await get('/api/brain/params'); assert.ok('w_syn' in p);
  const q = await sendJson('/api/brain/params', 'PUT', { w_syn: 0.42 });
  assert.equal(q.w_syn, 0.42); assert.equal((await get('/api/brain/params')).w_syn.value, 0.42);
});
test('groups + model + mesh proxied', async () => {
  assert.ok(Object.keys(await get('/api/brain/groups')).length > 0);
  const m = await get('/api/physics/model'); assert.ok(m.geoms.length > 0);
  assert.ok((await get('/api/physics/mesh/0')).vertices.length > 0);
});
test('ws /ws/brain bridges motor frames', async () => { const m = await wsFirst('ws://127.0.0.1:' + WP + '/ws/brain'); assert.ok('motor' in m && Array.isArray(m.spikes)); });
test('ws /ws/physics bridges pose, objects and state', async () => {
  const m = await wsFirst('ws://127.0.0.1:' + WP + '/ws/physics');
  assert.ok(Array.isArray(m.bodies) && m.bodies[0].pos.length === 3);
  assert.ok(Array.isArray(m.objects)); assert.equal(typeof m.state.mode, 'string');
});
test('world objects: create, drag (PUT), delete one, clear all', async () => {
  const created = await sendJson('/api/physics/world/objects', 'POST', { kind: 'food', pos: [10, 20] });
  assert.ok(created.id);
  assert.deepEqual((await sendJson('/api/physics/world/objects/' + created.id, 'PUT', { pos: [30, -5] })).pos, [30, -5]);
  const w = await get('/api/physics/world'); assert.ok(w.objects.some((o) => o.id === created.id && o.pos[0] === 30));
  assert.equal((await send('/api/physics/world/objects/' + created.id, 'DELETE')).status, 200);
  assert.equal((await send('/api/physics/world/objects/nope', 'PUT', { pos: [0, 0] })).status, 404);
  await sendJson('/api/physics/world/objects', 'POST', { kind: 'scary', pos: [0, 0] });
  await send('/api/physics/world/objects', 'DELETE');
  assert.equal((await get('/api/physics/world')).objects.length, 0);
  assert.equal((await send('/api/physics/world/objects', 'POST', { kind: 'bogus', pos: [0, 0] })).status, 422);
});
test('personality PUT roundtrip and range check', async () => {
  assert.equal((await sendJson('/api/physics/personality', 'PUT', { hunger: 0.5 })).hunger, 0.5);
  assert.equal((await get('/api/physics/personality')).hunger, 0.5);
  assert.equal((await send('/api/physics/personality', 'PUT', { hunger: 1.5 })).status, 422);
});
test('speed PUT returns factor and capacity; out of range is refused', async () => {
  const r = await sendJson('/api/physics/speed', 'PUT', { factor: 1.2 });
  assert.equal(r.factor, 1.2); assert.equal(typeof r.capacity, 'number');
  assert.equal((await send('/api/physics/speed', 'PUT', { factor: 3 })).status, 422);
});
test('static assets: connectome positions and group indices', async () => {
  const bin = await fetch(base + '/neuron_positions.bin'); assert.equal(bin.status, 200);
  assert.equal((await bin.arrayBuffer()).byteLength, 138639 * 3 * 2);
  const gi = await get('/group_indices.json');
  assert.equal(gi.groups.sugar_grn.idx.length, 21);
});
test('odour grid and wind are proxied (shape of physics /odor)', async () => {
  for (const sp of ['food', 'cva']) {
    const g = await get('/api/physics/odor?species=' + sp);
    assert.equal(g.c.length, g.nx * g.ny); assert.ok(g.cell_mm > 0 && 'max' in g && Array.isArray(g.wind) && 'antennae' in g);
  }
  assert.equal((await send('/api/physics/odor?species=bogus', 'GET')).status, 422);
  assert.deepEqual((await sendJson('/api/physics/odor/wind', 'POST', { x: 120, y: -40 })).wind, [120, -40]);
  assert.deepEqual((await get('/api/physics/odor?species=food')).wind, [120, -40]);
  assert.equal((await send('/api/physics/odor/wind', 'POST', { x: 999, y: 0 })).status, 422);
  await sendJson('/api/physics/odor/wind', 'POST', { x: 0, y: 0 });
});
// The heat mapping is pure TS; only runs from a source checkout (the production image ships no src/).
test('odour heat mapping: log scale, transparent below floor, per-species ramp', { skip: !fs.existsSync('src/lib/odorheat.ts') }, async () => {
  const h = await import(path.resolve('src/lib/odorheat.ts'));
  assert.equal(h.logLevel(0, 1), 0); assert.equal(h.logLevel(1e-4, 1), 0); assert.equal(h.logLevel(1, 1), 1);
  assert.ok(Math.abs(h.logLevel(Math.sqrt(1e-3), 1) - 0.5) < 1e-9, 'log midpoint');
  assert.equal(h.heatRGBA(0, 'food')[3], 0);
  assert.ok(h.heatRGBA(1, 'food')[3] > h.heatRGBA(0.1, 'food')[3]);
  assert.notDeepEqual(h.heatRGBA(0.5, 'food'), h.heatRGBA(0.5, 'cva'));
  const out = new Uint8Array(16); h.fillHeat({ nx: 2, ny: 2, c: [0, 0.001, 0.1, 1], max: 1, cell_mm: 16, x0_mm: 0, y0_mm: 0 }, 'food', out);
  assert.equal(out[3], 0); assert.equal(out[7], 0); assert.ok(out[11] > 0 && out[15] >= out[11]);
  assert.equal(h.validGrid({ nx: 2, ny: 2, c: [1, 2, 3], cell_mm: 1 }), false);
});
test('unknown service is 404', async () => { assert.equal((await fetch(base + '/api/bogus/x')).status, 404); });
