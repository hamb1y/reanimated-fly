<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import { World, TOKEN_COLORS, type ObjKind, type WorldObj } from '$lib/world';
  import { Connectome, loadGroupIndex, type GroupIndex } from '$lib/connectome';
  import { EyeView } from '$lib/eyes';
  import { j, put, post, del, parseParams, parseGroups, openStream, type ParamSpec, type GroupSpec } from '$lib/api';
  import { DIALS, PRESETS, DEFAULT_PERSONALITY, KIND_NAME, describeState, toCsv, type Pers, type Sample } from '$lib/status';
  import FloatPanel from '$lib/components/FloatPanel.svelte';
  import GroomPanel from '$lib/components/GroomPanel.svelte';

  const pct = (v: number) => Math.round(v * 100) + '%';
  const TOKENS: ObjKind[] = ['food', 'mate', 'scary'];
  const LABEL: Record<ObjKind, string> = { food: 'Food', mate: 'Mate', scary: 'Scary' };

  // ---- live state
  let brainWs = $state(false), brainHttp = $state(false), physWs = $state(false), physHttp = $state(false);
  const brainLive = $derived(brainWs && brainHttp);
  const bodyLive = $derived(physWs && physHttp);
  let brainHealth = $state<any>({}), physHealth = $state<any>({});
  let motor = $state({ forward: 0, turn: 0 });
  let spikeCount = $state(0);
  let groupRate = $state<Record<string, number>>({});
  let physState = $state<any>(null);
  let objects = $state<WorldObj[]>([]);
  let pers = $state<Pers>({ ...DEFAULT_PERSONALITY });
  let speed = $state(1);
  let capacity = $state<number | null>(null);
  const maxSpeed = $derived(Math.min(2, capacity ?? 2));
  let params = $state<ParamSpec[]>([]);
  let groups = $state<GroupSpec[]>([]);
  let rates = $state<Record<string, number>>({});
  let active = $state<Record<string, 'stim' | 'silence' | undefined>>({});
  let err = $state('');
  const wSyn = $derived(params.find((p) => p.key === 'w_syn'));

  // ---- interaction state
  let selected = $state<string | null>(null);
  let armed = $state<ObjKind | null>(null);
  let ghost = $state<{ kind: ObjKind; x: number; y: number } | null>(null);
  let followFly = $state(true);
  let showConn = $state(true);
  // ---- smell overlay: real concentration grid from physics GET /odor, polled at ~4 Hz while a species is selected
  let odorMode = $state<'off' | 'food' | 'cva'>('off');
  let odorMax = $state(0), odorAnt = $state<{ left: number; right: number } | null>(null);
  let windX = $state(0), windY = $state(0), windTimer: any = undefined;
  const fmtC = (v: number) => (v >= 0.01 ? v.toFixed(2) : v.toExponential(1));
  function sendWind() { clearTimeout(windTimer); windTimer = setTimeout(() => { windTimer = undefined; post('/api/physics/odor/wind', { x: windX, y: windY }).catch(() => {}); }, 120); }
  function calmWind() { windX = 0; windY = 0; sendWind(); }
  $effect(() => {
    const mode = odorMode;
    if (mode === 'off') { world?.setOdor(null, 'food'); odorAnt = null; odorMax = 0; return; }
    let stop = false, busy = false;
    const pull = async () => {
      if (busy || document.hidden) return;
      busy = true;
      try {
        const g = await j('/api/physics/odor?species=' + mode);
        if (stop) return;
        world?.setOdor(g, mode); odorMax = g.max ?? 0; odorAnt = g.antennae ?? null;
        if (Array.isArray(g.wind) && !windTimer) { windX = g.wind[0]; windY = g.wind[1]; }
      } catch { /* physics down: keep last frame */ } finally { busy = false; }
    };
    pull();
    const iv = setInterval(pull, 250);
    return () => { stop = true; clearInterval(iv); };
  });
  let showEyes = $state(true);
  let worldReady = $state(false);
  let eyeL = $state<HTMLCanvasElement>();
  let eyeR = $state<HTMLCanvasElement>();
  let chips = $state<string[]>([]);
  let chipNames = $state<string[]>([]);
  let hidden = $state(false);
  let rowCount = $state(0);
  let stageEl = $state<HTMLElement>();
  let viewHost = $state<HTMLElement>();
  let connHost = $state<HTMLElement>();
  let sparkEl = $state<HTMLCanvasElement>();
  let stageW = $state(1000), stageH = $state(700);
  const statusX = $derived(Math.max(14, stageW - 314));
  const connX = $derived(Math.max(14, stageW - 384));
  const connY = $derived(Math.max(14, stageH - 440));
  const eyesX = $derived(Math.max(14, (stageW - 340) / 2));
  const selObj = $derived(objects.find((o) => o.id === selected));
  const status = $derived(describeState(physState, objects));

  // ---- non-reactive helpers
  let world: World | undefined;
  let conn: Connectome | undefined;
  let lastPhys: any = null;
  let lastBrainT = 0;
  let modelOk = false;
  let firstSpeed = true;
  const hist: number[] = [];
  const timers: Record<string, any> = {};
  let persTimer: any = 0, persPending: Partial<Pers> = {};
  let speedTimer: any = 0;
  let dragId: string | null = null;
  let lastPutAt = 0, putTimer: any = 0, putPending: { id: string; pos: [number, number] } | null = null;
  const rows: Sample[] = [];
  const t0 = Date.now();

  // ---- brain params and groups (advanced)
  async function loadParams() { try { params = parseParams(await j('/api/brain/params')); } catch (e) { err = 'params: ' + e; } }
  async function loadGroups() {
    try { groups = parseGroups(await j('/api/brain/groups')); for (const g of groups) rates[g.name] ??= g.rate ?? 100; }
    catch (e) { err = 'groups: ' + e; }
  }
  function setParam(p: ParamSpec, v: number) {
    if (!isFinite(v)) return;
    p.value = v;
    clearTimeout(timers[p.key]);
    timers[p.key] = setTimeout(() => put('/api/brain/params', { [p.key]: v }).catch((e) => (err = String(e))), 150);
  }
  async function stim(g: string) { await post('/api/brain/stimulate', { group: g, rate_hz: rates[g] }); active[g] = 'stim'; }
  async function silence(g: string) { await post('/api/brain/silence', { group: g }); active[g] = 'silence'; }
  async function clearStim() { await post('/api/brain/clear'); active = {}; }
  async function resetBrain() { await post('/api/brain/reset'); active = {}; hist.length = 0; }
  async function resetBody() { try { await post('/api/physics/reset'); } catch (e) { err = 'reset: ' + e; } }

  // ---- model (fly mesh)
  async function loadModel() {
    try { const m = await j('/api/physics/model'); await world?.loadFly(m, (id) => j('/api/physics/mesh/' + id)); }
    catch (e) { err = 'model: ' + e; modelOk = false; }
  }

  // ---- personality dials and presets
  function setDial(k: keyof Pers, v: number) {
    pers[k] = v;
    persPending[k] = v;
    clearTimeout(persTimer);
    persTimer = setTimeout(flushPers, 120);
  }
  async function flushPers() {
    const body = persPending;
    persPending = {};
    try { await put('/api/physics/personality', body); } catch (e) { err = 'personality: ' + e; }
  }
  function preset(name: string) {
    const p = PRESETS[name];
    pers = { ...p };
    persPending = {};
    clearTimeout(persTimer);
    put('/api/physics/personality', p).catch((e) => (err = 'personality: ' + e));
  }

  // ---- time
  function setSpeed(v: number) {
    speed = v;
    clearTimeout(speedTimer);
    speedTimer = setTimeout(async () => {
      try { const r = await put('/api/physics/speed', { factor: v }); if (typeof r?.capacity === 'number') capacity = r.capacity; }
      catch (e) { err = 'speed: ' + e; }
    }, 120);
  }
  function updateCapacity() {
    const c = [physHealth?.speed?.capacity, brainHealth?.sim_capacity].filter((x) => typeof x === 'number') as number[];
    capacity = c.length ? Math.min(...c) : null;
    if (firstSpeed && typeof physHealth?.speed?.target === 'number') { speed = physHealth.speed.target; firstSpeed = false; }
  }
  // The target is never lowered automatically (capacity fluctuates with load); the label below warns instead.

  // ---- objects
  async function placeAt(kind: ObjKind, pos: [number, number]) {
    try { const r = await post('/api/physics/world/objects', { kind, pos }); if (r?.id != null) selected = String(r.id); }
    catch (e) { err = 'place: ' + e; }
  }
  async function deleteSelected() {
    const id = selected;
    if (id == null) return;
    selected = null;
    try { await del('/api/physics/world/objects/' + encodeURIComponent(id)); } catch (e) { err = 'delete: ' + e; }
  }
  async function clearWorld() {
    selected = null; armed = null;
    try { await del('/api/physics/world/objects'); } catch (e) { err = 'clear: ' + e; }
  }
  async function resetFly() { await resetBody(); }

  // Throttled PUT while dragging: at most one request per 50 ms (20/s), always the latest position.
  function queuePut(id: string, pos: [number, number]) {
    putPending = { id, pos };
    if (putTimer) return;
    putTimer = setTimeout(flushPut, Math.max(0, 50 - (performance.now() - lastPutAt)));
  }
  function flushPut() {
    putTimer = 0;
    const p = putPending;
    putPending = null;
    if (!p) return;
    lastPutAt = performance.now();
    put('/api/physics/world/objects/' + encodeURIComponent(p.id), { pos: p.pos }).catch(() => {});
  }

  function onCanvasDown(e: PointerEvent) {
    if (!world || e.button !== 0) return;
    if (armed) {
      const p = world.pickGround(e.clientX, e.clientY);
      if (p) placeAt(armed, p);
      armed = null;
      return;
    }
    const id = world.pickObject(e.clientX, e.clientY);
    selected = id;
    if (id) { dragId = id; world.setOrbit(false); try { world.canvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ } }
  }
  function onCanvasMove(e: PointerEvent) {
    if (!world || !dragId) return;
    const p = world.pickGround(e.clientX, e.clientY);
    if (!p) return;
    world.moveLocal(dragId, p);
    queuePut(dragId, p);
  }
  function onCanvasUp(e: PointerEvent) {
    if (!world || !dragId) return;
    const p = world.pickGround(e.clientX, e.clientY);
    if (p) { world.moveLocal(dragId, p); queuePut(dragId, p); }
    clearTimeout(putTimer); putTimer = 0; flushPut();
    dragId = null;
    world.setOrbit(true);
  }

  // Token: click arms it (next click in the view places it); drag drops it where released.
  function tokenDown(kind: ObjKind, e: PointerEvent) {
    const sx = e.clientX, sy = e.clientY;
    let moved = false;
    const move = (ev: PointerEvent) => {
      if (!moved && Math.hypot(ev.clientX - sx, ev.clientY - sy) > 6) moved = true;
      if (moved) ghost = { kind, x: ev.clientX, y: ev.clientY };
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      ghost = null;
      if (moved) {
        const r = world?.canvas.getBoundingClientRect();
        if (world && r && ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom) {
          const p = world.pickGround(ev.clientX, ev.clientY);
          if (p) placeAt(kind, p);
        }
      } else {
        armed = armed === kind ? null : kind;
      }
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  function onKey(e: KeyboardEvent) {
    const t = e.target as HTMLElement | null;
    if (t && t.closest && t.closest('input,textarea,select')) return;
    if ((e.key === 'Delete' || e.key === 'Backspace') && selected != null) { e.preventDefault(); deleteSelected(); }
    if (e.key === 'Escape') { armed = null; selected = null; }
  }

  function toggleFollow() { if (followFly) { followFly = false; world?.stopFollow(); } else { followFly = true; world?.followFly(45); } }
  function closeUp() { followFly = true; world?.followFly(20); }
  function frameKitchen() { followFly = false; world?.frameKitchen(); }
  function toggleChip(name: string) { chips = chips.includes(name) ? chips.filter((n) => n !== name) : [...chips, name]; }

  // ---- connectome panel: created while open, disposed on close
  $effect(() => {
    if (!showConn || !connHost) return;
    const c = new Connectome(connHost);
    conn = c;
    c.setGroups(untrack(() => chips));
    return () => { c.dispose(); if (conn === c) conn = undefined; };
  });
  $effect(() => { conn?.setGroups(chips); });
  $effect(() => { world?.setSelected(selected); });
  // ---- eye view: created while shown and the world exists, disposed otherwise
  $effect(() => {
    if (!showEyes || !worldReady || !eyeL || !eyeR || !world) return;
    const ev = new EyeView(world, eyeL, eyeR);
    return () => ev.dispose();
  });

  // ---- sparkline
  function drawSpark() {
    const c = sparkEl;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = (c.width = c.clientWidth * devicePixelRatio), h = (c.height = c.clientHeight * devicePixelRatio);
    ctx.clearRect(0, 0, w, h);
    if (hist.length < 2) return;
    const max = Math.max(10, ...hist) * 1.1;
    ctx.strokeStyle = '#e5574d';  // accent (lighter tint for lines on black)
    ctx.lineWidth = 1.5 * devicePixelRatio;
    ctx.beginPath();
    hist.forEach((v, i) => {
      const x = (i / 149) * w, y = h - 3 - (Math.min(v, max) / max) * (h - 6);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  function sample() {
    const lp = lastPhys ?? {};
    const cmd = lp.command ?? { forward: 0, turn: 0 };
    const pos = lp.bodies?.[1]?.pos ?? [0, 0, 0];
    rows.push({
      wall: (Date.now() - t0) / 1000, sim: lastBrainT / 1000, speed, mode: physState?.mode ?? '',
      hunger: physState?.hunger_level ?? null, cf: cmd.forward ?? 0, ct: cmd.turn ?? 0,
      mf: motor.forward, mt: motor.turn, x: pos[0], y: pos[1], z: pos[2],
      nk: physState?.nearest?.kind ?? '', nmm: physState?.nearest?.dist ?? null, nb: physState?.nearest?.bearing_deg ?? null,
      p: { ...pers }, drives: { ...(physState?.drives ?? {}) }, rates: { ...groupRate }
    });
    if (rows.length > 60000) rows.shift();
    rowCount = rows.length;
  }
  function exportCsv() {
    const url = URL.createObjectURL(new Blob([toCsv(rows)], { type: 'text/csv' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = 'flybrain-run-' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + '.csv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  onMount(() => {
    world = new World(viewHost!);
    world.setSelected(selected);
    worldReady = true;
    world.canvas.addEventListener('pointerdown', onCanvasDown);
    world.canvas.addEventListener('pointermove', onCanvasMove);
    world.canvas.addEventListener('pointerup', onCanvasUp);
    world.canvas.addEventListener('pointercancel', onCanvasUp);
    if (import.meta.env.MODE !== 'production') (window as any).__world = world; // non-production builds only

    loadParams(); loadGroups(); loadModel(); modelOk = true;
    loadGroupIndex().then((g) => (chipNames = Object.keys(g.groups))).catch(() => {});
    j('/api/physics/world').then((w) => {
      if (w?.personality) pers = { ...pers, ...w.personality };
      if (Array.isArray(w?.objects)) { objects = w.objects; world?.setObjects(w.objects); }
    }).catch(() => {});

    const offB = openStream('brain', (m) => {
      if (m.motor) motor = { forward: m.motor.forward ?? 0, turn: m.motor.turn ?? 0 };
      if (typeof m.t_ms === 'number') lastBrainT = m.t_ms;
      if (m.group_rates) groupRate = m.group_rates;
      const sp: number[] = Array.isArray(m.spikes) ? m.spikes.flat() : [];
      spikeCount = typeof m.n_spikes === 'number' ? m.n_spikes : sp.length;  // true count; the list itself is capped at 2000 for drawing
      hist.push(spikeCount); if (hist.length > 150) hist.shift(); drawSpark();
      conn?.pulse(sp);
    }, (o) => (brainWs = o));
    const offP = openStream('physics', (m) => {
      lastPhys = m;
      if (m.bodies) world?.setPose(m);
      if (Array.isArray(m.objects)) { objects = m.objects; world?.setObjects(m.objects, dragId); }
      if (m.state) physState = m.state;
    }, (o) => (physWs = o));

    const poll = setInterval(() => {
      j('/api/brain/health').then((h) => { brainHealth = h; brainHttp = true; updateCapacity(); }).catch(() => (brainHttp = false));
      j('/api/physics/health').then((h) => {
        physHealth = h; physHttp = true; updateCapacity();
        if (!modelOk) { modelOk = true; loadModel(); }
      }).catch(() => { physHttp = false; modelOk = false; });
      if (!params.length) loadParams();
      if (!groups.length) loadGroups();
    }, 2000);

    const sampler = setInterval(sample, 250);
    const ro = new ResizeObserver(() => { if (stageEl) { stageW = stageEl.clientWidth; stageH = stageEl.clientHeight; } });
    if (stageEl) ro.observe(stageEl);
    const onVis = () => (hidden = document.hidden);
    document.addEventListener('visibilitychange', onVis);
    onVis();

    return () => {
      offB(); offP(); clearInterval(poll); clearInterval(sampler); ro.disconnect();
      document.removeEventListener('visibilitychange', onVis);
      world?.dispose(); world = undefined;
    };
  });
</script>

<svelte:window onkeydown={onKey} />

<div class="shell">
  <section class="stage" bind:this={stageEl}>
    <div class="host" class:armed={armed !== null} bind:this={viewHost}></div>

    <FloatPanel title="Place things" x={14} y={14} w={236} z={4}>
      <div class="tokens">
        {#each TOKENS as k (k)}
          <button class="token" class:on={armed === k} onpointerdown={(e) => tokenDown(k, e)}>
            <i style:background={TOKEN_COLORS[k]}></i>{LABEL[k]}
          </button>
        {/each}
      </div>
      <p class="tech">Click a token, then click the view. Or drag it in. Drag things to move them. Select one and press Delete to remove it.</p>
      <div class="views">
        <button class:on={followFly} onclick={toggleFollow}>Follow</button>
        <button onclick={closeUp}>Close up</button>
        <button onclick={frameKitchen}>Kitchen</button>
      </div>
    </FloatPanel>

    <FloatPanel title="Smell" x={264} y={350} w={236} z={4}>
      <div class="views">
        {#each ['off', 'food', 'cva'] as m (m)}
          <button class:on={odorMode === m} onclick={() => (odorMode = m as typeof odorMode)}>{m === 'off' ? 'Off' : m === 'food' ? 'Food' : 'CVA'}</button>
        {/each}
      </div>
      {#if odorMode !== 'off'}
        <p class="tech">Heat on the counter: concentration, log scale (peak {fmtC(odorMax)}).{#if odorAnt}{" "}Antennae: L {fmtC(odorAnt.left)}, R {fmtC(odorAnt.right)}.{/if}</p>
      {/if}
      <div class="dial">
        <div class="dl"><label for="wx">Wind east</label><b>{windX.toFixed(0)}</b></div>
        <input id="wx" type="range" min="-300" max="300" step="10" bind:value={windX} oninput={sendWind} />
        <div class="dl"><label for="wy">Wind north</label><b>{windY.toFixed(0)}</b></div>
        <input id="wy" type="range" min="-300" max="300" step="10" bind:value={windY} oninput={sendWind} />
        <button onclick={calmWind} disabled={windX === 0 && windY === 0}>Calm</button>
      </div>
    </FloatPanel>

    <GroomPanel />

    <FloatPanel title="What the fly is doing" x={statusX} y={14} w={290} z={4}>
      <div class="st" class:alert={status.alert}>{status.title}</div>
      <div class="tech">{status.detail}</div>
      <div class="bar"><span>forward</span><div class="track"><i style:width={pct(motor.forward)}></i></div><b>{motor.forward.toFixed(2)}</b></div>
      <div class="bar"><span>turn</span><div class="track c"><i style:left={motor.turn < 0 ? 50 + motor.turn * 50 + '%' : '50%'} style:width={Math.abs(motor.turn) * 50 + '%'}></i></div><b>{motor.turn.toFixed(2)}</b></div>
      <canvas class="spark" bind:this={sparkEl}></canvas>
      <div class="sel">
        {#if selObj}
          <span>{KIND_NAME[selObj.kind]} selected</span>
          <button onclick={deleteSelected}>Delete</button>
        {:else}
          <span class="tech">Nothing selected. Brain spikes per frame: {spikeCount}</span>
        {/if}
      </div>
    </FloatPanel>

    {#if showEyes}
      <FloatPanel title="What the fly sees" x={eyesX} y={14} w={340} z={5} onclose={() => (showEyes = false)}>
        <div class="eyes">
          <div class="eye"><canvas bind:this={eyeL} width="150" height="150"></canvas><span>LEFT eye</span></div>
          <div class="eye"><canvas bind:this={eyeR} width="150" height="150"></canvas><span>RIGHT eye</span></div>
        </div>
        <p class="tech">The scene drawn from each eye's real spot on her head, about 700 dots (ommatidia) per eye, 120 degrees each. The same view is rendered by the physics service and sent to the brain's photoreceptors. The looming neurons do not respond to it (measured), so fleeing still uses a scripted looming signal as well.</p>
      </FloatPanel>
    {:else}
      <button class="eyesopen" onclick={() => (showEyes = true)}>Show what the fly sees</button>
    {/if}

    {#if showConn}
      <FloatPanel title="Connectome: 138,639 neurons" x={connX} y={connY} w={370} z={6} onclose={() => (showConn = false)}>
        <div class="conn" bind:this={connHost}></div>
        <div class="chips">
          {#each chipNames as name (name)}
            <button class="chip" class:on={chips.includes(name)} onclick={() => toggleChip(name)}>
              {name.replace(/_/g, ' ')}<small>{groupRate[name] != null ? Math.round(groupRate[name]) + ' Hz' : ''}</small>
            </button>
          {/each}
        </div>
        <p class="tech">Each dot is one neuron, placed at its FlyWire soma position. Flashes are spikes from the brain. Chips colour a group.</p>
      </FloatPanel>
    {/if}

    {#if hidden}
      <div class="banner top">Paused while this tab is in the background. The body keeps running; the view catches up when you come back.</div>
    {:else if !bodyLive}
      <div class="banner">Body offline. Start the physics service to see the fly.</div>
    {/if}
  </section>

  <aside class="col">
    <header class="top">
      <h1>Fly in a kitchen</h1>
      <div class="links">
        <span class="dot" class:on={brainLive}></span>Brain {brainLive ? 'live' : 'offline'}
        <span class="dot" class:on={bodyLive}></span>Body {bodyLive ? 'live' : 'offline'}
      </div>
    </header>
    {#if err}<div class="err">{err}</div>{/if}

    <section>
      <h2>The fly</h2>
      {#each DIALS as d (d.key)}
        <div class="dial">
          <div class="dl"><label for={'d-' + d.key}>{d.label}</label><b>{pct(pers[d.key])}</b></div>
          <input id={'d-' + d.key} type="range" min="0" max="1" step="0.05" value={pers[d.key]}
            oninput={(e) => setDial(d.key, e.currentTarget.valueAsNumber)} />
          <div class="tech">{d.tech}: {d.low} to {d.high}</div>
        </div>
      {/each}
      <div class="presets">
        {#each Object.keys(PRESETS) as name (name)}<button onclick={() => preset(name)}>{name}</button>{/each}
      </div>
    </section>

    <section>
      <h2>Brain and time</h2>
      {#if wSyn}
        <div class="dial">
          <div class="dl"><label for="ws">Brain strength</label><b>{wSyn.value.toFixed(3)}</b></div>
          <input id="ws" type="range" min={wSyn.min} max={wSyn.max} step={wSyn.step} value={wSyn.value}
            oninput={(e) => setParam(wSyn, e.currentTarget.valueAsNumber)} />
          <div class="tech">w_syn: how strongly brain cells push on each other</div>
        </div>
      {/if}
      <div class="dial">
        <div class="dl"><label for="speed">Time speed</label><b>{speed.toFixed(2)}×</b></div>
        <input id="speed" type="range" min="0.1" max="2" step="0.05" value={speed}
          oninput={(e) => setSpeed(e.currentTarget.valueAsNumber)} />
        <div class="tech">{#if capacity !== null && speed > capacity + 0.05}Too fast for this computer right now (about {capacity.toFixed(1)}× possible): it will play slower.{:else}This computer can do about {(capacity ?? 2).toFixed(1)}×.{/if} Body and brain run on one clock.</div>
      </div>
    </section>

    <section>
      <h2>Controls</h2>
      <div class="pair">
        <button onclick={resetFly}>Reset fly</button>
        <button onclick={clearWorld}>Clear world</button>
        <button class:on={showConn} onclick={() => (showConn = !showConn)}>{showConn ? 'Hide connectome' : 'Show connectome'}</button>
        <button onclick={exportCsv} disabled={rowCount === 0}>Export run (CSV)</button>
      </div>
      <div class="tech">{rowCount} rows recorded (4 per second)</div>
    </section>

    <section>
      <details>
        <summary>Advanced (for scientists)</summary>
        <div class="adv">
          <h3>Brain parameters</h3>
          {#each params as p (p.key)}
            <div class="param">
              <label for={'p-' + p.key} title={p.desc ?? ''}>{p.key}</label>
              <input class="num" type="number" step={p.step} value={p.value} disabled={p.readonly}
                onchange={(e) => setParam(p, e.currentTarget.valueAsNumber)} />
              <input id={'p-' + p.key} type="range" min={p.min} max={p.max} step={p.step} value={p.value} disabled={p.readonly}
                oninput={(e) => setParam(p, e.currentTarget.valueAsNumber)} />
            </div>
          {:else}<p class="tech">no parameters (brain offline?)</p>{/each}

          <h3>Brain groups</h3>
          {#each groups as g (g.name)}
            <div class="group" class:stim={active[g.name] === 'stim'} class:sil={active[g.name] === 'silence'}>
              <div class="gname"><span>{g.name}</span><small>{g.size ?? ''}{g.kind ? ' · ' + g.kind : ''}</small>
                <em>{groupRate[g.name] != null ? groupRate[g.name].toFixed(1) + ' Hz' : ''}</em></div>
              <div class="gctl">
                <input class="num" type="number" min="0" step="10" bind:value={rates[g.name]} title="rate_hz" />
                <button onclick={() => stim(g.name)}>stimulate</button>
                <button onclick={() => silence(g.name)}>silence</button>
              </div>
            </div>
          {:else}<p class="tech">no groups</p>{/each}
          <div class="pair">
            <button onclick={clearStim}>clear stim</button>
            <button onclick={resetBrain}>reset brain</button>
            <button onclick={resetBody}>reset body</button>
          </div>
          <div class="tech">brain {brainHealth.sim_speed != null ? 'sim speed ' + brainHealth.sim_speed.toFixed(2) + '×' : ''} · neurons {brainHealth.neurons?.toLocaleString() ?? '–'} · synapses {brainHealth.synapses?.toLocaleString() ?? '–'}</div>
        </div>
      </details>
    </section>
  </aside>
</div>

{#if ghost}
  <div class="ghost" style:left={ghost.x + 'px'} style:top={ghost.y + 'px'}>
    <i style:background={TOKEN_COLORS[ghost.kind]}></i>{LABEL[ghost.kind]}
  </div>
{/if}

<style>
  :global(:root) { --accent: #b3261e; --accent-text: #e5574d; }  /* the fly's eye red (the 3D eye colour is 0xb3261e) */
  :global(html, body) { margin: 0; height: 100%; background: #000; color: #fff; font: 14px/1.45 system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
  :global(button) { font: inherit; color: #fff; background: #000; border: 1px solid #fff; border-radius: 10px; padding: 6px 10px; cursor: pointer; }
  :global(button:hover:not(:disabled)) { background: var(--accent); border-color: var(--accent); color: #fff; }
  :global(button.on) { background: var(--accent); border-color: var(--accent); color: #fff; }
  :global(button:focus-visible, input:focus-visible) { outline: 2px solid var(--accent-text); outline-offset: 2px; }
  :global(button:disabled) { opacity: 0.35; cursor: default; }
  :global(input[type='range']) { width: 100%; accent-color: var(--accent); margin: 4px 0; }
  :global(input.num) { background: #000; color: #fff; border: 1px solid #fff; border-radius: 8px; padding: 3px 6px; font: inherit; width: 100%; box-sizing: border-box; }

  .shell { display: grid; grid-template-columns: minmax(0, 3fr) minmax(300px, 1fr); gap: 16px; height: 100vh; box-sizing: border-box; padding: 16px; background: #000; }
  .stage { position: relative; min-width: 0; min-height: 0; border-radius: 16px; overflow: hidden; background: #f4efe6; }
  .host { position: absolute; inset: 0; }
  .host :global(canvas) { display: block; touch-action: none; }
  .host.armed :global(canvas) { cursor: crosshair; }
  .col { overflow-y: auto; min-width: 0; padding: 0 6px 32px 0; }
  .top { display: flex; flex-direction: column; gap: 6px; }
  h1 { font-size: 22px; margin: 0; font-weight: 600; }
  h2 { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; margin: 22px 0 10px; font-weight: 600; }
  h3 { font-size: 13px; margin: 14px 0 6px; font-weight: 600; }
  .links { font-size: 13px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; border: 1px solid #fff; background: #000; margin-right: 2px; }
  .dot.on { background: var(--accent); border-color: var(--accent); }
  .err { margin-top: 8px; border: 1px solid #fff; border-radius: 10px; padding: 6px 8px; font-size: 12px; }
  .tech { font-size: 12px; opacity: 0.65; }
  .dial { margin-bottom: 14px; }
  .dl { display: flex; justify-content: space-between; align-items: baseline; }
  .dl b { font-variant-numeric: tabular-nums; font-weight: 500; }
  .views { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-top: 8px; }
  .views button { padding: 6px 4px; font-size: 12px; }
  .presets, .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }
  details summary { cursor: pointer; font-weight: 600; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; }
  .adv { margin-top: 8px; }
  .param { display: grid; grid-template-columns: 1fr 76px; gap: 2px 8px; margin-bottom: 10px; }
  .param input[type='range'] { grid-column: 1 / 3; }
  .param label { align-self: center; font-family: ui-monospace, monospace; font-size: 12px; }
  .group { border: 1px solid #fff; border-radius: 10px; padding: 7px 8px; margin-bottom: 7px; }
  .group.stim { border-width: 2px; } .group.sil { border-style: dashed; }
  .gname { display: flex; gap: 6px; align-items: baseline; }
  .gname span { font-family: ui-monospace, monospace; font-size: 12px; }
  .gname small { opacity: 0.6; } .gname em { margin-left: auto; font-style: normal; font-variant-numeric: tabular-nums; }
  .gctl { display: grid; grid-template-columns: 56px 1fr 1fr; gap: 5px; margin-top: 6px; }
  .tokens { display: grid; gap: 6px; }
  .token { display: flex; align-items: center; gap: 8px; text-align: left; }
  .token i, .ghost i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; border: 1px solid #fff; }
  .st { font-size: 20px; font-weight: 600; }
  .st.alert { color: var(--accent-text); text-decoration: underline; }
  .bar { display: grid; grid-template-columns: 56px 1fr 40px; gap: 8px; align-items: center; margin: 6px 0; font-size: 12px; }
  .bar b { font-variant-numeric: tabular-nums; font-weight: 500; text-align: right; }
  .track { position: relative; height: 8px; border: 1px solid #fff; border-radius: 4px; overflow: hidden; }
  .track i { position: absolute; top: 0; bottom: 0; left: 0; background: var(--accent); }
  .track.c::after { content: ''; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #fff; }
  .spark { width: 100%; height: 44px; display: block; margin-top: 6px; border-bottom: 1px solid #fff; }
  .sel { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 8px; font-size: 13px; }
  .conn { position: relative; height: 290px; border-radius: 10px; overflow: hidden; border: 1px solid #fff; background: #000; }
  .conn :global(canvas) { display: block; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .chip { font-size: 12px; padding: 3px 8px; display: inline-flex; gap: 6px; text-transform: none; }
  .chip small { opacity: 0.7; }
  .banner { position: absolute; left: 50%; transform: translateX(-50%); bottom: 14px; background: #000; color: #fff; border: 1px solid #fff; border-radius: 14px; padding: 8px 12px; font-size: 13px; max-width: 80%; box-sizing: border-box; z-index: 8; }
  .banner.top { top: 14px; bottom: auto; }
  .ghost { position: fixed; transform: translate(-50%, -120%); pointer-events: none; background: #000; border: 1px solid #fff; border-radius: 10px; padding: 4px 8px; display: flex; gap: 6px; align-items: center; z-index: 50; font-size: 13px; }
  .eyes { display: flex; gap: 10px; justify-content: center; }
  .eye { text-align: center; font-size: 11px; }
  .eye canvas { width: 150px; height: 150px; display: block; background: #000; border-radius: 50%; }
  .eyesopen { position: absolute; left: 50%; top: 14px; transform: translateX(-50%); z-index: 5; }
</style>
