import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { buildKitchen } from './kitchen';
import { OdorOverlay } from './odor';
import { createSpider } from './spider';
import { createFood } from './food';
import type { OdorGrid, OdorSpecies } from './odorheat';

let MAX_ANISO = 4;
type Snap = { t: number; arr: number; m: Map<number, { p: number[]; q: number[] }> };
const RENDER_DELAY_S = 0.05;  // playback delay of the pose jitter buffer
const FLOOR_LIFT_MM = 0.015;  // the leg meshes dip ~0.01-0.02 mm into the floor in the simulation; lift the drawn fly so feet rest on it

export type ObjKind = 'food' | 'mate' | 'scary';
export type WorldObj = { id: string; kind: ObjKind; pos: [number, number]; amount?: number };
type MeshJson = { vertices: number[][]; faces: number[][] };
type Part = { body: number; key: string; color: number; opacity: number; geo: THREE.BufferGeometry };
type Anim = { update(dt: number, moving: boolean): void; dispose(): void };
type FoodModel = { setAmount(a: number): void; dispose(): void };
type View = { kind: ObjKind; group: THREE.Group; ring: THREE.Mesh; rig?: FlyRig; pos: [number, number]; amount: number; anim?: Anim; food?: FoodModel };

export const TOKEN_COLORS: Record<ObjKind, string> = { food: '#f2683d', mate: '#5b7fd6', scary: '#d7261e' };

// Kitchen palette: warm, flat, cheap (MeshStandardMaterial, one shadow-casting light).
const KIT = {
  bg: 0xf4efe6, table: 0xc8966a, cabinet: 0xf1e6cf, wall: 0xb9e0cf, trim: 0xfffaf0, sky: 0xcbe8ff,
  bowl: 0x2a9d8f, apple: 0xd8453a, orange: 0xf28c28, banana: 0xf6cf3e, cup: 0xef7f74, pot: 0xc4704b,
  plant: 0x6cb85a, board: 0xe6bd7e, stem: 0x6b4a2b, handle: 0x9a8a72
};
const MATE_TINT = 0x4d6fd0;
const MATE_SCALE = 0.92;
const ONE = new THREE.Vector3(1, 1, 1);

/** The model ships every geom grey; colour parts by their MuJoCo name. */
export function partLook(name: string): { key: string; color: number; opacity: number } {
  const n = name.toLowerCase();
  if (/eye/.test(n)) return { key: 'eye', color: 0xb3261e, opacity: 1 };
  if (/wing/.test(n)) return { key: 'wing', color: 0xd7ebf5, opacity: 0.4 };
  if (/abdomen/.test(n)) return { key: 'abd', color: 0xe3a23e, opacity: 1 };
  if (/coxa|trochanter|femur|tibia|tarsus|claw|pulvill|leg/.test(n)) return { key: 'leg', color: 0xd9b06a, opacity: 1 };
  if (/thorax|head|rostrum|haustell|pedicel|arista|antenna|palp/.test(n)) return { key: 'body', color: 0x6e4b2f, opacity: 1 };
  return { key: 'other', color: 0x9a7650, opacity: 1 };
}

const mixHex = (a: number, b: number, t: number) => new THREE.Color(a).lerp(new THREE.Color(b), t).getHex();

/** One fly (player or mate): a set of body groups, each holding merged meshes of one colour key. */
class FlyRig {
  root = new THREE.Group();
  bodies = new Map<number, THREE.Group>();
  constructor(parts: Part[], tint: number | null, cache: Map<string, THREE.Material>) {
    for (const p of parts) {
      const ck = tint + '|' + p.key;
      let mat = cache.get(ck);
      if (!mat) {
        mat = new THREE.MeshStandardMaterial({
          color: tint == null ? p.color : mixHex(p.color, tint, 0.6), roughness: 0.6, metalness: 0,
          transparent: p.opacity < 1, opacity: p.opacity, depthWrite: p.opacity >= 1
        });
        cache.set(ck, mat);
      }
      const mesh = new THREE.Mesh(p.geo, mat);
      mesh.castShadow = p.opacity >= 1;
      let g = this.bodies.get(p.body);
      if (!g) { g = new THREE.Group(); this.bodies.set(p.body, g); this.root.add(g); }
      g.add(mesh);
    }
  }
}

export class World {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(36, 1, 1, 6000);
  readonly controls: OrbitControls;
  readonly canvas: HTMLCanvasElement;
  dirty = true;
  private flyRoot = new THREE.Group();
  private objRoot = new THREE.Group();
  private bodies = new Map<number, THREE.Group>();
  private views = new Map<string, View>();
  private parts: Part[] = [];
  private matCache = new Map<string, THREE.Material>();
  private spiderGeo?: THREE.BufferGeometry;
  private kitchen?: { dispose(): void };
  private odor?: OdorOverlay;
  private foodGeo = new THREE.SphereGeometry(2.4, 14, 10);
  private foodMat = new THREE.MeshStandardMaterial({ color: 0xe8473c, roughness: 0.45 });
  private scaryMat = new THREE.MeshStandardMaterial({ color: 0xd7261e, roughness: 0.5 });
  private mateMatPlaceholder = new THREE.MeshStandardMaterial({ color: MATE_TINT, roughness: 0.6 });
  private ringMat = new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.85, depthWrite: false });
  private ray = new THREE.Raycaster();
  private plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  private hit = new THREE.Vector3();
  private ndc = new THREE.Vector2();
  private thorax = new THREE.Vector3();
  private focusSet = false;
  private follow = true;
  private lastFocus = new THREE.Vector3();
  private raf = 0;
  private ro: ResizeObserver;
  private sun!: THREE.DirectionalLight;
  private shadowHalf = 330;
  private buf: Snap[] = [];
  private simRate = 1;
  private qa = new THREE.Quaternion();
  private qb = new THREE.Quaternion();

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.85;
    MAX_ANISO = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
    this.canvas = this.renderer.domElement;
    host.appendChild(this.canvas);
    this.scene.background = new THREE.Color(KIT.bg);
    this.scene.fog = new THREE.Fog(KIT.bg, 900, 3200);
    const pm = new THREE.PMREMGenerator(this.renderer);  // offline studio environment: soft reflections on eyes and legs
    this.scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.3;
    pm.dispose();
    this.scene.add(this.flyRoot, this.objRoot);
    this.buildKitchen();
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 1600;
    this.controls.addEventListener('change', () => (this.dirty = true));
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.resize();
    this.camera.near = 0.5;
    this.camera.layers.enable(1);
    this.camera.updateProjectionMatrix();
    this.placeCamera(45, true);
    const loop = () => { this.raf = requestAnimationFrame(loop); this.tick(); };
    loop();
  }

  private buildKitchen() {
    this.kitchen = buildKitchen(this.scene, { maxAniso: MAX_ANISO });
    const s = this.scene;
    // light: soft sky fill + one warm sun from the window (shadows: one 2048 map)
    s.add(new THREE.HemisphereLight(0xfff4e2, 0x9c7a5a, 0.75));
    const sun = (this.sun = new THREE.DirectionalLight(0xffe3b8, 2.0));
    sun.position.set(-160, 420, 380);
    sun.target.position.set(0, 0, 0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -330; sc.right = 330; sc.top = 330; sc.bottom = -330; sc.near = 50; sc.far = 1200;
    sc.updateProjectionMatrix();
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.05;
    sun.shadow.radius = 2;
    s.add(sun, sun.target);
    const fill = new THREE.DirectionalLight(0xcfe6ff, 0.5);
    fill.position.set(200, -300, 200);
    s.add(fill);
  }

  private placeCamera(dist: number, fly: boolean) {
    const target = fly && this.focusSet ? this.thorax.clone() : new THREE.Vector3(30, 20, 0);
    const dir = new THREE.Vector3(-0.3, -0.85, fly ? 0.38 : 0.55).normalize();  // low angle when following: the far wall and window show behind the fly
    this.controls.target.copy(target);
    this.camera.position.copy(target).addScaledVector(dir, dist);
    this.lastFocus.copy(target);
    this.controls.update();
    this.dirty = true;
  }

  /** Whole kitchen view; turns follow off. */
  frameKitchen() { this.follow = false; this.placeCamera(640, false); }
  /** Camera tracks the fly at a distance in mm (about 80 = close up, 220 = default). */
  followFly(dist: number) { this.follow = true; this.placeCamera(dist, true); }
  get following() { return this.follow; }
  /** The thorax body group (interpolated pose) and the eye centres in its frame: used by the eye view. */
  get flyBody(): THREE.Group | undefined { return this.bodies.get(1); }
  eyeLocal: { L: number[]; R: number[] } | null = null;
  stopFollow() { this.follow = false; }

  private resize() {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  private tick() {
    if (this.buf.length && this.applyInterp(performance.now())) this.dirty = true;
    if (this.follow && this.focusSet) {
      const d = this.thorax.clone().sub(this.lastFocus);
      if (d.lengthSq() > 1e-10) {
        this.camera.position.add(d);
        this.controls.target.add(d);
        this.lastFocus.copy(this.thorax);
        this.dirty = true;
      }
    }
    this.controls.update();
    if (this.dirty) {
      this.dirty = false;
      this.updateShadow();
      this.renderer.render(this.scene, this.camera);
    }
  }

  /** Build the fly from GET /model (+ /mesh/{id}). Parts are merged per body and colour. */
  async loadFly(model: any, fetchMesh: (id: number) => Promise<MeshJson>) {
    const geoms: any[] = (model.geoms ?? []).filter((g: any) => g.type !== 'plane' && g.mesh_id != null);
    const ids = [...new Set(geoms.map((g) => g.mesh_id as number))];
    const meshes = new Map<number, MeshJson>();
    await Promise.all(ids.map(async (id) => { try { meshes.set(id, await fetchMesh(id)); } catch { /* skip */ } }));
    const buckets = new Map<string, { body: number; key: string; color: number; opacity: number; geos: THREE.BufferGeometry[] }>();
    for (const g of geoms) {
      const m = meshes.get(g.mesh_id);
      if (!m || !m.vertices?.length) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(m.vertices.flat(), 3));
      geo.setIndex(m.faces.flat());
      geo.computeVertexNormals();
      const q = g.quat ?? [1, 0, 0, 0];
      geo.applyMatrix4(new THREE.Matrix4().compose(
        new THREE.Vector3(g.pos[0], g.pos[1], g.pos[2]), new THREE.Quaternion(q[1], q[2], q[3], q[0]), ONE));
      const look = partLook(String(g.name ?? ''));
      const k = g.body + '|' + look.key;
      let b = buckets.get(k);
      if (!b) { b = { body: g.body, key: look.key, color: look.color, opacity: look.opacity, geos: [] }; buckets.set(k, b); }
      b.geos.push(geo);
    }
    this.parts = [];
    for (const b of buckets.values()) {
      const merged = b.geos.length === 1 ? b.geos[0] : mergeGeometries(b.geos, false);
      if (!merged) continue;
      this.parts.push({ body: b.body, key: b.key, color: b.color, opacity: b.opacity, geo: merged });
    }
    this.flyRoot.clear();
    this.bodies.clear();
    const rig = new FlyRig(this.parts, null, this.matCache);
    this.bodies = rig.bodies;
    this.flyRoot.add(rig.root);
    // eye centres in the thorax frame (the eye meshes' centroids; left = +y), for the eye view
    const acc = { L: { s: [0, 0, 0], n: 0 }, R: { s: [0, 0, 0], n: 0 } };
    for (const g of geoms) {
      if (!/eye/i.test(String(g.name ?? ''))) continue;
      const m = meshes.get(g.mesh_id);
      if (!m?.vertices?.length) continue;
      const q = g.quat ?? [1, 0, 0, 0];
      const M = new THREE.Matrix4().compose(new THREE.Vector3(g.pos[0], g.pos[1], g.pos[2]), new THREE.Quaternion(q[1], q[2], q[3], q[0]), ONE);
      const c = new THREE.Vector3();
      for (const v of m.vertices) c.add(new THREE.Vector3(v[0], v[1], v[2]));
      c.divideScalar(m.vertices.length).applyMatrix4(M);
      const a = c.y > 0 ? acc.L : acc.R;
      a.s[0] += c.x; a.s[1] += c.y; a.s[2] += c.z; a.n++;
    }
    const avg = (a: { s: number[]; n: number }) => a.s.map((x) => x / a.n);
    this.eyeLocal = acc.L.n && acc.R.n ? { L: avg(acc.L), R: avg(acc.R) } : { L: [0.55, 0.3, 0.3], R: [0.55, -0.3, 0.3] };
    rig.root.traverse((o) => o.layers.set(1));  // layer 1 = her own body: hidden from her eyes, visible to the main camera
    for (const v of this.views.values()) if (v.kind === 'mate' && !v.rig) this.attachRig(v);
    this.dirty = true;
  }

  /** Physics pose frame: bodies:[{id,pos:[x,y,z],quat:[w,x,y,z]}]. */
  setPose(frame: { t?: number; bodies?: { id: number; pos: number[]; quat: number[] }[] }) {
    const arr = performance.now();
    const t = typeof frame.t === 'number' ? frame.t : arr / 1000;
    const last = this.buf[this.buf.length - 1];
    if (last && t < last.t - 1e-6) this.buf.length = 0;  // simulation was reset
    const m = new Map<number, { p: number[]; q: number[] }>();
    for (const b of frame.bodies ?? []) m.set(b.id, { p: b.pos, q: b.quat });
    this.buf.push({ t, arr, m });
    if (this.buf.length > 48) this.buf.shift();
    const n = this.buf.length;
    if (n >= 8) {  // sim seconds per wall second (follows the 0.1x..2x time control)
      const a = this.buf[n - 8], z = this.buf[n - 1];
      const dw = (z.arr - a.arr) / 1000;
      if (dw > 0.02) this.simRate += 0.2 * (Math.min(3, Math.max(0.05, (z.t - a.t) / dw)) - this.simRate);
    }
    if (n === 1) this.applyInterp(arr);
    this.dirty = true;
  }

  /**
   * Jitter buffer: frames are played back by their SIMULATION time (not by when they happened to arrive),
   * 50 ms late, blended between the two frames around that moment. Legs step at ~12 Hz, so this has to be smooth.
   * Returns true while frames are still flowing (keep rendering).
   */
  private applyInterp(now: number): boolean {
    const n = this.buf.length;
    if (!n) return false;
    const last = this.buf[n - 1];
    const target = last.t + ((now - last.arr) / 1000 - RENDER_DELAY_S) * this.simRate;
    let i = n - 1;
    while (i > 0 && this.buf[i].t > target) i--;
    const b0 = this.buf[i], b1 = this.buf[Math.min(i + 1, n - 1)];
    const span = b1.t - b0.t;
    const a = span > 1e-9 ? Math.min(1, Math.max(0, (target - b0.t) / span)) : 1;
    for (const [id, g] of this.bodies) {
      const c = b1.m.get(id), p = b0.m.get(id);
      if (!c || !p) continue;
      g.position.set(p.p[0] + (c.p[0] - p.p[0]) * a, p.p[1] + (c.p[1] - p.p[1]) * a, p.p[2] + (c.p[2] - p.p[2]) * a + FLOOR_LIFT_MM);
      this.qa.set(p.q[1], p.q[2], p.q[3], p.q[0]);
      this.qb.set(c.q[1], c.q[2], c.q[3], c.q[0]);
      g.quaternion.slerpQuaternions(this.qa, this.qb, a);
      if (id === 1) { this.thorax.copy(g.position); this.focusSet = true; }
    }
    this.updateMates();
    return now - last.arr < 250;
  }

  /** Shadow box follows what the camera looks at: sharp contact shadow on the fly when close, whole kitchen when far. */
  private updateShadow() {
    const half = Math.min(330, Math.max(26, this.camera.position.distanceTo(this.controls.target) * 0.9));
    const tgt = this.controls.target;
    this.sun.target.position.copy(tgt);
    this.sun.position.copy(tgt).add(new THREE.Vector3(-160, 420, 380).normalize().multiplyScalar(700));
    if (Math.abs(half - this.shadowHalf) / this.shadowHalf > 0.02) {
      const sc = this.sun.shadow.camera;
      sc.left = -half; sc.right = half; sc.top = half; sc.bottom = -half;
      sc.updateProjectionMatrix();
      this.shadowHalf = half;
    }
  }

  private updateMates() {
    const t1 = this.bodies.get(1);
    if (!this.focusSet || !t1) return;
    const tInv = new THREE.Matrix4().compose(t1.position, t1.quaternion, ONE).invert();
    const P = new THREE.Matrix4(), L = new THREE.Matrix4();
    const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    for (const v of this.views.values()) {
      if (v.kind !== 'mate') continue;
      const dx = this.thorax.x - v.pos[0], dy = this.thorax.y - v.pos[1];
      v.group.position.set(v.pos[0], v.pos[1], this.thorax.z);
      if (dx * dx + dy * dy > 1e-6) v.group.rotation.set(0, 0, Math.atan2(dy, dx));   // faces the player fly
      v.group.scale.setScalar(MATE_SCALE);
      if (!v.rig) continue;
      for (const [id, g] of v.rig.bodies) {
        const src = this.bodies.get(id);
        if (!src) continue;
        P.compose(src.position, src.quaternion, ONE);
        L.multiplyMatrices(tInv, P).decompose(p, q, s);
        g.position.copy(p); g.quaternion.copy(q); g.scale.copy(s);
      }
    }
  }

  private attachRig(v: View) {
    if (!this.parts.length) return;
    for (const c of [...v.group.children]) if (c !== v.ring) v.group.remove(c);
    v.rig = new FlyRig(this.parts, MATE_TINT, this.matCache);
    v.group.add(v.rig.root);
  }

  private makeView(o: WorldObj): View {
    const group = new THREE.Group();
    group.userData.objId = o.id;
    const ringR = o.kind === 'food' ? 6 : 13;
    const ring = new THREE.Mesh(new THREE.RingGeometry(ringR - 1.1, ringR, 40), this.ringMat);
    ring.visible = false;
    ring.layers.set(1);  // selection ring: not part of the world her eyes see
    ring.position.z = 0.2;
    ring.raycast = () => {};
    group.add(ring);
    const v: View = { kind: o.kind, group, ring, pos: o.pos, amount: 1 };
    if (o.kind === 'food') {
      const f = createFood();
      group.add(f.group);
      v.food = f;
    } else if (o.kind === 'scary') {
      const s = createSpider();  // static model (no animation)
      group.add(s.group);
      v.anim = s;
    } else if (this.parts.length) {
      this.attachRig(v);
    } else {
      const m = new THREE.Mesh(new THREE.SphereGeometry(4, 12, 8), this.mateMatPlaceholder);
      m.scale.set(1.6, 0.8, 0.8);
      m.position.z = 1;
      group.add(m);
    }
    this.objRoot.add(group);
    return v;
  }

  private removeView(id: string) {
    const v = this.views.get(id);
    if (!v) return;
    this.objRoot.remove(v.group);
    v.anim?.dispose();
    v.food?.dispose();
    this.views.delete(id);
  }

  /** Sync objects from the physics stream. skipId keeps a locally dragged object where the user put it. */
  setObjects(list: WorldObj[], skipId: string | null = null) {
    const seen = new Set<string>();
    for (const o of list) {
      seen.add(o.id);
      let v = this.views.get(o.id);
      if (v && v.kind !== o.kind) { this.removeView(o.id); v = undefined; }
      if (!v) { v = this.makeView(o); this.views.set(o.id, v); }
      if (o.id !== skipId) v.pos = o.pos;
      v.group.position.set(v.pos[0], v.pos[1], o.kind === 'mate' ? this.thorax.z : 0);
      if (o.kind === 'food') {
        v.amount = Math.max(0, Math.min(1, o.amount ?? 1));
        v.food?.setAmount(v.amount);
      }
    }
    for (const id of [...this.views.keys()]) if (!seen.has(id)) this.removeView(id);
    this.updateMates();
    this.dirty = true;
  }

  /** Immediate local move while dragging (the server copy is sent throttled). */
  moveLocal(id: string, pos: [number, number]) {
    const v = this.views.get(id);
    if (!v) return;
    v.pos = pos;
    v.group.position.set(pos[0], pos[1], v.kind === 'mate' ? this.thorax.z : 0);
    this.updateMates();
    this.dirty = true;
  }

  setSelected(id: string | null) {
    for (const [k, v] of this.views) v.ring.visible = k === id;
    this.dirty = true;
  }

  setOrbit(on: boolean) { this.controls.enabled = on; }

  private toNdc(cx: number, cy: number) {
    this.camera.updateMatrixWorld();
    const r = this.canvas.getBoundingClientRect();
    this.ndc.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(this.ndc, this.camera);
  }

  /** Screen point to the ground plane (z = 0), in physics units (mm). */
  pickGround(cx: number, cy: number): [number, number] | null {
    this.toNdc(cx, cy);
    const p = this.ray.ray.intersectPlane(this.plane, this.hit);
    return p ? [Math.round(p.x * 10) / 10, Math.round(p.y * 10) / 10] : null;
  }

  pickObject(cx: number, cy: number): string | null {
    this.toNdc(cx, cy);
    for (const h of this.ray.intersectObjects(this.objRoot.children, true)) {
      let o: THREE.Object3D | null = h.object;
      while (o && o.userData.objId === undefined) o = o.parent;
      if (o) return o.userData.objId as string;
    }
    return null;
  }

  /** Show the odour concentration grid (from physics /odor) on the counter; null hides it. */
  setOdor(g: OdorGrid | null, species: OdorSpecies) {
    this.odor ??= new OdorOverlay(this.scene);
    if (this.odor.set(g, species)) this.dirty = true;
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.odor?.dispose();
    this.ro.disconnect();
    this.kitchen?.dispose();
    this.controls.dispose();
    this.renderer.dispose();
    this.canvas.remove();
  }
}
