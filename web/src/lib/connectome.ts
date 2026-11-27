import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Neuron positions (brain index order) and named groups. Decode per neuron_positions.json.
export type GroupIndex = { n: number; groups: Record<string, { role: string; default_rate_hz: number; size: number; idx: number[] }> };
let posP: Promise<Float32Array> | undefined;
let grpP: Promise<GroupIndex> | undefined;

export function loadNeuronPositions(): Promise<Float32Array> {
  posP ??= (async () => {
    const [meta, bin] = await Promise.all([
      fetch('/neuron_positions.json').then((r) => r.json()),
      fetch('/neuron_positions.bin').then((r) => r.arrayBuffer())
    ]);
    const q = new Int16Array(bin), n = meta.n as number, sc = meta.scale as number, off = meta.offset as number[];
    const pos = new Float32Array(n * 3), cen = [0, 0, 0];
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
      const w = (q[i * 3 + k] / 32767) * sc + off[k];   // normalised * scale + offset
      pos[i * 3 + k] = w; cen[k] += w;
    }
    for (let k = 0; k < 3; k++) cen[k] /= n;
    let ext = 1e-9;
    for (let i = 0; i < n * 3; i++) { const k = i % 3; pos[i] -= cen[k]; ext = Math.max(ext, Math.abs(pos[i])); }
    for (let i = 0; i < n * 3; i++) pos[i] /= ext;     // fit to [-1, 1]
    return pos;
  })();
  return posP;
}

export function loadGroupIndex(): Promise<GroupIndex> {
  grpP ??= fetch('/group_indices.json').then((r) => r.json());
  return grpP;
}

const PAL = [
  [0, 0, 0], [0.35, 0.7, 1], [1, 0.6, 0.3], [0.55, 1, 0.5], [1, 0.4, 0.7],
  [1, 0.9, 0.4], [0.6, 0.9, 1], [0.9, 0.6, 1], [1, 1, 1]
].map((c) => new THREE.Vector3(c[0], c[1], c[2]));

const VERT = [
  'attribute float aLast;',
  'attribute float aGroup;',
  'uniform float uNow;',
  'uniform float uTau;',
  'uniform float uPx;',
  'uniform vec3 uPal[9];',
  'varying vec3 vCol;',
  'varying float vA;',
  'void main() {',
  '  float age = max(uNow - aLast, 0.0);',
  '  float b = aLast > 0.0 ? exp(-age / uTau) : 0.0;',
  '  int g = int(aGroup + 0.5);',
  '  vec3 base = g > 0 ? uPal[g] * 0.6 : vec3(0.14, 0.2, 0.28);',
  '  vec3 flash = g > 0 ? uPal[g] : vec3(1.0, 0.85, 0.55);',
  '  vCol = base + flash * b * 1.6;',
  '  vA = 0.3 + 0.7 * b;',
  '  vec4 mv = modelViewMatrix * vec4(position, 1.0);',
  '  gl_PointSize = uPx * (0.8 + 3.4 * b);',
  '  gl_Position = projectionMatrix * mv;',
  '}'
].join('\n');
const FRAG = [
  'varying vec3 vCol;',
  'varying float vA;',
  'void main() {',
  '  float d = length(gl_PointCoord - 0.5);',
  '  if (d > 0.5) discard;',
  '  gl_FragColor = vec4(vCol, (1.0 - 2.0 * d) * vA);',
  '}'
].join('\n');

/** All neurons as additive points. A spike sets aLast; the shader decays the flash (no per-frame CPU work). */
export class Connectome {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
  private controls: OrbitControls;
  private uniforms = { uNow: { value: 0 }, uTau: { value: 0.6 }, uPx: { value: 2 }, uPal: { value: PAL } };
  private geo?: THREE.BufferGeometry;
  private lastAttr?: THREE.BufferAttribute;
  private groupAttr?: THREE.BufferAttribute;
  private lastPulse = 0;
  private uploadLast = false;
  private dirty = true;
  private raf = 0;
  private ro: ResizeObserver;
  private gone = false;
  private groupIdx: GroupIndex | null = null;
  private names: string[] = [];

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setClearColor(0x000000, 1);
    host.appendChild(this.renderer.domElement);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0, -3.1, 1.1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.6;
    this.controls.addEventListener('change', () => (this.dirty = true));
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.resize();
    loadGroupIndex().then((g) => { this.groupIdx = g; this.applyGroups(); }).catch(() => {});
    loadNeuronPositions().then((p) => this.attach(p));
    const loop = () => {
      if (this.gone) return;
      this.raf = requestAnimationFrame(loop);
      this.tick();
    };
    loop();
  }

  private attach(pos: Float32Array) {
    if (this.gone) return;
    const n = pos.length / 3;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.lastAttr = new THREE.BufferAttribute(new Float32Array(n).fill(-1e6), 1);
    this.groupAttr = new THREE.BufferAttribute(new Float32Array(n), 1);
    geo.setAttribute('aLast', this.lastAttr);
    geo.setAttribute('aGroup', this.groupAttr);
    this.geo = geo;
    const mat = new THREE.ShaderMaterial({
      uniforms: this.uniforms, vertexShader: VERT, fragmentShader: FRAG,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
    });
    this.scene.add(new THREE.Points(geo, mat));
    this.dirty = true;
  }

  /** Spike indices (brain tensor index) from the brain stream. */
  pulse(indices: number[]) {
    if (!this.lastAttr || !indices.length) return;
    const t = performance.now() / 1000, a = this.lastAttr.array as Float32Array;
    for (const i of indices) if (i >= 0 && i < a.length) a[i] = t;
    this.lastPulse = t;
    this.uploadLast = true;
  }

  /** Colour the named groups (chips). Groups not listed are dim. */
  setGroups(names: string[]) {
    this.names = names;
    this.applyGroups();
  }

  private applyGroups() {
    if (!this.groupAttr || !this.groupIdx) return;
    const names = this.names;
    const g = this.groupAttr.array as Float32Array;
    g.fill(0);
    names.forEach((name, gi) => {
      for (const i of this.groupIdx!.groups[name]?.idx ?? []) if (i < g.length) g[i] = gi + 1;
    });
    this.groupAttr.needsUpdate = true;
    this.dirty = true;
  }

  private tick() {
    const now = performance.now() / 1000;
    this.uniforms.uNow.value = now;
    this.controls.update();
    const flashing = now - this.lastPulse < 1.5;
    if (this.uploadLast && this.lastAttr) { this.lastAttr.needsUpdate = true; this.uploadLast = false; }
    if (this.dirty || flashing || this.controls.autoRotate) {
      this.dirty = false;
      this.renderer.render(this.scene, this.camera);
    }
  }

  private resize() {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  dispose() {
    this.gone = true;
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.controls.dispose();
    this.geo?.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
