import * as THREE from 'three';
import type { World } from './world';

/**
 * "What the fly sees": the 3D scene is rendered from each eye's real position on the fly's head (about 120 degrees per
 * eye, her own body hidden), then sampled at ~700 ommatidia per eye on a hexagonal lattice and drawn as dots, like a
 * compound eye. It is a picture of what the eyes would receive. It is NOT fed to the brain (the brain's looming neurons
 * are still driven from object distance), and real flies resolve far less than this and see a different spectrum.
 */
const DEG = Math.PI / 180;
const FOV = 120;  // degrees per eye (square frustum, sampled inside a 58 degree radius)
const RES = 64;  // render-target pixels per side
const SPACING = 4.3;  // degrees between ommatidia (about 700 per eye inside the sampled circle)
const AXIS_YAW = 50;  // optical axis of each eye, degrees outward from straight ahead (leaves ~20 degrees of binocular overlap)
const AXIS_PITCH = 8;  // degrees up
const UPDATE_MS = 66;  // ~15 Hz

type Omm = { a: number; b: number; px: number; py: number };

function lattice(): Omm[] {
  const out: Omm[] = [];
  const dy = SPACING * 0.866, k = Math.tan((FOV / 2) * DEG);
  for (let r = -20; r <= 20; r++) {
    const b = r * dy;
    for (let c = -20; c <= 20; c++) {
      const a = c * SPACING + (r & 1 ? SPACING / 2 : 0);
      const th = Math.hypot(a, b);
      if (th > 58) continue;
      const ph = Math.atan2(b, a), t = Math.tan(th * DEG);
      out.push({ a, b, px: Math.round(((t * Math.cos(ph)) / k * 0.5 + 0.5) * (RES - 1)), py: Math.round(((t * Math.sin(ph)) / k * 0.5 + 0.5) * (RES - 1)) });
    }
  }
  return out;
}

export class EyeView {
  readonly count: number;
  private omm = lattice();
  private rt = new THREE.WebGLRenderTarget(RES, RES, { depthBuffer: true });
  private cam = new THREE.PerspectiveCamera(FOV, 1, 0.05, 4000);
  private buf = new Uint8Array(RES * RES * 4);
  private timer = 0;
  private v = new THREE.Vector3();
  private d = new THREE.Vector3();
  private u = new THREE.Vector3();
  private gone = false;

  constructor(private world: World, private cL: HTMLCanvasElement, private cR: HTMLCanvasElement) {
    this.count = this.omm.length;
    this.timer = window.setInterval(() => this.update(), UPDATE_MS);
    this.update();
  }

  dispose() {
    this.gone = true;
    clearInterval(this.timer);
    this.rt.dispose();
  }

  private update() {
    if (this.gone || document.hidden) return;
    const fly = this.world.flyBody, loc = this.world.eyeLocal;
    if (!fly || !loc) return;
    this.drawEye(this.cL, 'L', fly, loc.L);
    this.drawEye(this.cR, 'R', fly, loc.R);
  }

  private drawEye(canvas: HTMLCanvasElement, side: 'L' | 'R', fly: THREE.Object3D, local: number[]) {
    const q = fly.quaternion, s = side === 'L' ? 1 : -1;
    this.v.set(local[0], local[1], local[2]).applyQuaternion(q).add(fly.position);
    const p = AXIS_PITCH * DEG, y = AXIS_YAW * DEG;
    this.d.set(Math.cos(p) * Math.cos(y), s * Math.cos(p) * Math.sin(y), Math.sin(p)).applyQuaternion(q);
    this.u.set(0, 0, 1).applyQuaternion(q);
    this.cam.position.copy(this.v);
    this.cam.up.copy(this.u);
    this.cam.lookAt(this.v.x + this.d.x, this.v.y + this.d.y, this.v.z + this.d.z);
    this.cam.updateMatrixWorld(true);
    const r = this.world.renderer, prevRT = r.getRenderTarget(), prevAuto = r.shadowMap.autoUpdate;
    r.shadowMap.autoUpdate = false;  // reuse the last shadow map; the eye pass must not redo it
    r.setRenderTarget(this.rt);
    r.render(this.world.scene, this.cam);
    r.readRenderTargetPixels(this.rt, 0, 0, RES, RES, this.buf);
    r.setRenderTarget(prevRT);
    r.shadowMap.autoUpdate = prevAuto;
    this.paint(canvas);
  }

  private color(px: number, py: number): [number, number, number] {
    let R = 0, G = 0, B = 0, n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {  // 3x3 average: each ommatidium collects light over an area
      const x = px + dx, y = py + dy;
      if (x < 0 || y < 0 || x >= RES || y >= RES) continue;
      const i = (y * RES + x) * 4;
      R += this.buf[i]; G += this.buf[i + 1]; B += this.buf[i + 2]; n++;
    }
    R /= n * 255; G /= n * 255; B /= n * 255;  // linear light
    const lum = 0.3 * R + 0.59 * G + 0.11 * B, mix = 0.65;
    const f = (c: number) => Math.round(255 * Math.pow(Math.min(1, (lum + (c - lum) * mix) * 0.85), 1 / 2.2));
    return [f(R), f(G), f(B)];
  }

  private paint(canvas: HTMLCanvasElement) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2), w = canvas.clientWidth || 150, h = canvas.clientHeight || 150;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const g = canvas.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#000';
    g.fillRect(0, 0, w, h);
    const S = (Math.min(w, h) / 2 - 3) / 60, cx = w / 2, cy = h / 2, rad = SPACING * S * 0.5;
    for (const o of this.omm) {
      const [r, gg, b] = this.color(o.px, o.py);
      g.fillStyle = 'rgb(' + r + ',' + gg + ',' + b + ')';
      g.beginPath();
      g.arc(cx + o.a * S, cy - o.b * S, rad, 0, 6.2832);
      g.fill();
    }
    g.strokeStyle = 'rgba(255,255,255,0.35)';
    g.lineWidth = 1;
    g.beginPath();
    g.arc(cx, cy, 59 * S, 0, 6.2832);
    g.stroke();
  }
}
