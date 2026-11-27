import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

/** The props drawn here have logical footprints (collision + obstacle sensing) in physics/app/data/kitchen_props.json,
 *  served at run time by GET PROPS_URL. Move or resize a prop here and update that file too
 *  (physics/tools/check_kitchen_props.py fails until both match). */
export const PROPS_URL = '/api/physics/props';

/**
 * Procedural cosy kitchen, Z-up, millimetres. The counter top is the plane z = 0
 * (600 x 400, x -300..300, y -200..200); the back wall is y = +230 and the window
 * is near x = -40, z = 170. Everything is generated here: no assets, no network.
 *
 * Geometry is merged per material (a handful of draw calls) and colour variation
 * inside a material comes from vertex colours, so a whole family of props is one
 * mesh. Repeated books are an InstancedMesh.
 */

type Batch = { mat: THREE.Material; geos: THREE.BufferGeometry[]; cast: boolean; receive: boolean };

export function buildKitchen(scene: THREE.Scene, opts: { maxAniso: number }): { dispose(): void } {
  const aniso = Math.max(1, Math.min(8, opts.maxAniso || 1));
  const root = new THREE.Group();
  root.name = 'kitchen';
  scene.add(root);

  const textures: THREE.Texture[] = [];
  const materials: THREE.Material[] = [];
  const geometries: THREE.BufferGeometry[] = [];
  const batches = new Map<THREE.Material, Batch>();

  // ---------------------------------------------------------------- textures
  const canvas = (w: number, h: number, draw: (g: CanvasRenderingContext2D, w: number, h: number) => void): THREE.CanvasTexture => {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    draw(c.getContext('2d')!, w, h);
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = aniso;
    t.colorSpace = THREE.SRGBColorSpace;
    textures.push(t);
    return t;
  };
  const rnd = (() => { let s = 0x9e3779b9; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

  // Warm oak planks: used only on the counter, mapped in world millimetres.
  const woodTopTex = canvas(2048, 2048, (g, W, H) => {
    g.fillStyle = '#c99a63'; g.fillRect(0, 0, W, H);
    const planks = 6, ph = H / planks;
    for (let p = 0; p < planks; p++) {
      const y0 = p * ph, tone = 0.9 + rnd() * 0.16;
      g.fillStyle = `rgb(${lerp(150, 205, tone) | 0},${lerp(105, 150, tone) | 0},${lerp(58, 92, tone) | 0})`;
      g.fillRect(0, y0, W, ph);
      // grain: long, slightly wavy lines running with the board
      for (let i = 0; i < 130; i++) {
        const yy = y0 + rnd() * ph, amp = 1 + rnd() * 5;
        g.strokeStyle = `rgba(${60 + rnd() * 40 | 0},${32 + rnd() * 22 | 0},${12 + rnd() * 14 | 0},${(0.035 + rnd() * 0.1).toFixed(3)})`;
        g.lineWidth = 0.5 + rnd() * 2.6;
        g.beginPath(); g.moveTo(0, yy);
        g.bezierCurveTo(W * 0.3, yy + amp, W * 0.65, yy - amp, W, yy + (rnd() - 0.5) * 3);
        g.stroke();
      }
      // pores and tiny knots
      for (let i = 0; i < 1500; i++) {
        g.fillStyle = `rgba(70,40,16,${(rnd() * 0.1).toFixed(3)})`;
        g.fillRect(rnd() * W, y0 + rnd() * ph, 1 + rnd() * 3, 1);
      }
      for (let k = 0; k < 3; k++) {
        const kx = rnd() * W, ky = y0 + 0.2 * ph + rnd() * 0.6 * ph;
        for (let r = 7; r > 0; r--) {
          g.strokeStyle = `rgba(80,44,18,${(0.06 + rnd() * 0.06).toFixed(3)})`;
          g.lineWidth = 1.2;
          g.beginPath(); g.ellipse(kx, ky, r * 2.2, r * 0.8, 0, 0, Math.PI * 2); g.stroke();
        }
      }
      // board seam + a couple of staggered end joints
      g.fillStyle = 'rgba(52,30,12,0.72)'; g.fillRect(0, y0, W, 3.2);
      g.fillStyle = 'rgba(255,225,185,0.18)'; g.fillRect(0, y0 + 3.2, W, 1.6);
      for (let j = 0; j < 2; j++) {
        const jx = (rnd() * 0.7 + 0.15) * W;
        g.fillStyle = 'rgba(52,30,12,0.5)'; g.fillRect(jx, y0 + 4, 2.5, ph - 4);
      }
    }
    g.fillStyle = 'rgba(30,18,8,0.5)'; g.fillRect(0, H - 2, W, 2);
  });
  woodTopTex.repeat.set(1, 1);

  // Neutral grain for shelves, boards and knife block; tinted by vertex colour.
  const woodTex = canvas(1024, 1024, (g, W, H) => {
    g.fillStyle = '#d9c6a6'; g.fillRect(0, 0, W, H);
    for (let i = 0; i < 210; i++) {
      const y = rnd() * H, amp = 1 + rnd() * 6;
      g.strokeStyle = `rgba(${110 + rnd() * 40 | 0},${78 + rnd() * 26 | 0},${44 + rnd() * 18 | 0},${(0.04 + rnd() * 0.1).toFixed(3)})`;
      g.lineWidth = 0.5 + rnd() * 2.4;
      g.beginPath(); g.moveTo(0, y);
      g.bezierCurveTo(W * 0.3, y + amp, W * 0.7, y - amp, W, y + (rnd() - 0.5) * 4);
      g.stroke();
    }
    for (let i = 0; i < 900; i++) {
      g.fillStyle = `rgba(90,60,30,${(rnd() * 0.09).toFixed(3)})`;
      g.fillRect(rnd() * W, rnd() * H, 1 + rnd() * 2, 1);
    }
  });

  // Soft painted finish (near white, tinted by vertex colour).
  const paintTex = canvas(512, 512, (g, W, H) => {
    g.fillStyle = '#f6f1e7'; g.fillRect(0, 0, W, H);
    for (let i = 0; i < 900; i++) {
      g.fillStyle = `rgba(${190 + rnd() * 50 | 0},${180 + rnd() * 50 | 0},${160 + rnd() * 50 | 0},${(rnd() * 0.09).toFixed(3)})`;
      g.fillRect(rnd() * W, rnd() * H, 1 + rnd() * 5, 1 + rnd() * 2);
    }
    for (let i = 0; i < 26; i++) {
      g.strokeStyle = `rgba(255,255,255,${(0.04 + rnd() * 0.08).toFixed(3)})`;
      g.lineWidth = 2 + rnd() * 8;
      g.beginPath(); const x = rnd() * W; g.moveTo(x, 0); g.lineTo(x + 20, H); g.stroke();
    }
  });

  // Plaster for the walls and ceiling.
  const plasterTex = canvas(512, 512, (g, W, H) => {
    g.fillStyle = '#f0e6d5'; g.fillRect(0, 0, W, H);
    for (let i = 0; i < 2600; i++) {
      const v = 205 + rnd() * 40 | 0;
      g.fillStyle = `rgba(${v},${v - 8},${v - 22},${(rnd() * 0.1).toFixed(3)})`;
      g.fillRect(rnd() * W, rnd() * H, 1 + rnd() * 4, 1 + rnd() * 4);
    }
  });

  // Hand-made sage tiles with cream grout.
  const tileTex = canvas(512, 512, (g, W, H) => {
    g.fillStyle = '#e7ddcb'; g.fillRect(0, 0, W, H);
    const n = 6, s = W / n;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const t = rnd();
      const base = t < 0.12 ? [206, 156, 118] : [136 + rnd() * 18, 166 + rnd() * 16, 146 + rnd() * 16];
      g.fillStyle = `rgb(${base[0] | 0},${base[1] | 0},${base[2] | 0})`;
      g.fillRect(x * s + 3, y * s + 3, s - 6, s - 6);
      const grd = g.createLinearGradient(x * s, y * s, x * s + s, y * s + s);
      grd.addColorStop(0, 'rgba(255,255,255,0.26)'); grd.addColorStop(0.45, 'rgba(255,255,255,0)');
      grd.addColorStop(1, 'rgba(20,40,30,0.16)');
      g.fillStyle = grd; g.fillRect(x * s + 3, y * s + 3, s - 6, s - 6);
      for (let i = 0; i < 40; i++) {
        g.fillStyle = `rgba(255,255,255,${(rnd() * 0.09).toFixed(3)})`;
        g.fillRect(x * s + 3 + rnd() * (s - 8), y * s + 3 + rnd() * (s - 8), 1 + rnd() * 3, 1);
      }
    }
  });

  // Terracotta / cream checker floor.
  const floorTex = canvas(512, 512, (g, W, H) => {
    g.fillStyle = '#cbb79d'; g.fillRect(0, 0, W, H);
    const n = 4, s = W / n;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const alt = (x + y) % 2 === 0;
      const b = alt ? [180 + rnd() * 18, 100 + rnd() * 14, 70 + rnd() * 12] : [225 + rnd() * 14, 208 + rnd() * 14, 182 + rnd() * 14];
      g.fillStyle = `rgb(${b[0] | 0},${b[1] | 0},${b[2] | 0})`;
      g.fillRect(x * s + 4, y * s + 4, s - 8, s - 8);
      for (let i = 0; i < 70; i++) {
        g.fillStyle = `rgba(${alt ? '80,40,25' : '120,100,70'},${(rnd() * 0.1).toFixed(3)})`;
        g.fillRect(x * s + 5 + rnd() * (s - 10), y * s + 5 + rnd() * (s - 10), 1 + rnd() * 5, 1 + rnd() * 3);
      }
      const grd = g.createLinearGradient(x * s, y * s, x * s + s, y * s + s);
      grd.addColorStop(0, 'rgba(255,255,255,0.13)'); grd.addColorStop(1, 'rgba(40,25,15,0.12)');
      g.fillStyle = grd; g.fillRect(x * s + 4, y * s + 4, s - 8, s - 8);
    }
  });

  // Brushed metal (light grey so vertex colours can make steel, brass and iron).
  const metalTex = canvas(512, 512, (g, W, H) => {
    g.fillStyle = '#d8d8d6'; g.fillRect(0, 0, W, H);
    for (let i = 0; i < 2400; i++) {
      const v = 160 + rnd() * 90 | 0;
      g.fillStyle = `rgba(${v},${v},${v},${(0.05 + rnd() * 0.16).toFixed(3)})`;
      g.fillRect(0, rnd() * H, W, 0.5 + rnd() * 1.6);
    }
    for (let i = 0; i < 40; i++) {
      g.strokeStyle = `rgba(255,255,255,${(rnd() * 0.2).toFixed(3)})`;
      g.lineWidth = 0.6; g.beginPath(); const y = rnd() * H; g.moveTo(0, y); g.lineTo(W, y + (rnd() - 0.5) * 3); g.stroke();
    }
  });

  // Woven fabric (tinted cream / coral by vertex colour).
  const fabricTex = canvas(512, 512, (g, W, H) => {
    g.fillStyle = '#f2e9d9'; g.fillRect(0, 0, W, H);
    for (let y = 0; y < H; y += 6) {
      g.fillStyle = `rgba(150,130,105,${(0.06 + rnd() * 0.06).toFixed(3)})`; g.fillRect(0, y, W, 2);
      g.fillStyle = 'rgba(255,255,255,0.16)'; g.fillRect(0, y + 2, W, 1);
    }
    for (let x = 0; x < W; x += 6) {
      g.fillStyle = `rgba(140,120,95,${(0.05 + rnd() * 0.05).toFixed(3)})`; g.fillRect(x, 0, 2, H);
    }
  });

  // Wicker basket weave.
  const wickerTex = canvas(512, 512, (g, W, H) => {
    g.fillStyle = '#a97742'; g.fillRect(0, 0, W, H);
    for (let y = 0; y < H; y += 32) {
      for (let x = 0; x < W; x += 64) {
        const off = ((y / 32) % 2) * 32;
        g.fillStyle = `rgb(${170 + rnd() * 30 | 0},${120 + rnd() * 26 | 0},${62 + rnd() * 20 | 0})`;
        g.beginPath(); g.roundRect(x + off + 2, y + 2, 60, 28, 12); g.fill();
        g.strokeStyle = 'rgba(70,40,15,0.45)'; g.lineWidth = 2; g.stroke();
        g.fillStyle = 'rgba(255,225,170,0.18)'; g.fillRect(x + off + 8, y + 5, 44, 5);
      }
    }
    for (let i = 0; i < 500; i++) { g.fillStyle = `rgba(70,40,15,${rnd() * 0.15})`; g.fillRect(rnd() * W, rnd() * H, 1 + rnd() * 3, 1); }
  });

  // Rustic bread crust.
  const crustTex = canvas(512, 512, (g, W, H) => {
    g.fillStyle = '#b5753d'; g.fillRect(0, 0, W, H);
    for (let i = 0; i < 1800; i++) {
      const v = rnd();
      g.fillStyle = v < 0.5 ? `rgba(110,60,24,${(rnd() * 0.25).toFixed(3)})` : `rgba(245,215,160,${(rnd() * 0.16).toFixed(3)})`;
      g.beginPath(); g.arc(rnd() * W, rnd() * H, 0.6 + rnd() * 2.4, 0, Math.PI * 2); g.fill();
    }
    for (let i = 0; i < 26; i++) {
      g.strokeStyle = `rgba(90,48,18,${(0.1 + rnd() * 0.18).toFixed(3)})`; g.lineWidth = 2 + rnd() * 4;
      g.beginPath(); const x = rnd() * W, y = rnd() * H; g.moveTo(x, y); g.lineTo(x + (rnd() - 0.5) * 60, y + 8 + rnd() * 20); g.stroke();
    }
  });

  // Window view: soft sky, sun, garden.
  const skyTex = canvas(1024, 640, (g, W, H) => {
    const sky = g.createLinearGradient(0, 0, 0, H);
    sky.addColorStop(0, '#8fc4e8'); sky.addColorStop(0.5, '#c7e4f4'); sky.addColorStop(0.78, '#f6e7c8'); sky.addColorStop(1, '#e8d2a6');
    g.fillStyle = sky; g.fillRect(0, 0, W, H);
    const sun = g.createRadialGradient(W * 0.72, H * 0.24, 8, W * 0.72, H * 0.24, 240);
    sun.addColorStop(0, 'rgba(255,250,220,0.95)'); sun.addColorStop(0.18, 'rgba(255,236,180,0.55)'); sun.addColorStop(1, 'rgba(255,236,180,0)');
    g.fillStyle = sun; g.fillRect(0, 0, W, H);
    for (let i = 0; i < 22; i++) {
      const cx = rnd() * W, cy = H * (0.12 + rnd() * 0.38), r = 24 + rnd() * 70;
      const cg = g.createRadialGradient(cx, cy, 2, cx, cy, r);
      cg.addColorStop(0, `rgba(255,255,255,${(0.24 + rnd() * 0.3).toFixed(2)})`); cg.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = cg; g.beginPath(); g.ellipse(cx, cy, r * 1.8, r * 0.55, 0, 0, Math.PI * 2); g.fill();
    }
    // distant hedge, trees, a fence
    g.fillStyle = '#9cb67a'; g.beginPath(); g.moveTo(0, H * 0.79);
    for (let x = 0; x <= W; x += 40) g.lineTo(x, H * 0.79 + Math.sin(x * 0.011) * 16 - rnd() * 10);
    g.lineTo(W, H); g.lineTo(0, H); g.fill();
    g.fillStyle = '#7d9b61';
    for (let i = 0; i < 9; i++) {
      const x = 40 + i * 118 + rnd() * 30, y = H * 0.78, r = 30 + rnd() * 34;
      g.beginPath(); g.arc(x, y - r, r, 0, Math.PI * 2); g.fill();
      g.fillStyle = 'rgba(90,70,45,0.5)'; g.fillRect(x - 3, y - r, 6, r); g.fillStyle = '#7d9b61';
    }
    g.strokeStyle = 'rgba(120,95,60,0.55)'; g.lineWidth = 5;
    for (let x = 0; x < W; x += 46) { g.beginPath(); g.moveTo(x, H * 0.83); g.lineTo(x, H * 0.92); g.stroke(); }
    g.beginPath(); g.moveTo(0, H * 0.86); g.lineTo(W, H * 0.86); g.stroke();
    // glass reflection streaks
    g.fillStyle = 'rgba(255,255,255,0.1)';
    g.beginPath(); g.moveTo(W * 0.05, 0); g.lineTo(W * 0.2, 0); g.lineTo(W * 0.02, H); g.lineTo(-W * 0.1, H); g.fill();
    g.beginPath(); g.moveTo(W * 0.85, 0); g.lineTo(W * 0.92, 0); g.lineTo(W * 0.75, H); g.lineTo(W * 0.68, H); g.fill();
  });

  // Printed paper atlas: jam label, cereal front, clock face, botanical print.
  const printTex = canvas(1024, 1024, (g) => {
    g.fillStyle = '#efe2c8'; g.fillRect(0, 0, 1024, 1024);
    // -- jam label (top left)
    g.fillStyle = '#f7edda'; g.fillRect(16, 16, 300, 300);
    g.strokeStyle = '#b23a2e'; g.lineWidth = 7; g.strokeRect(26, 26, 280, 280);
    g.strokeStyle = 'rgba(178,58,46,0.5)'; g.lineWidth = 2; g.strokeRect(36, 36, 260, 260);
    g.fillStyle = '#a52f26'; g.font = 'bold 30px Georgia, serif'; g.textAlign = 'center';
    g.fillText('STRAWBERRY', 166, 82);
    g.font = 'bold 58px Georgia, serif'; g.fillText('JAM', 166, 142);
    g.fillStyle = '#c8452f'; g.beginPath();
    g.moveTo(166, 170); g.bezierCurveTo(128, 192, 134, 236, 166, 258); g.bezierCurveTo(198, 236, 204, 192, 166, 170); g.fill();
    g.fillStyle = '#5f8a4a'; g.beginPath(); g.ellipse(166, 170, 34, 12, 0, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#f7edda'; g.beginPath(); g.arc(154, 208, 3, 0, 7); g.arc(176, 222, 3, 0, 7); g.arc(180, 196, 2, 0, 7); g.fill();
    g.fillStyle = '#7b5a3a'; g.font = 'italic 18px Georgia, serif'; g.fillText('homemade · 2026', 166, 286);
    // -- cereal front (top right)
    g.fillStyle = '#e9a93a'; g.fillRect(340, 16, 320, 420);
    g.fillStyle = '#f6cf62'; g.beginPath(); g.arc(500, 120, 78, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#e9a93a'; g.beginPath(); g.arc(500, 120, 58, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#7b3f20'; g.font = 'bold 42px Georgia, serif'; g.textAlign = 'center';
    g.fillText('OAT', 500, 224); g.font = 'bold 54px Georgia, serif'; g.fillText('CRUNCH', 500, 278);
    g.fillStyle = '#5c3a1c'; g.font = 'bold 15px sans-serif'; g.fillText('WHOLEGRAIN · 100% OATS', 500, 306);
    g.fillStyle = '#f4ead2'; g.beginPath(); g.ellipse(500, 368, 112, 34, 0, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#c98a3c'; g.beginPath(); g.ellipse(500, 360, 92, 24, 0, 0, Math.PI * 2); g.fill();
    for (let i = 0; i < 26; i++) { g.fillStyle = i % 2 ? '#e0b06a' : '#b9782f'; g.beginPath(); g.arc(420 + rnd() * 160, 348 + rnd() * 22, 4 + rnd() * 5, 0, 7); g.fill(); }
    g.fillStyle = '#f6cf62'; g.fillRect(340, 16, 320, 10); g.fillRect(340, 426, 320, 10);
    // -- clock face (bottom left)
    g.fillStyle = '#f6edd8'; g.fillRect(16, 520, 300, 300);
    g.strokeStyle = '#b08d4f'; g.lineWidth = 6; g.beginPath(); g.arc(166, 670, 132, 0, Math.PI * 2); g.stroke();
    g.fillStyle = '#6b4a2b'; g.font = 'bold 26px Georgia, serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    const nums: [string, number][] = [['12', -90], ['3', 0], ['6', 90], ['9', 180]];
    for (const [n, a] of nums) {
      const r = 100, rad = (a * Math.PI) / 180;
      g.fillText(n, 166 + Math.cos(rad) * r, 670 + Math.sin(rad) * r);
    }
    for (let i = 0; i < 60; i++) {
      const a = (i / 60) * Math.PI * 2, r0 = i % 5 === 0 ? 112 : 120, r1 = 126;
      g.strokeStyle = i % 5 === 0 ? '#6b4a2b' : 'rgba(107,74,43,0.55)'; g.lineWidth = i % 5 === 0 ? 3 : 1.5;
      g.beginPath(); g.moveTo(166 + Math.cos(a) * r0, 670 + Math.sin(a) * r0); g.lineTo(166 + Math.cos(a) * r1, 670 + Math.sin(a) * r1); g.stroke();
    }
    g.fillStyle = '#8b6a43'; g.font = 'italic 13px Georgia, serif'; g.fillText('the kitchen', 166, 750);
    // -- botanical print (bottom right)
    g.fillStyle = '#f3ead8'; g.fillRect(380, 520, 300, 380);
    g.strokeStyle = '#8d7351'; g.lineWidth = 3; g.strokeRect(390, 530, 280, 360);
    g.strokeStyle = '#6f8c5a'; g.lineWidth = 4; g.beginPath(); g.moveTo(530, 850); g.bezierCurveTo(520, 760, 540, 690, 520, 600); g.stroke();
    for (let i = 0; i < 7; i++) {
      const y = 620 + i * 30, s = 1 - i * 0.06;
      g.fillStyle = i % 2 ? '#7f9c64' : '#6f8c5a';
      g.beginPath(); g.ellipse(530 - 28 * s, y, 30 * s, 11 * s, -0.5, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.ellipse(530 + 28 * s, y + 12, 30 * s, 11 * s, 0.5, 0, Math.PI * 2); g.fill();
    }
    g.fillStyle = '#6b5238'; g.font = 'italic 20px Georgia, serif'; g.textAlign = 'center'; g.fillText('from the garden', 530, 885);
  });

  // Rug: warm woven stripes.
  const rugTex = canvas(512, 256, (g, W, H) => {
    g.fillStyle = '#b6543f'; g.fillRect(0, 0, W, H);
    for (let y = 0; y < H; y += 24) {
      g.fillStyle = y % 48 === 0 ? '#e8d3b0' : '#c96b50'; g.fillRect(0, y, W, 14);
      g.fillStyle = 'rgba(255,255,255,0.12)'; g.fillRect(0, y + 2, W, 2);
      g.fillStyle = 'rgba(60,25,15,0.16)'; g.fillRect(0, y + 12, W, 2);
    }
    for (let x = 0; x < W; x += 8) { g.fillStyle = `rgba(60,25,15,${(0.04 + rnd() * 0.06).toFixed(3)})`; g.fillRect(x, 0, 2, H); }
    g.strokeStyle = '#7d3828'; g.lineWidth = 14; g.strokeRect(7, 7, W - 14, H - 14);
    for (let i = 0; i < 500; i++) { g.fillStyle = `rgba(255,240,210,${rnd() * 0.12})`; g.fillRect(rnd() * W, rnd() * H, 1 + rnd() * 2, 1); }
  });

  // ---------------------------------------------------------------- materials
  const std = (o: THREE.MeshStandardMaterialParameters) => {
    const m = new THREE.MeshStandardMaterial({ vertexColors: true, ...o });
    materials.push(m);
    return m;
  };
  const woodTop = new THREE.MeshStandardMaterial({ map: woodTopTex, color: 0xffffff, roughness: 0.68, metalness: 0, envMapIntensity: 0.4 });
  const wood = std({ map: woodTex, roughness: 0.66, metalness: 0 });
  const paint = std({ map: paintTex, roughness: 0.72, metalness: 0 });
  const plaster = std({ map: plasterTex, roughness: 0.94, metalness: 0 });
  const glaze = std({ roughness: 0.3, metalness: 0, envMapIntensity: 0.9 });
  const matte = std({ roughness: 0.86, metalness: 0 });
  const metal = std({ map: metalTex, roughness: 0.36, metalness: 0.62, envMapIntensity: 1.0 });
  const glass = new THREE.MeshStandardMaterial({
    color: 0xd9eee6, roughness: 0.06, metalness: 0, transparent: true, opacity: 0.3,
    depthWrite: false, side: THREE.DoubleSide, envMapIntensity: 1.4
  });
  const fabric = std({ map: fabricTex, roughness: 0.95, metalness: 0, side: THREE.DoubleSide });
  const wicker = new THREE.MeshStandardMaterial({ map: wickerTex, roughness: 0.9, metalness: 0 });
  const crust = new THREE.MeshStandardMaterial({ map: crustTex, roughness: 0.82, metalness: 0 });
  const floorMat = new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.72, metalness: 0 });
  const tileMat = new THREE.MeshStandardMaterial({ map: tileTex, roughness: 0.4, metalness: 0, envMapIntensity: 0.8 });
  const print = new THREE.MeshStandardMaterial({ map: printTex, roughness: 0.75, metalness: 0 });
  const sky = new THREE.MeshBasicMaterial({ map: skyTex, fog: false });
  const shellMat = new THREE.MeshBasicMaterial({ color: 0x6f6254, side: THREE.BackSide, fog: true });
  const rugMat = new THREE.MeshStandardMaterial({ map: rugTex, roughness: 0.98, metalness: 0 });
  const emissive = new THREE.MeshStandardMaterial({ color: 0xffe3ae, emissive: 0xffb860, emissiveIntensity: 2.4, roughness: 0.4 });
  materials.push(woodTop, glass, wicker, crust, floorMat, tileMat, print, sky, shellMat, rugMat, emissive);

  // ---------------------------------------------------------------- batching
  const batch = (mat: THREE.Material, cast = true, receive = true): Batch => {
    let b = batches.get(mat);
    if (!b) { b = { mat, geos: [], cast, receive }; batches.set(mat, b); }
    return b;
  };
  const tint = (g: THREE.BufferGeometry, color: THREE.ColorRepresentation) => {
    const c = new THREE.Color(color);
    const n = g.attributes.position.count;
    const arr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b; }
    g.setAttribute('color', new THREE.BufferAttribute(arr, 3));
    return g;
  };
  const add = (mat: THREE.Material, g: THREE.BufferGeometry, color?: THREE.ColorRepresentation, cast = true, receive = true) => {
    if (color !== undefined) tint(g, color);
    const ng = g.index ? g.toNonIndexed() : g;
    if (ng !== g) g.dispose();
    geometries.push(ng);
    batch(mat, cast, receive).geos.push(ng);
  };

  const box = (w: number, d: number, h: number, mat: THREE.Material, color: THREE.ColorRepresentation,
    x: number, y: number, z: number, rz = 0, ry = 0, rx = 0, cast = true, receive = true) => {
    const g = new THREE.BoxGeometry(w, d, h);
    if (rx) g.rotateX(rx);
    if (ry) g.rotateY(ry);
    if (rz) g.rotateZ(rz);
    g.translate(x, y, z);
    add(mat, g, color, cast, receive);
  };
  const cyl = (rt: number, rb: number, h: number, seg: number, mat: THREE.Material, color: THREE.ColorRepresentation,
    x: number, y: number, z: number, axis: 'z' | 'x' | 'y' = 'z') => {
    const g = new THREE.CylinderGeometry(rt, rb, h, seg);
    if (axis === 'z') g.rotateX(Math.PI / 2);
    else if (axis === 'x') g.rotateZ(Math.PI / 2);
    g.translate(x, y, z);
    add(mat, g, color);
  };
  const sph = (r: number, mat: THREE.Material, color: THREE.ColorRepresentation,
    x: number, y: number, z: number, sx = 1, sy = 1, sz = 1, ws = 16, hs = 12) => {
    const g = new THREE.SphereGeometry(r, ws, hs);
    g.scale(sx, sy, sz); g.translate(x, y, z);
    add(mat, g, color);
  };
  const lathe = (pts: THREE.Vector2[], seg: number, mat: THREE.Material, color: THREE.ColorRepresentation,
    x: number, y: number, z: number, sx = 1, sy = 1, sz = 1) => {
    const g = new THREE.LatheGeometry(pts, seg);
    g.rotateX(Math.PI / 2);
    g.scale(sx, sy, sz); g.translate(x, y, z);
    add(mat, g, color);
  };
  const torus = (R: number, r: number, mat: THREE.Material, color: THREE.ColorRepresentation,
    x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, arc = Math.PI * 2) => {
    const g = new THREE.TorusGeometry(R, r, 8, 20, arc);
    if (rx) g.rotateX(rx);
    if (ry) g.rotateY(ry);
    if (rz) g.rotateZ(rz);
    g.translate(x, y, z);
    add(mat, g, color);
  };
  const plane = (w: number, h: number, mat: THREE.Material, color: THREE.ColorRepresentation | undefined,
    x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, cast = false, receive = true) => {
    const g = new THREE.PlaneGeometry(w, h);
    if (rx) g.rotateX(rx);
    if (ry) g.rotateY(ry);
    if (rz) g.rotateZ(rz);
    g.translate(x, y, z);
    add(mat, g, color, cast, receive);
  };
  // Map a geometry's 0..1 UVs onto a rectangle of the 1024 print atlas.
  const atlas = (g: THREE.BufferGeometry, x: number, y: number, w: number, h: number) => {
    const uv = g.attributes.uv as THREE.BufferAttribute;
    const u0 = x / 1024, u1 = (x + w) / 1024, v0 = 1 - (y + h) / 1024, v1 = 1 - y / 1024;
    for (let i = 0; i < uv.count; i++) uv.setXY(i, u0 + uv.getX(i) * (u1 - u0), v0 + uv.getY(i) * (v1 - v0));
    uv.needsUpdate = true;
    return g;
  };
  const scaleUV = (g: THREE.BufferGeometry, sx: number, sy: number) => {
    const uv = g.attributes.uv as THREE.BufferAttribute;
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * sx, uv.getY(i) * sy);
    uv.needsUpdate = true;
    return g;
  };

  // ---------------------------------------------------------------- room shell
  // Pre-create batches that must not cast shadows (glass, paper, sky, light, room shell parts).
  batch(glass, false, true);
  batch(print, false, true);
  batch(sky, false, false);
  batch(emissive, false, false);
  batch(plaster, false, true);
  batch(floorMat, false, true);
  batch(rugMat, false, true);
  batch(tileMat, false, true);

  // A huge inside-out box: whatever the orbit does, there is never a void.
  const shellGeo = new THREE.BoxGeometry(6000, 6000, 4000);
  const shell = new THREE.Mesh(shellGeo, shellMat);
  shell.position.set(0, 0, 0);
  shell.receiveShadow = false;
  root.add(shell);
  geometries.push(shellGeo);

  const WALL = 0xf0e4d0, CEIL = 0xf7f0e3, TRIM = 0xfbf4e6;
  // floor, ceiling, back/side/front walls of the detailed room
  const floor = new THREE.PlaneGeometry(1800, 1730);
  floorTex.repeat.set(1.5, 1.44);
  floor.translate(0, -635, -300);
  add(floorMat, floor, undefined, false, true);
  const ceil = new THREE.PlaneGeometry(1800, 1730);
  ceil.rotateX(Math.PI); ceil.translate(0, -635, 1000);
  add(plaster, ceil, CEIL, false, true);
  plane(1800, 1300, plaster, WALL, 0, 230, 350, Math.PI / 2, 0, 0, false, true);       // back wall
  plane(1730, 1300, plaster, WALL, -900, -635, 350, 0, Math.PI / 2, 0, false, true);     // left wall
  plane(1730, 1300, plaster, WALL, 900, -635, 350, 0, -Math.PI / 2, 0, false, true);     // right wall
  plane(1800, 1300, plaster, WALL, 0, -1500, 350, -Math.PI / 2, 0, 0, false, true);      // front wall
  // baseboards and chair rail
  box(1800, 10, 18, paint, TRIM, 0, 225, -291);
  box(10, 1730, 18, paint, TRIM, -895, -635, -291);
  box(10, 1730, 18, paint, TRIM, 895, -635, -291);
  box(1800, 10, 14, paint, TRIM, 0, -1495, -291);
  for (const x of [-895, 895]) box(12, 1730, 10, paint, 0xe8dcc6, x, -635, -80);
  // ceiling beams
  for (const y of [10, -620]) box(1800, 90, 70, wood, 0x8a5f3b, 0, y, 965);
  // rug in front of the counter
  const rug = new THREE.PlaneGeometry(620, 260);
  rug.translate(0, -430, -299);
  add(rugMat, rug, undefined, false, true);
  // tall pantry left and tall fridge right (fridge body built later with metal batch)
  box(220, 250, 1000, paint, 0xd9d2bd, -490, 105, 200);            // pantry body z -300..700
  box(4, 238, 960, paint, 0xcfc7b0, -598, 105, 200);               // shadow side
  for (const z of [80, 340]) {
    box(206, 8, 244, paint, 0xe6ddc8, -490, -21, z);
    box(176, 4, 214, paint, 0xd2c9b2, -490, -26, z);
    cyl(3, 3, 26, 8, metal, 0xb08d4f, -490, -30, z + 90, 'x');
  }
  box(220, 8, 30, paint, 0xcdc4ad, -490, 105, 686);
  // wall art on the right wall and a small shelf on the left
  const art = new THREE.PlaneGeometry(190, 250);
  atlas(art, 380, 520, 300, 380);
  art.rotateY(-Math.PI / 2); art.translate(894, -520, 320);
  add(print, art, undefined, false, true);
  box(8, 214, 266, wood, 0x7a5636, 897, -520, 320, 0, 0, 0, false);
  // a second framed botanical on the left wall, near the counter
  const artL = new THREE.PlaneGeometry(200, 260);
  atlas(artL, 380, 520, 300, 380);
  artL.rotateY(Math.PI / 2); artL.translate(-896, -130, 270);
  add(print, artL, undefined, false, true);
  box(8, 224, 284, wood, 0x7a5636, -900, -130, 270, 0, 0, 0, false);
  // left wall shelf with two jars
  box(150, 26, 14, wood, 0x9a6c42, -886, -430, 180, 0, 0, 0, true);
  box(14, 100, 60, paint, 0xe6ddc8, -892, -430, 150);
  cyl(17, 17, 58, 14, glass, 0xffffff, -858, -450, 215);
  cyl(17, 17, 58, 14, glass, 0xffffff, -858, -404, 215);
  cyl(15, 15, 40, 14, matte, 0xb98a4a, -858, -450, 206);
  cyl(15, 15, 44, 14, matte, 0x8a5a3a, -858, -404, 208);
  cyl(18, 18, 7, 14, metal, 0x9a9488, -858, -450, 247);
  cyl(18, 18, 7, 14, metal, 0x9a9488, -858, -404, 247);
  // floor plant in the left corner
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(34, 0), new THREE.Vector2(40, 6), new THREE.Vector2(44, 48), new THREE.Vector2(40, 50),
    new THREE.Vector2(36, 48), new THREE.Vector2(32, 8), new THREE.Vector2(0, 8)], 20, glaze, 0xb4674a, -760, -80, -300);
  cyl(36, 34, 5, 18, matte, 0x4a3524, -760, -80, -252);
  for (let i = 0; i < 9; i++) {
    const a = (i / 9) * Math.PI * 2, r = 24 + (i % 3) * 18, h = 80 + (i % 4) * 34;
    sph(30, matte, i % 2 ? 0x5f8f4e : 0x4f7d40, -760 + Math.cos(a) * r, -80 + Math.sin(a) * r, -210 + h, 0.5, 0.9, 0.25, 10, 8);
  }

  // ---------------------------------------------------------------- counter + sink
  // Butcher-block top with a real sink cutout; world-space UVs on the caps.
  const shape = new THREE.Shape();
  shape.moveTo(-300, -200); shape.lineTo(300, -200); shape.lineTo(300, 200); shape.lineTo(-300, 200); shape.closePath();
  const hole = new THREE.Path();
  hole.moveTo(-275, -25); hole.lineTo(-275, 45); hole.lineTo(-165, 45); hole.lineTo(-165, -25); hole.closePath();
  shape.holes.push(hole);
  const topGeo = new THREE.ExtrudeGeometry(shape, { depth: 26, bevelEnabled: true, bevelThickness: 1.2, bevelSize: 1.2, bevelSegments: 1, curveSegments: 1 });
  topGeo.translate(0, 0, -27.2);
  {
    const uv = topGeo.attributes.uv as THREE.BufferAttribute;
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) / 600 + 0.5, uv.getY(i) / 400 + 0.5);
  }
  add(woodTop, topGeo);

  // hollow cabinet shell so the sink can hang inside it
  const SAGE = 0x7f9a83, SAGE_D = 0x6d8672, SAGE_L = 0x8fa995, DARK = 0x3f3a32;
  box(16, 376, 264, paint, SAGE_D, -292, 0, -158);                   // left side
  box(16, 376, 264, paint, SAGE_D, 292, 0, -158);                    // right side
  box(560, 14, 264, paint, SAGE_D, 0, 187, -158);                    // back
  box(560, 6, 264, paint, DARK, 0, -180, -158);                      // dark front backing (visible in the gaps)
  box(560, 376, 12, paint, DARK, 0, 0, -284);                        // bottom
  box(560, 12, 20, paint, 0x4c463c, 0, -174, -280);                  // recessed toe kick
  // doors and drawers: recessed panel + raised shaker frame, in front of the dark backing
  const door = (cx: number, z: number, w: number, h: number) => {
    box(w, 6, h, paint, SAGE, cx, -186, z);                          // centre panel, front at -189
    box(w, 5, 16, paint, SAGE_L, cx, -192, z + h / 2 - 8);           // frame, front at -194.5
    box(w, 5, 16, paint, SAGE_L, cx, -192, z - h / 2 + 8);
    box(16, 5, h, paint, SAGE_L, cx - w / 2 + 8, -192, z);
    box(16, 5, h, paint, SAGE_L, cx + w / 2 - 8, -192, z);
    box(w - 22, 1.5, 2, paint, SAGE_D, cx, -189.2, z + h / 2 - 17);  // shadow line inside the frame
    box(w - 22, 1.5, 2, paint, SAGE_D, cx, -189.2, z - h / 2 + 17);
    box(2, 1.5, h - 32, paint, SAGE_D, cx - w / 2 + 17, -189.2, z);
    box(2, 1.5, h - 32, paint, SAGE_D, cx + w / 2 - 17, -189.2, z);
  };
  door(-190, -155, 196, 226);
  door(0, -155, 180, 226);
  // right-hand drawer stack
  for (const [z, h] of [[-62, 70], [-140, 66], [-226, 66]] as [number, number][]) {
    box(176, 6, h, paint, SAGE, 190, -186, z);
    box(176, 5, 12, paint, SAGE_L, 190, -192, z + h / 2 - 6);
    box(176, 5, 12, paint, SAGE_L, 190, -192, z - h / 2 + 6);
    box(12, 5, h, paint, SAGE_L, 190 - 82, -192, z);
    box(12, 5, h, paint, SAGE_L, 190 + 82, -192, z);
  }
  // brass bar handles
  const handle = (x: number, z: number, len: number) => {
    cyl(3.2, 3.2, len, 8, metal, 0xb08d4f, x, -199, z, 'x');
    for (const dx of [-len / 2 + 4, len / 2 - 4]) cyl(2.4, 2.4, 8, 6, metal, 0xb08d4f, x + dx, -196, z, 'y');
  };
  handle(-190, -60, 52); handle(0, -60, 46);
  handle(190, -62, 48); handle(190, -140, 44); handle(190, -226, 44);
  // sink basin and rim
  const STEEL = 0xb9bec2;
  box(110, 70, 3, metal, STEEL, -220, 10, -87);
  box(2, 70, 88, metal, STEEL, -274, 10, -44);
  box(2, 70, 88, metal, STEEL, -166, 10, -44);
  box(110, 2, 88, metal, STEEL, -220, -24, -44);
  box(110, 2, 88, metal, STEEL, -220, 44, -44);
  box(114, 4, 2, metal, STEEL, -220, -25, 0.6);
  box(114, 4, 2, metal, STEEL, -220, 45, 0.6);
  // drain and basket strainer
  cyl(19, 19, 1.6, 22, metal, 0x8a8f92, -220, 10, -85.4);
  torus(16, 1.5, metal, 0x6f7478, -220, 10, -84.9);
  box(30, 2.4, 1, metal, 0x5f6468, -220, 10, -85);
  box(2.4, 30, 1, metal, 0x5f6468, -220, 10, -85);
  cyl(4, 4, 1.4, 10, matte, 0x2f3336, -220, 10, -84.6);
  box(4, 74, 2, metal, STEEL, -275, 10, 0.6);
  box(4, 74, 2, metal, STEEL, -165, 10, 0.6);
  cyl(9, 9, 1.2, 16, metal, 0x8d9296, -220, 10, -85);
  for (let i = 0; i < 3; i++) box(14, 1, 0.8, metal, 0x6f7478, -220, 4 + i * 6, -84.2);
  // gooseneck faucet
  cyl(11, 12, 7, 18, metal, 0xb08d4f, -220, 76, 3.5);
  cyl(4.2, 4.6, 112, 14, metal, STEEL, -220, 76, 63);
  const faucetCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-220, 76, 118), new THREE.Vector3(-220, 76, 145), new THREE.Vector3(-220, 60, 154),
    new THREE.Vector3(-220, 34, 146), new THREE.Vector3(-220, 24, 126), new THREE.Vector3(-220, 24, 116)
  ]);
  const faucetSpout = new THREE.TubeGeometry(faucetCurve, 26, 3.6, 8, false);
  add(metal, faucetSpout, STEEL);
  cyl(4, 4, 6, 12, metal, STEEL, -220, 24, 113);
  cyl(5, 5, 1.4, 12, metal, 0x7f8488, -220, 24, 109.8);
  // lever
  box(4, 4, 26, metal, 0xb08d4f, -207, 76, 12, 0, 0.5);
  sph(4, metal, 0xb08d4f, -207, 70, 24, 1, 1, 0.6);
  // sink-side sponge / soap
  box(24, 18, 12, matte, 0xd9c34a, -250, 82, 6, 0.3);
  cyl(7, 7, 22, 12, glaze, 0xdfe8e2, -174, 86, 11);
  cyl(3, 3, 8, 8, metal, 0x9a9488, -174, 86, 26);

  // ---------------------------------------------------------------- backsplash, window
  // Tiles stop under the sill; taller tiled panels frame the window on both sides.
  const tilePlane = new THREE.PlaneGeometry(660, 88);
  scaleUV(tilePlane, 660 / 330, 88 / 330);
  tilePlane.rotateX(Math.PI / 2); tilePlane.translate(0, 228.6, 44);
  add(tileMat, tilePlane, undefined, false, true);
  const tileSide = (w: number, cx: number) => {
    const g = new THREE.PlaneGeometry(w, 160);
    scaleUV(g, w / 330, 160 / 330);
    g.rotateX(Math.PI / 2); g.translate(cx, 228.6, 170);
    add(tileMat, g, undefined, false, true);
  };
  tileSide(120, -270);
  tileSide(210, 225);
  // window: bright sky, cream frame, mullions, sill, curtains
  const skyPlane = new THREE.PlaneGeometry(300, 190);
  skyPlane.rotateX(Math.PI / 2); skyPlane.translate(-40, 229.2, 185);
  add(sky, skyPlane, undefined, false, false);
  box(18, 14, 220, paint, TRIM, -199, 224, 185);                    // outer frame
  box(18, 14, 220, paint, TRIM, 119, 224, 185);
  box(336, 14, 18, paint, TRIM, -40, 224, 289);
  box(336, 14, 18, paint, TRIM, -40, 224, 81);
  box(9, 10, 190, paint, 0xefe6d2, -40, 226, 185);                  // mullions
  box(300, 10, 9, paint, 0xefe6d2, -40, 226, 185);
  box(360, 62, 14, paint, 0xf3e8d2, -40, 208, 87);                  // sill
  box(360, 10, 6, paint, 0xe6dcc4, -40, 178, 79);
  box(336, 8, 12, paint, TRIM, -40, 224, 72);                       // apron
  // curtain rod + two gathered curtains
  cyl(3, 3, 400, 10, metal, 0xb08d4f, -40, 199, 303, 'x');
  sph(6, metal, 0xb08d4f, -240, 199, 303, 1, 1, 1, 10, 8);
  sph(6, metal, 0xb08d4f, 160, 199, 303, 1, 1, 1, 10, 8);
  const curtain = (x0: number, x1: number, z0: number, z1: number, tintC: number) => {
    const nx = 12, nz = 20, pos: number[] = [], uv: number[] = [], idx: number[] = [];
    for (let j = 0; j <= nz; j++) for (let i = 0; i <= nx; i++) {
      const u = i / nx, v = j / nz;
      const x = lerp(x0, x1, u), z = lerp(z0, z1, v);
      const y = 198 + Math.sin(u * Math.PI * 4.5) * 7 * (0.6 + v * 0.4) + Math.sin(v * 5 + u * 3) * 1.4;
      pos.push(x, y, z); uv.push(u, v);
    }
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
      const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx); g.computeVertexNormals();
    add(fabric, g, tintC);
  };
  curtain(-238, -158, 112, 300, 0xe9d8bb);
  curtain(78, 158, 112, 300, 0xe9d8bb);
  // herbs on the sill
  const herb = (hx: number, hy: number, s: number) => {
    lathe([new THREE.Vector2(0, 0), new THREE.Vector2(13, 0), new THREE.Vector2(16, 4), new THREE.Vector2(18, 26), new THREE.Vector2(15, 27),
      new THREE.Vector2(13, 24), new THREE.Vector2(12, 5), new THREE.Vector2(0, 5)], 16, glaze, 0xb4674a, hx, hy, 94);
    cyl(14, 13, 3, 14, matte, 0x4a3524, hx, hy, 121);
    for (let i = 0; i < 11; i++) {
      const a = (i / 11) * Math.PI * 2 + s, r = 4 + (i % 3) * 5, h = 14 + (i % 4) * 7;
      const g = new THREE.CylinderGeometry(0.8, 0.9, h, 4);
      g.translate(hx + Math.cos(a) * r * 0.5, hy + Math.sin(a) * r * 0.5, 122 + h / 2);
      add(matte, g, 0x5f8a4a);
      sph(7, matte, i % 2 ? 0x6f9c55 : 0x5a8746, hx + Math.cos(a) * r, hy + Math.sin(a) * r, 126 + h, 0.9, 0.55, 0.25, 8, 6);
    }
  };
  herb(-155, 216, 0.4);
  herb(40, 216, 1.7);

  // ---------------------------------------------------------------- upper cabinet, shelves, clock, pendant
  // wall cabinet left of the window
  box(165, 110, 220, paint, SAGE, -287, 175, 360);
  box(170, 114, 10, paint, SAGE_L, -287, 175, 473);
  box(170, 116, 8, paint, SAGE_D, -287, 175, 247);
  for (const dx of [-41, 41]) {
    box(76, 8, 200, paint, SAGE, -287 + dx, 117, 360);
    box(56, 3, 180, paint, SAGE_L, -287 + dx, 112.4, 360);
    box(48, 3, 170, paint, SAGE_D, -287 + dx, 111.6, 360);
    cyl(3.2, 3.2, 12, 8, metal, 0xb08d4f, -287 + dx + 30, 113, 300, 'y');
  }
  // brass rail and hanging utensils under the cabinet
  cyl(2.4, 2.4, 140, 8, metal, 0xb08d4f, -287, 150, 238, 'x');
  for (const [ux, kind] of [[-340, 0], [-292, 1], [-240, 2]] as [number, number][]) {
    torus(5, 1.4, metal, 0xb08d4f, ux, 150, 238, Math.PI / 2);
    if (kind === 0) { cyl(1.4, 1.4, 50, 6, wood, 0x8a5f3b, ux, 150, 210); box(18, 4, 26, wood, 0x9a6c42, ux, 150, 184); }
    if (kind === 1) { cyl(1.4, 1.4, 44, 6, wood, 0x8a5f3b, ux, 150, 214); for (let i = 0; i < 5; i++) cyl(0.7, 0.7, 22, 4, metal, 0xb9bec2, ux - 5 + i * 2.5, 150, 194, 'z'); }
    if (kind === 2) { cyl(1.4, 1.4, 36, 6, wood, 0x8a5f3b, ux, 150, 218); sph(8, metal, 0xb9bec2, ux, 150, 198, 1, 1, 0.5, 12, 8); }
  }
  // open shelves right of the window
  for (const z of [258, 368]) {
    box(204, 112, 16, wood, 0x9a6c42, 240, 174, z);
    box(204, 6, 4, wood, 0xc08a58, 240, 120, z - 9);
  }
  for (const x of [155, 325]) {
    box(6, 12, 54, metal, 0x5a5650, x, 126, 232);
    box(6, 86, 6, metal, 0x5a5650, x, 165, 248);
  }
  // plates on the lower shelf
  const plate = (px: number, py: number, pz: number, col: number) => {
    lathe([new THREE.Vector2(0, 0), new THREE.Vector2(28, 0), new THREE.Vector2(34, 2), new THREE.Vector2(37, 6),
      new THREE.Vector2(35, 8), new THREE.Vector2(32, 5), new THREE.Vector2(28, 3), new THREE.Vector2(0, 3)], 24, glaze, col, px, py, pz);
  };
  for (let i = 0; i < 4; i++) plate(166, 180, 266 + i * 5, i % 2 ? 0xf4ead6 : 0xe9dfc8);
  // jars on the lower shelf
  const jar = (jx: number, jy: number, jz: number, fill: number, fillH: number) => {
    cyl(16, 16, 56, 16, glass, 0xffffff, jx, jy, jz + 28);
    cyl(14, 14, fillH, 16, matte, fill, jx, jy, jz + fillH / 2 + 2);
    cyl(18, 18, 7, 16, metal, 0x9a9488, jx, jy, jz + 59);
    cyl(3, 3, 4, 8, metal, 0x7a7468, jx, jy, jz + 64);
  };
  jar(224, 184, 266, 0xc98a3c, 34);
  jar(262, 178, 266, 0x9a5a32, 42);
  jar(300, 184, 266, 0xd9b45a, 30);
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(16, 0), new THREE.Vector2(22, 6), new THREE.Vector2(24, 16), new THREE.Vector2(21, 17),
    new THREE.Vector2(19, 14), new THREE.Vector2(17, 6), new THREE.Vector2(0, 5)], 18, glaze, 0x2a9d8f, 330, 178, 266);
  // upper shelf: a jug and a small potted plant
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(16, 0), new THREE.Vector2(20, 8), new THREE.Vector2(18, 34), new THREE.Vector2(13, 36),
    new THREE.Vector2(12, 30), new THREE.Vector2(14, 10), new THREE.Vector2(0, 8)], 18, glaze, 0xf0e6d0, 305, 180, 376);
  cyl(13, 11, 26, 14, glaze, 0xb4674a, 250, 180, 376);
  cyl(11, 10, 2, 12, matte, 0x4a3524, 250, 180, 403);
  for (let i = 0; i < 7; i++) {
    const a = (i / 7) * Math.PI * 2;
    sph(8, matte, i % 2 ? 0x6f9c55 : 0x5a8746, 250 + Math.cos(a) * 10, 180 + Math.sin(a) * 10, 408 + (i % 3) * 5, 0.9, 0.5, 0.25, 8, 6);
  }
  // wall clock above the window
  const face = new THREE.CircleGeometry(45, 40);
  atlas(face, 16, 520, 300, 300);
  face.rotateX(Math.PI / 2); face.translate(-40, 225.2, 390);
  add(print, face, undefined, false, true);
  torus(46, 4.2, wood, 0x8a5f3b, -40, 225.2, 390, Math.PI / 2);
  const hand = (len: number, wid: number, ang: number, col: number) => {
    const g = new THREE.BoxGeometry(wid, 1.6, len);
    g.translate(0, 0, len / 2 - 4); g.rotateY(ang); g.translate(-40, 224.2, 390);
    add(metal, g, col);
  };
  hand(30, 4, -Math.PI / 3, 0x3a332c);
  hand(40, 2.6, Math.PI / 3, 0x3a332c);
  hand(43, 1, Math.PI * 0.94, 0xb23a2e);
  sph(3.5, metal, 0xb08d4f, -40, 224, 390, 1, 0.4, 1, 10, 8);
  // pendant lamp over the right of the counter
  const PX = 248, PY = -6;
  cyl(18, 18, 7, 18, metal, 0xb08d4f, PX, PY, 995);
  cyl(1.6, 1.6, 367, 6, metal, 0x3f3a32, PX, PY, 811);
  lathe([new THREE.Vector2(5, 622), new THREE.Vector2(16, 619), new THREE.Vector2(48, 597), new THREE.Vector2(70, 567),
    new THREE.Vector2(73, 560), new THREE.Vector2(67, 560), new THREE.Vector2(64, 567), new THREE.Vector2(43, 593),
    new THREE.Vector2(14, 615), new THREE.Vector2(5, 615), new THREE.Vector2(5, 622)], 28, glaze, 0xb4674a, PX, PY, 0);
  torus(70, 2.4, metal, 0xb08d4f, PX, PY, 560, Math.PI / 2);
  sph(12, emissive, 0xffe3ae, PX, PY, 583, 1, 1, 1, 16, 12);
  const pendant = new THREE.PointLight(0xffc98a, 0.55, 700, 2);
  pendant.position.set(PX, PY, 583);
  root.add(pendant);


  // ---------------------------------------------------------------- counter props
  // Hero: fruit bowl at (165, 40), outside the free middle.
  const BX = 165, BY = 40;
  lathe([new THREE.Vector2(0, 1), new THREE.Vector2(24, 0), new THREE.Vector2(30, 3), new THREE.Vector2(44, 14),
    new THREE.Vector2(52, 30), new THREE.Vector2(54, 36), new THREE.Vector2(50, 38), new THREE.Vector2(47, 35),
    new THREE.Vector2(43, 29), new THREE.Vector2(34, 16), new THREE.Vector2(26, 7), new THREE.Vector2(0, 6)],
    32, glaze, 0x2a9d8f, BX, BY, 0);
  const apple = (ax: number, ay: number, az: number, r: number, col: number) => {
    sph(r, glaze, col, ax, ay, az, 1, 1, 0.9);
    cyl(0.9, 0.9, 7, 5, matte, 0x5a3a1e, ax, ay, az + r * 0.85);
    sph(5, matte, 0x5f8a4a, ax + 4, ay, az + r * 0.9, 1, 0.45, 0.2, 8, 6);
  };
  apple(BX - 16, BY - 2, 24, 15, 0xd8453a);
  apple(BX + 12, BY + 10, 26, 13, 0x9bbf4a);
  sph(14, glaze, 0xf28c28, BX + 12, BY - 10, 23, 1, 1, 0.94, 18, 12);
  for (let i = 0; i < 28; i++) {
    const a = rnd() * Math.PI * 2, b = Math.acos(2 * rnd() - 1);
    sph(0.9, matte, 0xd97a20, BX + 12 + Math.sin(b) * Math.cos(a) * 13, BY - 10 + Math.sin(b) * Math.sin(a) * 13, 23 + Math.cos(b) * 12, 1, 1, 1, 5, 4);
  }
  // pear
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(9, 0), new THREE.Vector2(13, 8), new THREE.Vector2(11, 18),
    new THREE.Vector2(8, 26), new THREE.Vector2(6, 34), new THREE.Vector2(0, 36)], 18, glaze, 0xc9d16a, BX - 4, BY + 22, 8);
  cyl(0.8, 0.8, 7, 5, matte, 0x5a3a1e, BX - 4, BY + 22, 46);
  // banana draped over the rim
  const bananaCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(BX - 46, BY + 20, 40), new THREE.Vector3(BX - 28, BY + 24, 48),
    new THREE.Vector3(BX - 2, BY + 22, 50), new THREE.Vector3(BX + 26, BY + 16, 44), new THREE.Vector3(BX + 42, BY + 8, 34)
  ]);
  add(glaze, new THREE.TubeGeometry(bananaCurve, 26, 4.6, 8, false), 0xf6cf3e);
  cyl(3, 1.4, 6, 6, matte, 0x6b5a2b, BX + 42, BY + 8, 34);
  // grapes
  for (let i = 0; i < 9; i++) sph(4.4, glaze, i % 2 ? 0x6d3f72 : 0x5b335f, BX + 24 + (i % 3) * 5, BY + 18 + ((i / 3) | 0) * 4, 20 + (i % 2) * 4, 1, 1, 1, 8, 6);

  // Mug with handle and coffee, on a cork coaster.
  const MX = 150, MY = -96;
  cyl(29, 29, 3, 22, matte, 0xa87a4a, MX, MY, 1.5);
  for (let i = 0; i < 18; i++) { const a = rnd() * 6.28, r = rnd() * 24; sph(0.8, matte, 0x7a522e, MX + Math.cos(a) * r, MY + Math.sin(a) * r, 3.1, 1, 1, 0.3, 4, 3); }
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(18, 0), new THREE.Vector2(20, 3), new THREE.Vector2(20, 44),
    new THREE.Vector2(17, 46), new THREE.Vector2(16, 44), new THREE.Vector2(16, 4), new THREE.Vector2(0, 3)], 24, glaze, 0xef7f74, MX, MY, 3);
  torus(10, 3.2, glaze, 0xef7f74, MX - 21, MY, 27, Math.PI / 2);
  cyl(15.4, 15.4, 1, 18, glaze, 0x3b2416, MX, MY, 45.5);
  sph(7, matte, 0xf3e6d0, MX + 10, MY - 7, 47, 1, 1, 0.25, 10, 6);
  // little biscuit plate
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(24, 0), new THREE.Vector2(30, 3), new THREE.Vector2(31, 6),
    new THREE.Vector2(28, 6), new THREE.Vector2(25, 3), new THREE.Vector2(0, 3)], 22, glaze, 0xf2e8d4, 208, -122, 0);
  for (const [cx, cy, ca] of [[202, -126, 0.3], [214, -118, 1.2]] as [number, number, number][]) {
    cyl(9, 9, 3.4, 14, matte, 0xc98a4a, cx, cy, 5, 'z');
    for (let i = 0; i < 5; i++) { const a = rnd() * 6.28, r = rnd() * 6; sph(1.2, matte, 0x4a2a18, cx + Math.cos(a) * r, cy + Math.sin(a) * r, 7, 1, 1, 0.5, 5, 4); }
  }

  // Cutting board with a knife, lemon and a few crumbs.
  box(150, 92, 8, wood, 0xd2a46a, -146, -122, 4, 0.14);
  box(146, 86, 1.2, wood, 0xe0b77e, -146, -122, 8.2, 0.14);
  // knife
  const knifeG = new THREE.BoxGeometry(52, 7, 1.2); knifeG.rotateZ(0.14); knifeG.translate(-160, -132, 9.2);
  add(metal, knifeG, 0xb9bec2);
  box(26, 11, 5, wood, 0x4a2f1c, -121, -128.5, 10, 0.14);
  sph(10, glaze, 0xe8c84a, -120, -105, 15, 1.25, 0.9, 0.9, 14, 10);
  cyl(2.6, 2.6, 3, 8, matte, 0x8a7a3a, -132, -105, 16);
  for (let i = 0; i < 12; i++) {
    const cx = -180 + rnd() * 100, cy = -160 + rnd() * 60;
    sph(0.5 + rnd() * 0.7, crust, 0xffffff, cx, cy, 0.4, 1, 1, 0.5, 5, 4);
  }

  // Jam and honey jars with a paper label.
  const jam = (jx: number, jy: number, jh: number, fillCol: number, label: boolean) => {
    cyl(17, 17, jh, 20, glass, 0xffffff, jx, jy, jh / 2);
    cyl(15, 15, jh - 14, 20, glaze, fillCol, jx, jy, (jh - 14) / 2 + 2);
    cyl(18.5, 18.5, 8, 20, metal, 0xb08d4f, jx, jy, jh + 3);
    cyl(4, 4, 2, 10, metal, 0x8a6a38, jx, jy, jh + 7.5);
    if (label) {
      const lg = new THREE.CylinderGeometry(17.7, 17.7, jh * 0.6, 18, 1, true, -1.1, 2.2);
      atlas(lg, 16, 16, 300, 300);
      lg.rotateX(Math.PI / 2);
      lg.translate(jx, jy, jh * 0.45);
      add(print, lg, undefined, false, true);
    }
  };
  jam(82, 132, 54, 0x8e2330, true);
  jam(42, 138, 42, 0xd39a2e, false);
  cyl(0.9, 0.9, 34, 5, wood, 0xc9a06a, 42, 138, 44, 'z');          // honey dipper
  sph(4.5, wood, 0xc9a06a, 42, 138, 28, 1, 1, 1.4, 8, 6);

  // Sugar bowl, salt and pepper on a little tray.
  box(96, 46, 4, wood, 0xb98a55, -28, 150, 2);
  box(96, 4, 6, wood, 0xa97742, -28, 128, 3);
  box(96, 4, 6, wood, 0xa97742, -28, 172, 3);
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(16, 0), new THREE.Vector2(22, 7), new THREE.Vector2(24, 20),
    new THREE.Vector2(21, 22), new THREE.Vector2(19, 19), new THREE.Vector2(17, 8), new THREE.Vector2(0, 6)], 20, glaze, 0xf1e7d2, -62, 150, 4);
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(22, 0), new THREE.Vector2(24, 3), new THREE.Vector2(20, 8), new THREE.Vector2(0, 8)], 20, glaze, 0xf1e7d2, -62, 150, 24);
  sph(5, glaze, 0xf1e7d2, -62, 150, 36, 1, 1, 1, 10, 8);
  box(4, 1.4, 14, metal, 0xb9bec2, -45, 150, 28, 0.5);
  for (const [sx, sy, col] of [[-22, 146, 0xf4ecdc], [-2, 154, 0xe8dcc4]] as [number, number, number][]) {
    cyl(8, 9, 30, 14, glaze, col, sx, sy, 19);
    cyl(8.4, 8.4, 6, 14, metal, 0xb9bec2, sx, sy, 36);
    for (let i = 0; i < 5; i++) { const a = (i / 5) * 6.28; sph(0.8, matte, 0x4a3a2a, sx + Math.cos(a) * 3, sy + Math.sin(a) * 3, 39, 1, 1, 0.4, 4, 3); }
  }

  // Kettle (teal enamel, brass handle) under the left cabinet.
  const KX = -180, KY = 152;
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(28, 0), new THREE.Vector2(40, 9), new THREE.Vector2(42, 34),
    new THREE.Vector2(36, 48), new THREE.Vector2(20, 54), new THREE.Vector2(0, 54)], 28, glaze, 0x4f8f86, KX, KY, 0);
  cyl(18, 20, 4, 18, metal, 0xb08d4f, KX, KY, 55);
  cyl(12, 15, 6, 18, glaze, 0x4f8f86, KX, KY, 59);
  sph(4, metal, 0xb08d4f, KX, KY, 64, 1, 1, 0.8, 10, 8);
  const kettleSpout = new THREE.CatmullRomCurve3([
    new THREE.Vector3(KX + 34, KY, 18), new THREE.Vector3(KX + 48, KY, 30), new THREE.Vector3(KX + 55, KY, 46), new THREE.Vector3(KX + 57, KY, 54)
  ]);
  add(glaze, new THREE.TubeGeometry(kettleSpout, 12, 5, 8, false), 0x4f8f86);
  torus(24, 3.4, metal, 0xb08d4f, KX, KY, 40, Math.PI / 2, 0, 0, Math.PI);
  box(2.5, 4, 14, metal, 0xb08d4f, KX - 28, KY, 35, 0, 0.4);

  // Bread basket with a linen cloth and a scored loaf.
  const BXK = 225, BYK = 126;
  lathe([new THREE.Vector2(0, 0), new THREE.Vector2(38, 0), new THREE.Vector2(44, 7), new THREE.Vector2(50, 36),
    new THREE.Vector2(47, 38), new THREE.Vector2(44, 34), new THREE.Vector2(40, 8), new THREE.Vector2(0, 6)], 26, wicker, 0xffffff, BXK, BYK, 0);
  cyl(42, 42, 3, 24, fabric, 0xf0e2c8, BXK, BYK, 33);
  for (let i = 0; i < 8; i++) { const a = (i / 8) * 6.28; sph(12, fabric, 0xe8d6b8, BXK + Math.cos(a) * 38, BYK + Math.sin(a) * 38, 38, 1.2, 0.9, 0.35, 8, 6); }
  sph(30, crust, 0xffffff, BXK - 4, BYK, 52, 1.45, 0.7, 0.62, 20, 14);
  for (let i = 0; i < 3; i++) {
    const dx = -14 + i * 14, dy = -4 + i * 2;
    const q = 1 - (dx / 43.5) * (dx / 43.5) - (dy / 21) * (dy / 21);
    const sz = 52 + 18.6 * Math.sqrt(Math.max(0.08, q)) - 0.6;
    const g = new THREE.BoxGeometry(24, 3, 2.6);
    g.rotateZ(0.22); g.translate(BXK - 4 + dx, BYK + dy, sz);
    add(matte, g, 0x7a441e);
  }
  sph(18, crust, 0xffffff, BXK + 30, BYK - 22, 44, 1.1, 0.85, 0.8, 14, 10);
  // crumbs and a jam spill
  for (let i = 0; i < 16; i++) sph(0.6 + rnd() * 0.9, crust, 0xffffff, BXK - 40 + rnd() * 90, BYK - 50 + rnd() * 70, 0.5, 1, 1, 0.5, 5, 4);
  sph(11, glaze, 0x7c1f2a, 198, 78, 0.6, 1.5, 0.9, 0.07, 18, 10);
  sph(4, glaze, 0x7c1f2a, 216, 70, 0.5, 1.3, 0.9, 0.1, 10, 8);
  sph(2.6, glaze, 0x7c1f2a, 190, 90, 0.5, 1.2, 0.8, 0.12, 8, 6);
  sph(3.2, glaze, 0x7c1f2a, 208, 88, 0.5, 1.6, 0.7, 0.1, 8, 6);

  // Knife block at the right edge.
  box(46, 58, 104, wood, 0x6b4a2b, 266, 6, 52, 0.08);
  box(40, 52, 8, wood, 0x7d5a37, 266, 6, 106, 0.08);
  for (let i = 0; i < 3; i++) {
    box(3, 22, 3, matte, 0x2a2018, 254 + i * 12, 6, 109, 0.08);
    const hg = new THREE.BoxGeometry(8, 14, 30);
    hg.rotateX(-0.3); hg.rotateY(0.3); hg.translate(254 + i * 12, 12, 124);
    add(wood, hg, i % 2 ? 0x3f2a18 : 0x5a3a20);
  }

  // Cereal box, printed front.
  box(64, 32, 150, paint, 0xe0a03a, -268, 176, 75, -0.12);
  const cereal = new THREE.PlaneGeometry(58, 142);
  atlas(cereal, 340, 16, 320, 420);
  cereal.rotateX(Math.PI / 2); cereal.rotateZ(-0.12); cereal.translate(-268, 159.4, 76);
  add(print, cereal, undefined, false, true);
  box(64, 32, 6, paint, 0xd08a2a, -268, 176, 152, -0.12);

  // A dish towel draped over the front edge, with woven stripes near the hem.
  {
    const nx = 24, nz = 34, pos: number[] = [], uv: number[] = [], idx: number[] = [], col: number[] = [];
    const cream = new THREE.Color(0xe9d9bf), hem = new THREE.Color(0xdcc9ab), stripe = new THREE.Color(0xb6543f);
    for (let j = 0; j <= nz; j++) for (let i = 0; i <= nx; i++) {
      const u = i / nx, v = j / nz;
      const x = lerp(28, 108, u) + Math.sin(u * Math.PI * 3.2) * 1.4;
      let y: number, z: number;
      if (v < 0.22) {
        const t = v / 0.22;
        y = lerp(-158, -198.4, t);
        z = 1.7 + Math.sin(u * Math.PI * 3) * 1.1 + Math.sin(u * Math.PI * 7) * 0.35;
      } else if (v < 0.29) {
        const t = (v - 0.22) / 0.07;
        y = -198.4 - Math.sin(t * Math.PI) * 4.6;
        z = 1.7 - t * 4.6;
      } else {
        const t = (v - 0.29) / 0.71;
        y = -203 - Math.sin(u * Math.PI * 5) * 2.4 * (0.55 + t * 0.45);
        z = -2.9 - t * 166;
      }
      pos.push(x, y, z); uv.push(u, v);
      const c = (v > 0.70 && v < 0.755) || (v > 0.80 && v < 0.845) ? stripe : (v > 0.93 ? hem : cream);
      col.push(c.r, c.g, c.b);
    }
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
      const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals();
    add(fabric, g, undefined);
  }

  // ---------------------------------------------------------------- tall fridge (right background)
  const FX = 480, FY = 105, FZ = 150;
  box(240, 250, 900, metal, 0xb4b8b6, FX, FY, FZ);
  box(246, 254, 8, metal, 0x9aa0a4, FX, FY, 604);
  box(246, 254, 10, metal, 0x6f7478, FX, FY, -296);
  for (const dx of [-57, 57]) {
    box(116, 4, 698, matte, 0x3f403c, FX + dx, -21, 245);            // dark gasket border
    box(110, 6, 690, metal, 0xc6c9c6, FX + dx, -23, 245);            // door
    box(104, 2, 684, metal, 0xb2b6b4, FX + dx, -26.4, 245);          // recessed inner face
    box(2, 2, 660, metal, 0xdfe2df, FX + dx - 48, -27.2, 245);       // highlight edge
    cyl(3.8, 3.8, 300, 8, metal, 0xb08d4f, FX + dx + (dx < 0 ? 44 : -44), -34, 290);
    for (const z of [80, 400]) cyl(2.6, 2.6, 14, 6, metal, 0xb08d4f, FX + dx + (dx < 0 ? 44 : -44), -28, z, 'y');
    for (let i = 0; i < 3; i++) box(70, 1.4, 2.4, matte, 0x7d827e, FX + dx, -27, 520 + i * 9);
  }
  box(6, 8, 690, matte, 0x4a4a46, FX, -25, 245);
  box(30, 3, 10, metal, 0xb08d4f, FX, -27.5, 575);
  box(18, 1.5, 2, matte, 0x3f403c, FX, -28.8, 575);
  box(220, 6, 120, metal, 0xc2c7cb, FX, -23, -230);
  box(226, 4, 126, matte, 0x3f403c, FX, -21, -230);
  cyl(3.8, 3.8, 150, 8, metal, 0xb08d4f, FX, -32, -196, 'x');
  // magnets and a child's drawing
  for (const [mx, mz, mc] of [[-90, 360, 0xd8453a], [-72, 300, 0xf2b134], [-100, 250, 0x2a9d8f]] as [number, number, number][]) {
    box(12, 3, 12, matte, mc, FX + mx, -27, mz);
  }
  const kid = new THREE.PlaneGeometry(78, 96);
  atlas(kid, 380, 520, 300, 380);
  kid.rotateX(Math.PI / 2); kid.translate(FX - 30, -29, 190);
  add(print, kid, undefined, false, true);
  for (const [tx, tz] of [[-64, 238], [4, 238]] as [number, number][]) box(12, 2, 8, matte, 0xe8dcc4, FX - 30 + tx, -30, tz);

  // ---------------------------------------------------------------- books (InstancedMesh)
  const bookGeo = new THREE.BoxGeometry(1, 1, 1);
  const bookMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.82, metalness: 0 });
  const books = new THREE.InstancedMesh(bookGeo, bookMat, 6);
  const bookCols = [0xb6543f, 0x2a6f68, 0xd9a33a, 0x7d5a37, 0x5b6f8a, 0xc98a4a];
  const bm = new THREE.Matrix4(), bq = new THREE.Quaternion(), bs = new THREE.Vector3(), bp = new THREE.Vector3();
  for (let i = 0; i < 6; i++) {
    const h = 54 + (i % 3) * 9, th = 9 + (i % 2) * 4, lean = i === 5 ? -0.22 : (i === 4 ? 0.12 : 0);
    bq.setFromEuler(new THREE.Euler(0, lean, 0));
    bp.set(150 + i * 15.5, 178, 376 + h / 2);
    bs.set(th, 58, h);
    bm.compose(bp, bq, bs);
    books.setMatrixAt(i, bm);
    books.setColorAt(i, new THREE.Color(bookCols[i]));
  }
  books.instanceMatrix.needsUpdate = true;
  if (books.instanceColor) books.instanceColor.needsUpdate = true;
  books.castShadow = true; books.receiveShadow = true;
  root.add(books);
  geometries.push(bookGeo); materials.push(bookMat);

  // ---------------------------------------------------------------- finalize
  for (const b of batches.values()) {
    if (!b.geos.length) continue;
    const merged = b.geos.length === 1 ? b.geos[0] : mergeGeometries(b.geos, false);
    if (!merged) continue;
    if (merged !== b.geos[0]) { for (const g of b.geos) g.dispose(); }
    const mesh = new THREE.Mesh(merged, b.mat);
    mesh.castShadow = b.cast;
    mesh.receiveShadow = b.receive;
    mesh.matrixAutoUpdate = false;
    root.add(mesh);
  }

  return {
    dispose() {
      scene.remove(root);
      for (const m of materials) m.dispose();
      for (const t of textures) t.dispose();
      for (const g of geometries) g.dispose();
      for (const b of batches.values()) for (const g of b.geos) g.dispose();
      root.clear();
    }
  };
}
