import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

/**
 * A procedural, appetising food morsel for a fruit fly: a ripe cut wedge of
 * apple — glossy red skin on the outside, pale cream flesh with a bruised,
 * oxidised edge on the two cut faces — sitting on a smear of syrup with a
 * small highlight droplet on top.
 *
 * Units: millimetres, z up, the morsel sits on the floor (z = 0) and is about
 * 5 mm across. No assets are used: the flesh and skin colour variation is
 * painted into two tiny CanvasTextures. Triangle count stays well under 2000.
 *
 * `setAmount(a)` (a in 0..1) shrinks the morsel and its syrup as the fly eats:
 * the scale never drops below 0.3 and the object hides entirely at a = 0.
 * `dispose()` frees every geometry, material and texture the object created.
 */

const TAU = Math.PI * 2;

// Deterministic PRNG (the same trick kitchen.ts uses) so every morsel spawns identically.
const rnd = (() => {
  let s = 0x51ab2b7d;
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
})();

// Wedge shape, in mm. The wedge is a flat-bottomed slice of a sphere of radius R,
// bounded by the two cut faces at azimuth +-phi0 and the bottom plane z = zBottom.
const R = 2.6; // sphere radius
const PHI0 = 0.62; // half the wedge angle, rad
const THETA_MAX = 1.2; // polar angle down to the flat bottom, rad
const Z_BOTTOM = R * Math.cos(THETA_MAX); // bottom plane, lifted to z = 0 below
const RHO_MAX = R * Math.sin(THETA_MAX); // radius of the bottom fan

export function createFood(): { group: THREE.Group; setAmount(a: number): void; dispose(): void } {
  const geos: THREE.BufferGeometry[] = [];
  const mats: THREE.Material[] = [];
  const texs: THREE.Texture[] = [];

  // ---------------------------------------------------------------- textures
  const canvas = (w: number, h: number, draw: (g: CanvasRenderingContext2D) => void): THREE.CanvasTexture => {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    draw(c.getContext('2d')!);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    texs.push(t);
    return t;
  };

  // Apple flesh: pale cream at the pit (u=0), ripe peach toward the skin (u=1),
  // deeper at the bottom (v=1), with juice mottling and a bruised oxidised edge.
  const fleshTex = canvas(128, 128, (g) => {
    const grad = g.createLinearGradient(0, 0, 0, 128);
    grad.addColorStop(0, '#fff4dc');
    grad.addColorStop(0.5, '#ffe0b0');
    grad.addColorStop(1, '#ffc184');
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
    const core = g.createRadialGradient(0, 64, 4, 0, 64, 122);
    core.addColorStop(0, 'rgba(255,255,238,0.9)');
    core.addColorStop(0.45, 'rgba(255,255,238,0)');
    core.addColorStop(1, 'rgba(232,120,54,0.28)');
    g.fillStyle = core;
    g.fillRect(0, 0, 128, 128);
    for (let i = 0; i < 260; i++) {
      g.fillStyle = `rgba(250,196,132,${(0.04 + rnd() * 0.09).toFixed(3)})`;
      g.beginPath();
      g.arc(rnd() * 128, rnd() * 128, 0.6 + rnd() * 2.4, 0, TAU);
      g.fill();
    }
    for (let i = 0; i < 6; i++) {
      const bx = 20 + rnd() * 88;
      const by = 112 - rnd() * 16;
      const r = 10 + rnd() * 12;
      const b = g.createRadialGradient(bx, by, 1, bx, by, r);
      b.addColorStop(0, 'rgba(150,92,44,0.5)');
      b.addColorStop(0.6, 'rgba(150,92,44,0.18)');
      b.addColorStop(1, 'rgba(150,92,44,0)');
      g.fillStyle = b;
      g.beginPath();
      g.arc(bx, by, r, 0, TAU);
      g.fill();
    }
  });

  // Apple skin: glossy red with blush streaks, pale lenticels and a top sheen.
  const skinTex = canvas(128, 64, (g) => {
    g.fillStyle = '#cf3a2b';
    g.fillRect(0, 0, 128, 64);
    for (let i = 0; i < 90; i++) {
      g.fillStyle = `rgba(${(196 + rnd() * 42) | 0},${(38 + rnd() * 28) | 0},${(26 + rnd() * 20) | 0},${(0.06 + rnd() * 0.12).toFixed(3)})`;
      g.beginPath();
      g.ellipse(rnd() * 128, rnd() * 64, 3 + rnd() * 10, 1 + rnd() * 3, rnd() * TAU, 0, TAU);
      g.fill();
    }
    for (let i = 0; i < 180; i++) {
      g.fillStyle = `rgba(250,220,170,${(0.06 + rnd() * 0.1).toFixed(3)})`;
      g.fillRect(rnd() * 128, rnd() * 64, 0.8 + rnd() * 1.3, 0.8 + rnd() * 1.3);
    }
    const hi = g.createLinearGradient(0, 0, 0, 32);
    hi.addColorStop(0, 'rgba(255,255,255,0.16)');
    hi.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = hi;
    g.fillRect(0, 0, 128, 32);
  });

  // ---------------------------------------------------------------- materials
  const fleshMat = new THREE.MeshStandardMaterial({
    map: fleshTex,
    color: 0xffffff,
    roughness: 0.34,
    metalness: 0,
    envMapIntensity: 0.5,
    side: THREE.DoubleSide,
  });
  const skinMat = new THREE.MeshStandardMaterial({
    map: skinTex,
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0,
    envMapIntensity: 0.8,
    side: THREE.DoubleSide,
  });
  const syrupMat = new THREE.MeshStandardMaterial({
    color: 0xd9882c,
    roughness: 0.16,
    metalness: 0,
    transparent: true,
    opacity: 0.68,
    envMapIntensity: 1.4,
  });
  const glintMat = new THREE.MeshStandardMaterial({
    color: 0xfffbf2,
    roughness: 0.05,
    metalness: 0,
    transparent: true,
    opacity: 0.9,
    envMapIntensity: 2.2,
  });
  mats.push(fleshMat, skinMat, syrupMat, glintMat);

  // ---------------------------------------------------------------- geometry
  const buildGeometry = (
    positions: number[],
    normals: number[],
    uvs: number[],
    indices: number[]
  ): THREE.BufferGeometry => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    g.setIndex(indices);
    return g;
  };

  // Curved outer surface of the wedge (the skin), a band of the sphere r = R.
  const buildSkin = (): THREE.BufferGeometry => {
    const nPhi = 8;
    const nTheta = 6;
    const pos: number[] = [];
    const nor: number[] = [];
    const uv: number[] = [];
    const idx: number[] = [];
    for (let j = 0; j <= nTheta; j++) {
      const th = (j / nTheta) * THETA_MAX;
      const st = Math.sin(th);
      const ct = Math.cos(th);
      for (let i = 0; i <= nPhi; i++) {
        const ph = -PHI0 + (i / nPhi) * 2 * PHI0;
        const x = R * st * Math.cos(ph);
        const y = R * st * Math.sin(ph);
        const z = R * ct;
        pos.push(x, y, z - Z_BOTTOM);
        const inv = 1 / Math.hypot(x, y, z);
        nor.push(x * inv, y * inv, z * inv);
        uv.push(i / nPhi, j / nTheta);
      }
    }
    for (let j = 0; j < nTheta; j++) {
      for (let i = 0; i < nPhi; i++) {
        const a = j * (nPhi + 1) + i;
        const b = a + nPhi + 1;
        const c = b + 1;
        const d = a + 1;
        idx.push(a, b, d, b, c, d);
      }
    }
    return buildGeometry(pos, nor, uv, idx);
  };

  // One flat cut face (flesh): a plane through the z axis at azimuth +-phi0,
  // bounded below by the bottom plane and above by the sphere.
  const buildSide = (side: 1 | -1): THREE.BufferGeometry => {
    const nR = 4;
    const nZ = 5;
    const pos: number[] = [];
    const nor: number[] = [];
    const uv: number[] = [];
    const idx: number[] = [];
    const cx = Math.cos(PHI0);
    const sy = Math.sin(PHI0);
    for (let j = 0; j <= nZ; j++) {
      const t = j / nZ;
      for (let i = 0; i <= nR; i++) {
        const rho = (i / nR) * RHO_MAX;
        const zTop = Math.sqrt(Math.max(0, R * R - rho * rho));
        const z = Z_BOTTOM + t * (zTop - Z_BOTTOM);
        pos.push(rho * cx, side * rho * sy, z - Z_BOTTOM);
        nor.push(-sy, side * cx, 0);
        uv.push(i / nR, 1 - t);
      }
    }
    for (let j = 0; j < nZ; j++) {
      for (let i = 0; i < nR; i++) {
        const a = j * (nR + 1) + i;
        const b = a + nR + 1;
        const c = b + 1;
        const d = a + 1;
        if (side > 0) idx.push(a, b, d, b, c, d);
        else idx.push(a, d, b, b, d, c);
      }
    }
    return buildGeometry(pos, nor, uv, idx);
  };

  // Flat bottom fan (flesh, hidden against the floor), closing the wedge.
  const buildBottom = (): THREE.BufferGeometry => {
    const nR = 4;
    const nPhi = 8;
    const pos: number[] = [];
    const nor: number[] = [];
    const uv: number[] = [];
    const idx: number[] = [];
    for (let i = 0; i <= nR; i++) {
      const rho = (i / nR) * RHO_MAX;
      for (let j = 0; j <= nPhi; j++) {
        const ph = -PHI0 + (j / nPhi) * 2 * PHI0;
        pos.push(rho * Math.cos(ph), rho * Math.sin(ph), 0);
        nor.push(0, 0, -1);
        uv.push(i / nR, j / nPhi);
      }
    }
    for (let i = 0; i < nR; i++) {
      for (let j = 0; j < nPhi; j++) {
        const a = i * (nPhi + 1) + j;
        const b = a + nPhi + 1;
        const c = b + 1;
        const d = a + 1;
        idx.push(a, d, b, b, d, c);
      }
    }
    return buildGeometry(pos, nor, uv, idx);
  };

  const group = new THREE.Group();
  group.name = 'food';

  const mesh = (g: THREE.BufferGeometry, m: THREE.Material, name: string): THREE.Mesh => {
    const o = new THREE.Mesh(g, m);
    o.name = name;
    o.castShadow = true;
    return o;
  };

  // Flesh: merge the two cut faces and the bottom into one draw call.
  const sideL = buildSide(1);
  const sideR = buildSide(-1);
  const bottom = buildBottom();
  const fleshGeo = mergeGeometries([sideL, sideR, bottom], false);
  if (!fleshGeo) throw new Error('food: failed to merge flesh geometry');
  geos.push(fleshGeo);
  sideL.dispose();
  sideR.dispose();
  bottom.dispose();
  group.add(mesh(fleshGeo, fleshMat, 'food-flesh'));

  // Skin: the curved outer band.
  const skinGeo = buildSkin();
  geos.push(skinGeo);
  group.add(mesh(skinGeo, skinMat, 'food-skin'));

  // Syrup smear: a soft translucent puddle under the wedge with two oozed droplets.
  const blobGeo = new THREE.SphereGeometry(1, 10, 6);
  geos.push(blobGeo);
  const syrup = (x: number, y: number, z: number, sx: number, sy: number, sz: number) => {
    const o = mesh(blobGeo, syrupMat, 'food-syrup');
    o.position.set(x, y, z);
    o.scale.set(sx, sy, sz);
    o.receiveShadow = true;
    group.add(o);
  };
  syrup(0.3, -0.1, 0.06, 2.1, 1.7, 0.16);
  syrup(1.5, 0.9, 0.04, 0.85, 0.6, 0.11);
  syrup(-1.4, -0.7, 0.03, 0.5, 0.4, 0.09);

  // A tiny clear highlight droplet on the skin, catching the light.
  const glintGeo = new THREE.SphereGeometry(1, 8, 6);
  geos.push(glintGeo);
  const glint = mesh(glintGeo, glintMat, 'food-glint');
  glint.position.set(0.85, 0.26, 1.5);
  glint.scale.set(0.34, 0.34, 0.28);
  group.add(glint);

  // ---------------------------------------------------------------- amount
  const setAmount = (a: number) => {
    const amt = Math.max(0, Math.min(1, a));
    if (amt < 1e-3) {
      group.visible = false;
      return;
    }
    group.visible = true;
    // Never smaller than 0.3 of full size; the bottom stays on the floor (z = 0).
    group.scale.setScalar(0.3 + 0.7 * amt);
  };

  const dispose = () => {
    for (const g of geos) g.dispose();
    for (const m of mats) m.dispose();
    for (const t of texs) t.dispose();
    group.clear();
  };

  setAmount(1);
  return { group, setAmount, dispose };
}
