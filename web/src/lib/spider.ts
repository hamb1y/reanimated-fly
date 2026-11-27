import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

/**
 * Procedural spider for the fruit-fly demo: a predator in the red/dark palette.
 * Units: millimetres, z up, origin at floor centre, facing +x, footprint about 24 mm.
 * Static body parts are merged per material. Each leg is a chain hip > femur > knee > tibia;
 * a planar two-link IK (knee up) places every foot on the floor, so the gait only has to
 * drive the hip yaw and the foot lift.
 */

const DEG = Math.PI / 180;
const TAU = Math.PI * 2;
const RED = 0xd7261e;
const HIP_Z = 3.8;            // leg mounts, mm above the floor
const FOOT_Z = 0.2;           // tarsus tip radius: feet rest on the floor
const LF = 4.7;               // femur length, mm
const LT = 5.0;               // tibia length, mm
const GAIT_HZ = 1.6;          // stride cycles per second at full walk
const SWING = 0.3;            // hip yaw either side of centre, rad
const LIFT = 2.2;             // max foot lift during swing, mm
const BOB = 0.35;             // body dip per half stride while walking, mm
const THREAT_PERIOD = 7;      // s between display starts
const THREAT_DUR = 1.8;       // s to rise and relax
const THREAT_RISE = 0.6;      // body raise during the display, mm
const BREATH_HZ = 0.3;

type Slot = { side: 1 | -1; x: number; az: number; reach: number };
/**
 * Legs 1-8 (index i = leg i + 1). Odd legs 1,3,5,7 are tetrapod group A and even legs 2,4,6,8 group B.
 * Legs 1 and 2 are the front pair (raised in the threat display). az is the leg direction, 0 = forward.
 */
const SLOTS: Slot[] = [
  { side: 1, x: 3.4, az: 35 * DEG, reach: 8.0 },     // 1: left, front
  { side: -1, x: 3.4, az: -35 * DEG, reach: 8.0 },   // 2: right, front
  { side: -1, x: 2.0, az: -75 * DEG, reach: 8.4 },   // 3: right, second
  { side: 1, x: 2.0, az: 75 * DEG, reach: 8.4 },     // 4: left, second
  { side: 1, x: 0.8, az: 105 * DEG, reach: 8.4 },    // 5: left, third
  { side: -1, x: 0.8, az: -105 * DEG, reach: 8.4 },  // 6: right, third
  { side: -1, x: 0.2, az: -135 * DEG, reach: 8.0 },  // 7: right, back
  { side: 1, x: 0.2, az: 135 * DEG, reach: 8.0 }     // 8: left, back
];

/** Walk phase in [0,1) of leg 1..8 at gait time t (s). Group B runs half a cycle behind group A. */
export function gaitPhase(legIndex: number, t: number): number {
  const x = t * GAIT_HZ + (legIndex % 2 === 0 ? 0.5 : 0);
  return x - Math.floor(x);
}

/** Leg mount offset from the body centre line, on the cephalothorax ellipsoid (3.0 x 2.4 x 1.6 about x=2.6, z=4). */
const hipOffset = (x: number) =>
  0.92 * 2.4 * Math.sqrt(Math.max(0, 1 - ((x - 2.6) / 3) ** 2 - ((HIP_Z - 4) / 1.6) ** 2));

type Leg = { side: 1 | -1; az: number; reach: number; yaw: THREE.Group; femur: THREE.Group; knee: THREE.Group };

/** Planar two-link IK in the leg plane (x = radial, z = up, hip at origin), knee up. Writes femur and knee joint angles. */
function solve(leg: Leg, R: number, zr: number) {
  const D = Math.hypot(R, zr);
  const cosA = Math.min(1, Math.max(-1, (LF * LF + D * D - LT * LT) / (2 * LF * D)));
  const a1 = Math.atan2(zr, R) + Math.acos(cosA);          // femur angle, knee above the hip-foot line
  const a2 = Math.atan2(zr - LF * Math.sin(a1), R - LF * Math.cos(a1));  // tibia angle
  leg.femur.rotation.y = -a1;     // rotation about Y maps +x to (cos a, 0, -sin b): b = -a
  leg.knee.rotation.y = a1 - a2;  // relative knee flex
}

export function createSpider(): { group: THREE.Group; update(dt: number, moving: boolean): void; dispose(): void } {
  const geos: THREE.BufferGeometry[] = [];
  const mats: THREE.Material[] = [];
  const std = (color: number, roughness: number, metalness = 0) => {
    const m = new THREE.MeshStandardMaterial({ color, roughness, metalness });
    mats.push(m);
    return m;
  };
  const track = (g: THREE.BufferGeometry) => { geos.push(g); return g; };
  const ball = (rx: number, ry: number, rz: number, x: number, y: number, z: number) =>
    track(new THREE.SphereGeometry(1, 16, 12).scale(rx, ry, rz).translate(x, y, z));
  /** Cylinder from the origin along +x, tapering from rBase to rTip. */
  const along = (len: number, rBase: number, rTip: number) => {
    const g = new THREE.CylinderGeometry(rTip, rBase, len, 8);
    return track(g.rotateZ(-Math.PI / 2).translate(len / 2, 0, 0));
  };
  const mesh = (g: THREE.BufferGeometry, m: THREE.Material) => {
    const o = new THREE.Mesh(g, m);
    o.castShadow = true;
    return o;
  };

  const bodyMat = std(0x241b18, 0.55);   // cephalothorax, pedicel, hip knobs, chelicerae
  const abdMat = std(0x1a1311, 0.35);    // glossy abdomen
  const markMat = std(RED, 0.4);         // red dorsal marks
  const legMat = std(0x2b211d, 0.5);
  const jointMat = std(0x4a1a15, 0.35);  // dark red-brown knees, pedipalp tips
  const fangMat = std(0x8c1d16, 0.3);
  const eyeMat = std(RED, 0.12);         // glossy eyes

  const group = new THREE.Group();
  const body = new THREE.Group();        // bobs as one: legs, palps and abdomen ride on it
  group.add(body);

  /** Merge static parts that share a material into one mesh (falls back to separate meshes if attributes differ). */
  const addStatic = (parts: THREE.BufferGeometry[], m: THREE.Material) => {
    const merged = mergeGeometries(parts, false);
    if (merged) { track(merged); body.add(mesh(merged, m)); return; }
    for (const p of parts) body.add(mesh(p, m));
  };

  const sides = [1, -1] as const;
  addStatic([
    ball(3.0, 2.4, 1.6, 2.6, 0, 4.0),        // cephalothorax
    ball(0.9, 0.8, 0.8, -0.4, 0, 4.0),       // pedicel, joins the abdomen
    ...sides.map((s) => ball(1.0, 0.85, 1.0, 4.9, s * 0.5, 3.4)),   // chelicerae
    ...SLOTS.map((s) => ball(0.8, 0.8, 0.8, s.x, s.side * hipOffset(s.x), HIP_Z))  // leg hip knobs
  ], bodyMat);

  const fangDir = new THREE.Vector3(0.35, 0, -1).normalize();
  const fangQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), fangDir);
  addStatic(sides.map((s) => track(new THREE.ConeGeometry(0.32, 1.7, 8)).applyQuaternion(fangQ).translate(5.0, s * 0.45, 2.9)), fangMat);

  addStatic(sides.flatMap((s) => [
    ball(0.42, 0.42, 0.42, 5.2, s * 0.9, 4.6),   // large anterior eyes
    ball(0.3, 0.3, 0.3, 4.5, s * 1.35, 4.8)      // small lateral eyes
  ]), eyeMat);

  // Abdomen: animated (breathing), pivot at its centre. Red marks ride on it.
  const abdomen = new THREE.Group();
  abdomen.position.set(-4.1, 0, 4.1);
  abdomen.add(mesh(track(new THREE.SphereGeometry(1, 20, 14).scale(4.1, 3.3, 3.2)), abdMat));
  const marks = mergeGeometries([ball(0.7, 1.1, 0.22, 0.9, 0, 2.95), ball(0.7, 1.0, 0.22, -1.0, 0, 2.9)], false);
  if (marks) { track(marks); abdomen.add(mesh(marks, markMat)); }
  body.add(abdomen);

  // Pedipalps: two-link chain (flare about vertical, then pitch forward-down). Flared during the threat display.
  const palpGeo = along(3.0, 0.3, 0.2);
  const palpTip = ball(0.45, 0.4, 0.4, 3.0, 0, 0);
  const palps = sides.map((s) => {
    const mount = new THREE.Group();
    mount.position.set(4.6, s * 1.0, 3.0);
    const flare = new THREE.Group();
    const pitch = new THREE.Group();
    pitch.rotation.y = 0.55;
    pitch.add(mesh(palpGeo, legMat), mesh(palpTip, jointMat));
    flare.add(pitch);
    mount.add(flare);
    body.add(mount);
    return { side: s, flare, pitch };
  });

  // Legs: shared segment geometry, one chain per leg.
  const femurGeo = along(LF, 0.42, 0.32);
  const tibiaGeo = along(LT, 0.32, 0.18);
  const kneeGeo = ball(0.4, 0.4, 0.4, 0, 0, 0);
  const tipGeo = ball(0.22, 0.22, 0.22, LT, 0, 0);
  const legs: Leg[] = SLOTS.map((s) => {
    const mount = new THREE.Group();
    mount.position.set(s.x, s.side * hipOffset(s.x), HIP_Z);
    mount.rotation.z = s.az;
    const yaw = new THREE.Group();     // swings the whole leg plane about the vertical hip axis
    const femur = new THREE.Group();   // femur hinge at the hip
    const knee = new THREE.Group();    // knee hinge at the femur tip
    knee.position.set(LF, 0, 0);
    knee.add(mesh(kneeGeo, jointMat), mesh(tibiaGeo, legMat), mesh(tipGeo, jointMat));
    femur.add(mesh(femurGeo, legMat), knee);
    yaw.add(femur);
    mount.add(yaw);
    body.add(mount);
    return { side: s.side, az: s.az, reach: s.reach, yaw, femur, knee };
  });

  // Clock and blend state: all motion is a function of these plus dt.
  let clock = 0;      // wall time of the animation, s
  let gait = 0;      // gait time, advances only while walking (eased in and out)
  let walk = 0;      // 0 standing .. 1 walking

  const update = (dt: number, moving: boolean) => {
    const h = Math.min(0.25, Math.max(0, dt));
    clock += h;
    walk += ((moving ? 1 : 0) - walk) * (1 - Math.exp(-6 * h));
    gait += h * walk;
    const still = 1 - walk;

    // Threat display: a smooth bump every THREAT_PERIOD seconds, only while standing.
    const local = clock % THREAT_PERIOD;
    const thr = local < THREAT_DUR ? Math.sin((Math.PI * local) / THREAT_DUR) ** 2 * still : 0;

    // Body: dip twice per stride while walking; rise during the threat display.
    const cyc = gait * GAIT_HZ;
    const bob = walk * BOB * (1 - Math.cos(TAU * 2 * cyc)) / 2;
    const bodyZ = bob + THREAT_RISE * thr;
    body.position.z = bodyZ;

    // Abdomen breathes slowly (idle).
    const breath = Math.sin(TAU * BREATH_HZ * clock);
    abdomen.position.z = 4.1 + 0.12 * breath;
    abdomen.scale.set(1 + 0.012 * breath, 1 + 0.012 * breath, 1 + 0.03 * breath);

    for (let i = 0; i < 8; i++) {
      const leg = legs[i];
      const n = i + 1;
      const p = gaitPhase(n, gait);
      // Swing (p < 0.5): foot goes back to front and lifts. Stance: foot slides front to back on the floor.
      let yawAmt: number, lift: number;
      if (p < 0.5) {
        const q = 2 * p;
        yawAmt = -SWING + SWING * (1 - Math.cos(Math.PI * q));
        lift = LIFT * Math.sin(Math.PI * q) ** 2;
      } else {
        yawAmt = SWING - 2 * SWING * (2 * p - 1);
        lift = 0;
      }
      yawAmt *= walk;
      lift *= walk;

      // Idle: tiny, non-synchronised shifts of reach and yaw.
      const idleR = 0.12 * still * Math.sin(TAU * 0.17 * clock + n * 2.3);
      const idleY = 0.03 * still * Math.sin(TAU * 0.21 * clock + n * 1.7);

      let R = leg.reach + idleR;
      let zr = FOOT_Z - (HIP_Z + bodyZ) + lift;   // foot height relative to the hip
      // Front pair rises in the threat display.
      let w = 0;
      if (i < 2) { w = thr; R += (4.2 - R) * w; zr += (2.2 - zr) * w; }
      leg.yaw.rotation.z = -leg.side * yawAmt * (1 - w) + idleY;
      solve(leg, R, zr);
    }

    for (const pp of palps) {
      pp.flare.rotation.z = pp.side * (0.25 + 0.7 * thr + 0.03 * breath);
      pp.pitch.rotation.y = 0.55 - 0.25 * thr;
    }
  };

  const dispose = () => {
    for (const g of geos) g.dispose();
    for (const m of mats) m.dispose();
  };

  update(0, false);   // initial rest pose
  return { group, update, dispose };
}
