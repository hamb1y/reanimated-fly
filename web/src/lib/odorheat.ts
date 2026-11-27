// Pure odour -> heat-colour mapping (no three.js, so it is unit-testable).
// Data is REAL: the concentration grid served by physics GET /odor. Only the colour ramp is authored.
export type OdorSpecies = 'food' | 'cva';
export type OdorGrid = {
  species?: string; x0_mm: number; y0_mm: number; cell_mm: number; nx: number; ny: number;
  c: number[]; max: number; wind?: number[]; antennae?: { left: number; right: number };
};

export const ODOR_FLOOR = 1e-3;   // concentrations below this are drawn fully transparent
export const ODOR_TOP_MIN = 0.05; // the colour scale never zooms in below this ceiling (keeps faint noise dim)

/** Log-scaled 0..1: 0 at ODOR_FLOOR, 1 at max(grid max, ODOR_TOP_MIN). */
export function logLevel(v: number, max: number): number {
  const top = Math.max(max, ODOR_TOP_MIN);
  if (!(v > ODOR_FLOOR)) return 0;
  return Math.min(1, Math.log(v / ODOR_FLOOR) / Math.log(top / ODOR_FLOOR));
}

const RAMPS: Record<OdorSpecies, [number, number, number][]> = {
  food: [[110, 0, 60], [215, 10, 40], [255, 90, 20], [255, 235, 140]],   // crimson -> red -> orange -> pale yellow (reads on the oak counter)
  cva: [[40, 20, 120], [90, 70, 220], [90, 170, 240], [215, 245, 255]]    // indigo -> blue -> cyan -> near white
};

/** RGBA bytes for one cell. Alpha rises with level so low concentration is nearly invisible. */
export function heatRGBA(t: number, species: OdorSpecies): [number, number, number, number] {
  if (t <= 0) return [0, 0, 0, 0];
  const r = RAMPS[species] ?? RAMPS.food, f = Math.min(1, t) * (r.length - 1), i = Math.min(r.length - 2, Math.floor(f)), a = f - i;
  const m = (k: number) => Math.round(r[i][k] + (r[i + 1][k] - r[i][k]) * a);
  return [m(0), m(1), m(2), Math.round(255 * Math.min(0.85, 0.2 + 0.65 * t))];
}

/** Fill an nx*ny RGBA byte buffer; row j (y) is written to texture row j, so texture v = y (bottom-up, matching three's flipY=false data textures). */
export function fillHeat(g: OdorGrid, species: OdorSpecies, out: Uint8Array): void {
  const n = g.nx * g.ny;
  for (let k = 0; k < n; k++) {
    const [r, gg, b, a] = heatRGBA(logLevel(g.c[k] ?? 0, g.max), species);
    out[k * 4] = r; out[k * 4 + 1] = gg; out[k * 4 + 2] = b; out[k * 4 + 3] = a;
  }
}

export const validGrid = (g: any): g is OdorGrid =>
  !!g && Number.isFinite(g.nx) && Number.isFinite(g.ny) && g.nx > 1 && g.ny > 1 && Array.isArray(g.c) && g.c.length === g.nx * g.ny && g.cell_mm > 0;
