import type { WorldObj } from './world';

export type Pers = { hunger: number; fear: number; lust: number; receptivity: number };
export const DEFAULT_PERSONALITY: Pers = { hunger: 0.8, fear: 0.9, lust: 0.9, receptivity: 0.9 };
export const PRESETS: Record<string, Pers> = {
  Coward: { hunger: 0.5, fear: 0.95, lust: 0.4, receptivity: 0.5 },
  Starving: { hunger: 1, fear: 0.5, lust: 0.3, receptivity: 0.6 },
  Lovesick: { hunger: 0.5, fear: 0.4, lust: 1, receptivity: 0.95 },
  Chill: { hunger: 0.2, fear: 0.2, lust: 0.3, receptivity: 0.6 }
};
export const DIALS: { key: keyof Pers; label: string; tech: string; low: string; high: string }[] = [
  { key: 'hunger', label: 'How hungry', tech: 'hunger', low: 'full', high: 'starving' },
  { key: 'fear', label: 'How scared', tech: 'fear', low: 'brave', high: 'terrified' },
  { key: 'lust', label: 'How lovesick', tech: 'lust', low: 'not interested', high: 'lovesick' },
  { key: 'receptivity', label: 'How receptive', tech: 'receptivity', low: 'walks away', high: 'stays close' }
];
export const KIND_NAME: Record<string, string> = { food: 'Food', mate: 'Mate', scary: 'Scary thing' };

const side = (b: number) => (Math.abs(b) < 20 ? 'straight ahead' : b > 0 ? 'to the left' : 'to the right');

/** Plain-language status from the physics stream's state block (mode, nearest, hunger_level). */
export function describeState(st: any, _objects?: WorldObj[]): { title: string; detail: string; alert: boolean } {
  if (!st) return { title: 'Waiting for the body', detail: 'No world data yet.', alert: false };
  const n = st.nearest;
  const near = n ? (KIND_NAME[n.kind] ?? n.kind) + ', ' + Math.round(n.dist) + ' mm away, ' + side(n.bearing_deg) : 'Nothing close by';
  const hungry = (st.hunger_level ?? 0) >= 0.35;
  switch (st.mode) {
    case 'eat': return { title: 'Eating', detail: near, alert: false };
    case 'chase': return n?.kind === 'mate'
      ? { title: 'Chasing a mate', detail: near, alert: false }
      : { title: 'Looking for food', detail: near, alert: false };
    case 'flee': return { title: 'Running away!', detail: near, alert: true };
    case 'explore': return hungry ? { title: 'Looking for food', detail: near, alert: false } : { title: 'Wandering', detail: near, alert: false };
    case 'idle': return n?.kind === 'mate'
      ? { title: 'Staying close to a mate', detail: near, alert: false }
      : { title: 'Standing still', detail: near, alert: false };
    default: return { title: String(st.mode ?? 'unknown'), detail: near, alert: false };
  }
}

export type Sample = {
  wall: number; sim: number; speed: number; mode: string; hunger: number | null;
  cf: number; ct: number; mf: number; mt: number; x: number; y: number; z: number;
  nk: string; nmm: number | null; nb: number | null; p: Pers; drives: Record<string, number>; rates: Record<string, number>;
};

/** Run log as CSV: time, command, position, drives (physics) and brain group rates. */
export function toCsv(rows: Sample[]): string {
  const gset = new Set<string>();
  rows.forEach((r) => Object.keys(r.rates).forEach((k) => gset.add(k)));
  const gcols = [...gset].sort();
  const head = ['wall_s', 'sim_t_s', 'speed_factor', 'mode', 'hunger_level', 'command_forward', 'command_turn',
    'motor_forward', 'motor_turn', 'fly_x_mm', 'fly_y_mm', 'fly_z_mm', 'nearest_kind', 'nearest_mm', 'nearest_bearing_deg',
    'personality_hunger', 'personality_fear', 'personality_lust', 'personality_receptivity', 'drives']
    .concat(gcols.map((g) => 'rate_hz_' + g));
  const f = (v: number | null) => (v == null || !isFinite(v) ? '' : String(Math.round(v * 1000) / 1000));
  const lines = rows.map((r) => {
    const drives = Object.entries(r.drives).map(([k, v]) => k + '=' + f(v)).join(';');
    return [f(r.wall), f(r.sim), f(r.speed), r.mode, f(r.hunger), f(r.cf), f(r.ct), f(r.mf), f(r.mt),
      f(r.x), f(r.y), f(r.z), r.nk, f(r.nmm), f(r.nb), f(r.p.hunger), f(r.p.fear), f(r.p.lust), f(r.p.receptivity),
      '"' + drives + '"']
      .concat(gcols.map((g) => f(r.rates[g] ?? null))).join(',');
  });
  return [head.join(',')].concat(lines).join('\n') + '\n';
}
