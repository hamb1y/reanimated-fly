// Normalisers: the UI is built from whatever /params and /groups return (real brain shapes accepted).
export type ParamSpec = { key: string; value: number; min: number; max: number; step: number; readonly: boolean; unit?: string; desc?: string };
export type GroupSpec = { name: string; size?: number; kind?: string; desc?: string; rate?: number };

const KNOWN: Record<string, [number, number]> = {
  w_syn: [0, 2], tau_m_ms: [1, 60], v_th_mV: [-70, -20], v_reset_mV: [-90, -30],
  tau_syn_ms: [0.5, 30], t_ref_ms: [0, 10], poisson_rate_hz: [0, 200]
};
const niceStep = (range: number) => { const r = range / 400; const p = 10 ** Math.floor(Math.log10(r)); return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= r) ?? p; };

/** Accepts {key: number}, {key: {value,min,max,step,readonly,unit,desc}}, optionally nested under "params". */
export function parseParams(raw: any): ParamSpec[] {
  const src = raw && typeof raw.params === 'object' && !Array.isArray(raw.params) ? raw.params : raw;
  const meta = raw?.meta ?? raw?.ranges ?? {};
  const out: ParamSpec[] = [];
  for (const [key, v] of Object.entries<any>(src ?? {})) {
    let value: number, min: number | undefined, max: number | undefined, step: number | undefined, ro = false;
    const m = meta[key] ?? {};
    if (typeof v === 'number') value = v;
    else if (v && typeof v === 'object' && typeof v.value === 'number') ({ value, min, max, step } = v), (ro = !!(v.readonly ?? v.read_only));
    else continue;
    min ??= m.min; max ??= m.max; step ??= m.step; ro ||= !!(m.readonly ?? m.read_only);
    if (key.startsWith('dt_')) ro = true;
    if (min === undefined || max === undefined) {
      const k = KNOWN[key];
      if (k) [min, max] = k;
      else if (value >= 0) [min, max] = [0, Math.max(1, Math.abs(value) * 4)];
      else [min, max] = [value * 2, Math.min(0, value / 2)];
    }
    min = Math.min(min!, value); max = Math.max(max!, value);
    step ??= niceStep(max - min);
    out.push({ key, value, min, max, step, readonly: ro, unit: v?.unit ?? m.unit, desc: v?.desc ?? m.desc });
  }
  return out;
}

/** Accepts the real brain shape {name: {size, role, default_rate_hz, description}} and simpler lists. */
export function parseGroups(raw: any): GroupSpec[] {
  const src = Array.isArray(raw) ? raw : Array.isArray(raw?.groups) ? raw.groups : raw?.groups ?? raw;
  if (Array.isArray(src)) return src.map((g: any) => (typeof g === 'string' ? { name: g } : { ...g, name: g.name ?? g.group ?? String(g.id) }));
  return Object.entries<any>(src ?? {}).map(([name, g]) => ({
    name,
    size: typeof g === 'number' ? g : Array.isArray(g) ? g.length : (g?.size ?? g?.n ?? g?.count),
    kind: typeof g === 'object' && !Array.isArray(g) ? (g?.role ?? g?.kind) : undefined,
    desc: g?.description ?? g?.desc,
    rate: g?.default_rate_hz
  }));
}

export async function j(path: string, init?: RequestInit) {
  const r = await fetch(path, init);
  if (!r.ok) throw new Error(path + ' ' + r.status);
  return r.json();
}
const jsonInit = (method: string, body?: unknown): RequestInit =>
  body === undefined ? { method } : { method, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) };
export const post = (path: string, body: unknown = {}) => j(path, jsonInit('POST', body));
export const put = (path: string, body: unknown) => j(path, jsonInit('PUT', body));
export const del = (path: string) => j(path, jsonInit('DELETE'));

/** Auto-reconnecting WebSocket to /ws/<svc>. */
export function openStream(svc: 'brain' | 'physics', onMsg: (m: any) => void, onState: (open: boolean) => void) {
  let ws: WebSocket | undefined, timer: any, dead = false, delay = 500;
  const connect = () => {
    ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/' + svc);
    ws.onopen = () => { delay = 500; onState(true); };
    ws.onmessage = (e) => { try { onMsg(JSON.parse(e.data)); } catch {} };
    ws.onclose = () => { onState(false); if (!dead) timer = setTimeout(connect, (delay = Math.min(delay * 1.6, 5000))); };
    ws.onerror = () => ws?.close();
  };
  connect();
  return () => { dead = true; clearTimeout(timer); ws?.close(); };
}
