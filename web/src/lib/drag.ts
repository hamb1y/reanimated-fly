// Svelte action: makes a floating panel movable by its handle. Positions are left/top inside the offsetParent.
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function draggable(node: HTMLElement, opts: { handle?: string } = {}) {
  const handle = ((opts.handle && node.querySelector(opts.handle)) as HTMLElement | null) ?? node;
  let d: { sx: number; sy: number; ox: number; oy: number } | null = null;
  const down = (e: PointerEvent) => {
    if ((e.target as HTMLElement).closest('button,input,select,textarea,a')) return;
    d = { sx: e.clientX, sy: e.clientY, ox: node.offsetLeft, oy: node.offsetTop };
    handle.setPointerCapture(e.pointerId);
    e.preventDefault();
  };
  const move = (e: PointerEvent) => {
    if (!d) return;
    const p = node.offsetParent as HTMLElement | null;
    const maxX = Math.max(0, (p?.clientWidth ?? 0) - node.offsetWidth);
    const maxY = Math.max(0, (p?.clientHeight ?? 0) - node.offsetHeight);
    node.style.left = clamp(d.ox + e.clientX - d.sx, 0, maxX) + 'px';
    node.style.top = clamp(d.oy + e.clientY - d.sy, 0, maxY) + 'px';
  };
  const up = () => (d = null);
  handle.addEventListener('pointerdown', down);
  handle.addEventListener('pointermove', move);
  handle.addEventListener('pointerup', up);
  handle.addEventListener('pointercancel', up);
  return {
    destroy() {
      handle.removeEventListener('pointerdown', down);
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      handle.removeEventListener('pointercancel', up);
    }
  };
}
