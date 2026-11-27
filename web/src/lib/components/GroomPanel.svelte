<script lang="ts">
  import { onMount } from 'svelte';
  import FloatPanel from './FloatPanel.svelte';

  let { x = 14, y = 540, h = $bindable(0) }: { x?: number; y?: number; h?: number } = $props();
  let g = $state<any>(null);
  let err = $state('');

  async function call(path: string, body?: unknown) {
    try {
      const r = await fetch('/api/physics' + path, body === undefined ? undefined : { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
      g = await r.json(); err = '';
    } catch (e) { err = String(e); }
  }
  onMount(() => { call('/groom'); const t = setInterval(() => call('/groom'), 700); return () => clearInterval(t); });

  const dusty = $derived((g?.dust_hz ?? 0) > 0);
  const manual = $derived(!!g?.manual);
</script>

<FloatPanel title="Dust and grooming" {x} {y} w={236} z={4} bind:h>
  <div class="groom-row">
    <button class:on={dusty} onclick={() => call('/dust', { on: !dusty })}>{dusty ? 'Stop dust' : 'Dust her antennae'}</button>
    <button class:on={manual} onclick={() => call('/groom', { on: !manual })}>{manual ? 'Stop grooming' : 'Groom now'}</button>
  </div>
  {#if err}<p class="groom-note">{err}</p>{/if}
</FloatPanel>
