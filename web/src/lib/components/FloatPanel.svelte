<script lang="ts">
  import type { Snippet } from 'svelte';
  import { draggable } from '$lib/drag';

  let { title, x, y, w = 280, z = 4, onclose, children }: {
    title: string; x: number; y: number; w?: number; z?: number; onclose?: () => void; children: Snippet;
  } = $props();
</script>

<div class="fp" style:left={x + 'px'} style:top={y + 'px'} style:width={w + 'px'} style:z-index={z} use:draggable={{ handle: '.grip' }}>
  <div class="grip">
    <span>{title}</span>
    {#if onclose}<button class="close" onclick={onclose} aria-label="Close panel">Close</button>{/if}
  </div>
  <div class="body">{@render children()}</div>
</div>

<style>
  .fp { position: absolute; background: #000; border: 1px solid #fff; border-radius: 14px; color: #fff; box-sizing: border-box; max-width: calc(100% - 28px); }
  .grip { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 8px 10px; border-bottom: 1px solid #b3261e; font-weight: 600; font-size: 13px; cursor: move; user-select: none; touch-action: none; }
  .body { padding: 10px; }
  .close { padding: 2px 8px; font-size: 12px; }
</style>
