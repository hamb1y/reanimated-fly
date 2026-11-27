<script lang="ts">
  import type { Snippet } from 'svelte';
  import { draggable } from '$lib/drag';

  let { title, x, y, w = 280, z = 4, h = $bindable(0), onclose, children }: {
    title: string; x: number; y: number; w?: number; z?: number; h?: number; onclose?: () => void; children: Snippet;
  } = $props();
</script>

<div class="panel" bind:offsetHeight={h} style:left={x + 'px'} style:top={y + 'px'} style:width={w + 'px'} style:z-index={z} use:draggable={{ handle: '.grip' }}>
  <div class="panel-grip">
    <span class="panel-title">{title}</span>
    {#if onclose}<button class="panel-close" onclick={onclose} aria-label="Close panel">Close</button>{/if}
  </div>
  <div class="panel-body">{@render children()}</div>
</div>
