# brain (port 5031)

Whole-brain leaky integrate-and-fire simulation of the FlyWire connectome (Shiu et al.).
Contract: /SPEC.md, section "brain". Reference numerics: reference/run_pytorch.py.

The brain runs on CUDA when a GPU is visible (Triton kernels in a CUDA graph) and on CPU otherwise (torch eager).
One image serves both; the choice is made at start-up (BRAIN_DEVICE).

## Run

Images (build from the repo root; docker build needs --network host on this host):

    docker build --network host -t flybrain-brain brain/
    docker build --network host --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t flybrain-brain:cpu brain/

- default image: CUDA torch wheel (cu128), about 7.6 GB because of the nvidia-* libraries. Runs on CPU too.
- ":cpu" tag: CPU-only torch, about 1.5 GB. Same code, no GPU path.

Start (host networking; published ports do not work on this host):

    docker run --rm --network host -v $PWD/data:/data:ro flybrain-brain                  # CPU host
    docker run --rm --network host --gpus all -v $PWD/data:/data:ro flybrain-brain       # GPU host, needs nvidia-container-toolkit

Tests (in container, need the data mounted; add --gpus all to run the CUDA tests):

    docker run --rm --network host --gpus all -v $PWD/data:/data:ro --entrypoint python flybrain-brain -m pytest -q

Env:

| var | default | meaning |
|---|---|---|
| BRAIN_DEVICE | auto | auto: CUDA if torch sees a GPU and the kernels start, otherwise CPU (logged). cuda: error if no GPU. cpu |
| BRAIN_GPU_MEM_MIB | 1024 | torch allocator cap on the GPU for this process. The CUDA context is extra (about 340 MiB). |
| DATA_DIR | /data | connectome (read only) |
| CACHE_DIR | /cache | CSC arrays cached here, Triton kernel cache in $CACHE_DIR/triton |
| TORCH_THREADS | 8 | CPU threads (CPU backend only) |
| BRAIN_SPEED_TARGET | 1.0 | real-time factor cap, 1.0 = real time, 0 = flat out (tests use 0) |

## GPU in docker-compose (for the orchestrator)

Merge into the brain service (keep the existing build, network_mode, command, volumes, healthcheck keys).
Needs nvidia-container-toolkit on the host. On a machine without a GPU, leave out the deploy block; the image then runs on CPU.

```yaml
  brain:
    environment:
      BRAIN_DEVICE: auto
      BRAIN_GPU_MEM_MIB: "1024"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## API

- GET /health -> {ok, neurons, synapses, sim_speed, sim_capacity, speed_target, sim_ms, threads, device}
  sim_speed = sim ms per wall ms over the last ~2 s; sim_capacity = same divided by busy time (what it could do flat out).
  device is new ("cuda" or "cpu"); nothing else changed.
- GET /params -> {key: {value, min, max, step, unit, desc, readonly}} (ranges for the web sliders).
  PUT /params takes a flat partial object, e.g. {"tau_m_ms": 25}. Out of range / unknown key / dt_ms != 0.1 -> 422.
  POST /params/reset restores the reference values.
- GET /groups -> {name: {size, role, default_rate_hz, description}}
- POST /stimulate {group | neuron_ids, rate_hz}; rate_hz defaults to the group's default_rate_hz; 0 removes it.
  neuron_ids may be FlyWire root ids or tensor indices (< 138639).
- POST /silence {group | neuron_ids}; POST /clear (drops all stimulation and silencing); POST /reset (clears
  dynamical state and time, keeps stimulation and silencing).
- PUT /speed {target} (extra): real-time factor cap, 0 = flat out.
- GET /state (extra): active stimulation/silencing.
- WS /stream, 30 Hz JSON: {t_ms, motor:{forward 0..1, turn -1..1}, group_rates:{name: hz}, spikes:[neuron idx...],
  params:{flat}, sim_speed}.

## Model (app/model.py, app/cuda_lif.py)

Update order is a transcription of the reference AlphaLIF/TorchModel step (dt 0.1 ms, float32). Steps are numbered
from 1.

- Delay: a spike at step s enters the synaptic input at step s+20 (delay line of 19 slots, plus one). A block of
  BLOCK = 20 steps therefore depends only on spikes of earlier blocks. This is what allows 20 steps per launch.
- conductance: g' = g (1 - dt/tauSyn) + gate * wScale * I; spiking neurons get g = 0.
- refractory gate: a neuron that spiked at step s gets no synaptic input at s+1 .. s+22 (tRefrac 2.2 ms).
  Stimulated neurons are exempt, as in the reference. The CUDA path keeps the last spike step per neuron instead of a
  22-step history (same gate).
- membrane u = v - v_rest: u += wScale*250*Bernoulli(rate*dt) for stimulated neurons; u += dt/tauMem (g - u);
  spike if u > v_th - v_rest; reset to v_reset.
- Live parameters: w_syn (also scales the Poisson kicks, as in the reference), tau_m, v_th, v_reset, tau_syn, t_ref,
  poisson_rate_hz (background kicks to random neurons, an addition to the reference, default 0).
- Connectome: 138,639 neurons, 15,091,983 nonzero synapses from 2025_Connectivity_783.parquet (SPEC says ~5M; the data
  has 15.1M rows). Stored as CSC by presynaptic neuron (indptr int64, post int32, w float32): 121 MiB on the GPU.

CUDA backend, per block of 20 steps (one CUDA graph, one replay per block, no host sync inside):

1. _lif_block_kernel (Triton, one thread per neuron): loops over the 20 steps in registers (gate, input, kicks,
   conductance, membrane, threshold, reset). Writes spikes S[20, n]. Also clears the block input, updates the
   rolling 100 ms window and the since-last-frame counters.
2. spike compaction: torch cumsum of S, then _compact_kernel writes the list of spiking (step, neuron) pairs.
3. _scatter_kernel: for every pair, atomic adds of W[:, pre] into the input of the next block. Work scales with the
   spikes, not with the 15M synapses.

Differences from the CPU backend (same model, documented):
- Poisson background: CPU draws a Poisson count of kicks, CUDA draws a Bernoulli per neuron per step (same mean,
  at most one kick per neuron per step).
- atomic adds change the float32 summation order (as the CPU index_add does relative to the reference).

Sync points: run() waits for the device once per 32 ms chunk (the runner's pacing unit, 320 steps), and each /stream
frame reads two counter vectors (about 30 per second). No per-step or per-block host sync.

## Verification

- tests/test_model_toy.py: all semantics tests run on CPU and CUDA. test_cuda_matches_cpu_raster_deterministic:
  identical spike rasters on a random 60-neuron network under deterministic drive (probability 1 per step).
- tests/test_device.py: BRAIN_DEVICE resolution and CPU fallback.
- tools/bench.py --verify: connectome, 600 steps, 125 neurons driven at probability 1: CUDA and CPU rasters identical
  (103,612 spike events each).
- tools/verify_reference.py 400 (float32, CPU backend, vs reference/run_pytorch.py): 4 of 400 steps differ by one
  neuron (first at step 238). Same float32 summation-order effect as before; not a logic difference.

## Vision (app/retina.py, app/retina.json, POST /vision)

- GET /vision: map and transducer status. POST /vision {left[661], right[661], max}: one luminance image per eye (physics sends 0..255
  with max 255, see physics/README.md "Real vision"; enable with VISION_TO_BRAIN=1 on physics). DELETE /vision stops the drive.
- Each photoreceptor (R1-6, R7, R8; 5,208 left + 5,374 right in the map) samples its 3 nearest ommatidia (Gaussian, sigma 2.2 deg) and
  fires at 200 Hz x clip(2 (A - I) / A, 0, 1): darkening relative to its adapted level A (tau 1 s, local). OFF code, reasons in
  retina.py. Static scenes fade out; receding or brightening objects give no input. Env: VISION_EYES, VISION_ADAPT, VISION_SUBSAMPLE.
- Measured (brain/tools/measure_vision_loom.py, details in physics/README.md): with the reference constants the photoreceptor drive
  does not reach LC4 / LPLC2 / the giant fibre (0 Hz for a looming object and for a fully dark field). Activity dies 2 synapses
  from the receptors. The looming groups therefore stay scripted by default (physics LOOMING_DRIVE).

## Stimulation groups (app/groups.json, built by tools/make_groups.py)

Ids come from reference/rojas/brain_body_bridge.py (DN_NEURONS, STIMULI) plus the FlyWire annotation table for LPLC2.

- Inputs: sugar_grn (21, 200 Hz), bitter_grn (42), looming (LC4, 104), lplc2 (210, 100 Hz), jo_mechano (209),
  or56a_olfactory (39), p9 (2).
- Readout / motor (descending neurons):
  - forward_walk: P9_left, P9_right, P9_oDN1_left, P9_oDN1_right (rojas DN_GROUPS["forward"]). This is the
    forward-walk choice; it is the walking DN set the reference uses, not a guess from the SEZ pickle.
  - turn_left: DNa01_left, DNa02_left; turn_right: DNa01_right, DNa02_right (rojas DN_GROUPS turn_L / turn_R).
  - backward_walk (MDN x4), escape_gf (GF_1/2), groom_adn1, feed_mn9 (MN9 left/right): readout only.

## Motor mapping (app/runner.py)

Per-neuron mean rate of each readout group over a rolling 100 ms window.
forward = rate(forward_walk) / 200 Hz, clipped to [0, 1].
turn = (rate(turn_right) - rate(turn_left)) / 200 Hz, clipped to [-1, 1]; positive = right.
Both are smoothed with a time constant of 0.15 s. 200 Hz is MOTOR_NORM_HZ, the DNRateDecoder max_rate in the
rojas reference.

## Measured speed

Setup: this desktop, Ryzen 9 9900X (12 cores / 24 threads), RTX 5070 Ti. The GPU was shared with another
process at 100 % utilisation during every run, so the GPU numbers are pessimistic; an earlier run (lower
neighbour load) gave 27x / 26x / 21x for idle / sugar / heavy. Image flybrain-brain:cuda, torch 2.11.0+cu128,
Triton 3.6.0. CPU runs use docker --cpus 8 with TORCH_THREADS=8, a proxy for the 8 Zen 3 laptop cores (the laptop
is about 30 % slower per core and throttles, so the CPU numbers are optimistic for it).

sim_speed = sim ms per wall ms. "chunk" = wall time of one 32 ms run() call (the unit the runner paces with;
real time needs chunk < 32 ms). Heavy = 338 stimulated neurons, stress = 627.

Engine (tools/bench.py, 2 s of sim per scenario, flat out):

| scenario | GPU capacity | GPU chunk mean / max | CPU 8 cores capacity | CPU chunk mean |
|---|---|---|---|---|
| idle | 9.2x | 3.5 / 3.5 ms | 1.57x | 20 ms |
| sugar_grn 100 Hz (21 neurons, 9.7k spikes/s) | 9.1x | 3.5 / 3.6 ms | 0.84x | 38 ms |
| heavy: sugar 100 + forward_walk 150 + looming 200 + jo 300 (338 neurons, 129k spikes/s) | 8.3x | 3.9 / 3.9 ms | 0.57x | 56 ms |
| stress: all 7 input groups at default rates (627 neurons, 651k spikes/s) | 7.9x | 4.0 / 4.1 ms | 0.31x | 102 ms |

Live server (uvicorn, HTTP, /health; paced = speed target 1.0, flat out = target 0):

| scenario | GPU paced | GPU flat out | CPU 8 cores paced | CPU 8 cores flat out |
|---|---|---|---|---|
| idle | 1.00x | 9.4x | 1.00x | 1.25x |
| sugar 100 Hz | 1.00x | 9.1x | 0.59x | 0.57x |
| heavy (338) | 1.00x | 8.3x | 0.41x | 0.39x |
| stress (627) | 1.00x | 7.8x | 0.25x | 0.27x |

The CPU server numbers are below the engine numbers because the frame builder and the event loop share the GIL and
the 8-CPU quota. The CPU path is only the fallback; on 8 cores it is real time at idle and not with any stimulation.

GPU memory: torch peak reserved 210 MiB (193 MiB allocated: connectome 121 MiB, ring and spike buffers about
70 MiB). Whole process on the GPU, as nvidia-smi reports it including the CUDA context: 554 MiB, under the 1024 MiB cap.
Both are below the 1 GiB limit used here and below the 2 GiB laptop limit.

Laptop (RTX 3050 Ti, not measured): the GPU part of a block is bound by memory bandwidth and the 3050 Ti has about
1/4.7 of the 5070 Ti's (192 vs 896 GB/s). Scaling the measured chunk time by that ratio gives about 16 ms per 32 ms
chunk, i.e. about 2x real time. This is an estimate; it must be measured on the laptop.

## Docker GPU access on this desktop

`docker run --gpus all` fails here: "failed to discover GPU vendor from CDI: no known GPU vendor found". The
nvidia-container-toolkit / CDI spec is not installed on this host. The tests above passed the GPU by hand:
device nodes (/dev/nvidia0, nvidiactl, nvidia-uvm, nvidia-uvm-tools, nvidia-modeset), the host driver libraries
(libcuda, libnvidia-ml, libnvidia-ptxjitcompiler, nvidia-smi) bind-mounted read-only, and TRITON_LIBCUDA_PATH
pointing at the mounted libcuda. The laptop should use the standard toolkit path (compose snippet above).

## Deviations from SPEC.md

- GET /params returns the {value,min,max,step,...} shape the task asked for (the SPEC said a flat current
  object). PUT /params still takes the flat form. The stream frame keeps flat params.
- /stream spikes: a flat list of neuron indices that spiked since the previous frame, uniformly subsampled to
  2000. SPEC "recent window" is read as "since last frame". The web UI uses len(spikes) as the spike count.
- Silencing is a mask (silenced neurons never spike and have no input); the connectome is not edited.
- Stimulated neurons are exempt from the refractory gate. The exemption follows the current stimulation
  (the reference fixes the set at construction).
- /reset does not clear stimulation or silencing (it resets state and the clock only). /clear does both.
- Extra endpoints: /params/reset, /state, PUT /speed, /params/meta removed.
- /health gains "device". Synapse count is 15.1M, not ~5M (see above).
- The sim advances in blocks of 20 steps (2 ms); run() only accepts multiples of 20 steps.
