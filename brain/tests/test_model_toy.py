"""Model semantics on a hand-made 4-neuron network (no data needed). Runs on CPU and, if present, CUDA."""
import numpy as np
import pytest
import torch

from app.model import BLOCK, DELAY_SLOTS, Brain

torch.set_num_threads(1)
CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def toy(edges, n=4, device="cpu"):
    """edges: (pre, post, weight). Returns Brain with CSC built from them."""
    pre = np.array([e[0] for e in edges], dtype=np.int64)
    post = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([e[2] for e in edges], dtype=np.float32)
    o = np.argsort(pre, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    return Brain(indptr, post[o], w[o], device=device, record=True)


def spike_steps(b, steps):
    """Run steps (multiple of BLOCK) -> {neuron: [steps at which it spiked]} (1-based step numbers)."""
    b.raster = []
    b.run(steps)
    seen = {}
    for t, idx in b.raster:
        for i in idx.tolist():
            seen.setdefault(i, []).append(t)
    return seen


@pytest.fixture(params=DEVICES)
def dev(request):
    return request.param


def test_no_input_no_spikes(dev):
    b = toy([(0, 1, 100.0)], device=dev)
    assert spike_steps(b, 500) == {}


def test_forced_spike_propagates_with_delay(dev):
    b = toy([(0, 1, 100.0), (1, 2, 100.0)], device=dev)
    b.stimulate([0], 1000.0 / 0.1)  # probability 1 per step: deterministic kicks
    seen = spike_steps(b, 160)
    assert 0 in seen and 1 in seen and 2 in seen
    # neuron 0 spikes at step 1; synaptic input enters the 19-slot delay line one step later
    assert seen[0][0] == 1
    assert seen[1][0] >= seen[0][0] + DELAY_SLOTS
    assert seen[2][0] > seen[1][0] + DELAY_SLOTS - 1


def test_inhibition_blocks(dev):
    b = toy([(0, 1, 100.0), (2, 1, -100.0)], device=dev)
    b.stimulate([0, 2], 1000.0 / 0.1)
    assert 1 not in spike_steps(b, 200)


def test_silence_stops_propagation_and_clear_restores(dev):
    b = toy([(0, 1, 100.0)], device=dev)
    b.stimulate([0], 1000.0 / 0.1)
    b.silence([1])
    assert 1 not in spike_steps(b, 200)
    b.clear()
    b.stimulate([0], 1000.0 / 0.1)
    assert 1 in spike_steps(b, 200)


def test_params_live_w_syn_zero_disables_propagation(dev):
    b = toy([(0, 1, 100.0)], device=dev)
    b.stimulate([0], 1000.0 / 0.1)
    b.set_params({"w_syn": 0.0})
    # w_syn also scales the Poisson kick (reference: scale * poisson), so nothing fires at all
    assert spike_steps(b, 200) == {}
    b.set_params({"w_syn": 0.275})
    assert 1 in spike_steps(b, 200)


def test_refractory_gates_synaptic_input_live(dev):
    def rate(t_ref):
        b = toy([(0, 1, 100.0)], device=dev)
        b.stimulate([0], 1000.0 / 0.1)
        b.set_params({"t_ref_ms": t_ref})
        return len(spike_steps(b, 1500).get(1, []))

    assert rate(0.0) > rate(2.2) > rate(8.0) > 0


def test_run_rejects_partial_blocks(dev):
    b = toy([(0, 1, 100.0)], device=dev)
    with pytest.raises(ValueError):
        b.run(BLOCK + 1)


@pytest.mark.skipif(not CUDA, reason="no CUDA device")
def test_cuda_matches_cpu_raster_deterministic():
    """Probability-1 drive (no RNG): CUDA and CPU must produce the same spike raster on a small recurrent net."""
    rng = np.random.default_rng(3)
    n = 60
    edges = [(int(a), int(b), float(w)) for a, b, w in zip(rng.integers(0, n, 400), rng.integers(0, n, 400),
                                                             rng.choice([60.0, 90.0, -40.0], 400))]
    res = {}
    for dev in ("cpu", "cuda"):
        b = toy(edges, n=n, device=dev)
        b.stimulate(list(range(8)), 1000.0 / 0.1)
        res[dev] = spike_steps(b, 600)
    assert res["cpu"], "network should be active"
    assert res["cuda"] == res["cpu"]
