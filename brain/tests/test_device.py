"""Device selection (no GPU needed: CUDA availability is monkeypatched)."""
import numpy as np
import pytest
import torch

from app import model
from app.model import Brain, make_brain, resolve_device


def tiny():
    indptr = np.array([0, 1, 1], dtype=np.int64)
    return indptr, np.array([1], dtype=np.int32), np.array([100.0], dtype=np.float32)


def test_resolve_device_values():
    assert resolve_device(None) == "auto" and resolve_device("CPU") == "cpu" and resolve_device(" cuda ") == "cuda"
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_cpu_request_builds_cpu_brain():
    b = make_brain(*tiny(), requested="cpu")
    assert isinstance(b, Brain) and b.device == "cpu"


def test_auto_without_cuda_uses_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert make_brain(*tiny(), requested="auto").device == "cpu"


def test_auto_falls_back_when_cuda_init_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("no driver")

    monkeypatch.setattr(model, "Brain", lambda *a, **k: boom() if k.get("device") == "cuda" else Brain(*a, **k))
    assert make_brain(*tiny(), requested="auto").device == "cpu"


def test_explicit_cuda_failure_is_an_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model, "Brain", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no GPU")))
    with pytest.raises(RuntimeError):
        make_brain(*tiny(), requested="cuda")
