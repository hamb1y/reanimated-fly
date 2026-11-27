import pytest

from app import brainloop as BL
from app.behavior import LOOM_HZ, LPLC2_HZ


def _r(mode="explore", forward=0.8, turn=0.1, drives=None, flee=None):
    return {"mode": mode, "forward": forward, "turn": turn, "drives": drives or {}, "flee": flee}


def test_speed_request_becomes_a_walking_neuron_rate():
    out = BL.apply(_r(forward=0.5), (0.4, 0.0))
    assert out["drives"]["p9"] == pytest.approx(0.5 * BL.GO_HZ_PER_UNIT)
    assert out["forward"] == pytest.approx(0.4)  # the fly walks at what the BRAIN produced, not the request
    assert out["turn"] == pytest.approx(0.1) and out["steering"] == "rule"  # wander turn stays a rule


def test_no_walk_drive_when_standing():
    out = BL.apply(_r(forward=0.0), (0.0, 0.0))
    assert "p9" not in out["drives"]


def test_no_brain_passes_the_scripted_command_through():
    out = BL.apply(_r(forward=0.7, turn=-0.3), None)
    assert out["forward"] == 0.7 and out["turn"] == -0.3 and out["steering"] == "rule"
    assert out["drives"]["p9"] == pytest.approx(0.7 * BL.GO_HZ_PER_UNIT)  # still tries to drive a brain that may reconnect


def test_scary_object_on_the_left_drives_the_left_looming_neurons_only():
    flee = {"bearing": 0.8, "dist": 20.0, "grow": 0.0, "fear": 1.0}
    out = BL.apply(_r("flee", forward=1.0, drives={"looming": 99.0, "lplc2": 99.0, "escape_gf": 50.0}, flee=flee), (1.0, 0.0))
    d = out["drives"]
    assert "looming" not in d and "lplc2" not in d and "looming_right" not in d and "lplc2_right" not in d
    assert d["looming_left"] > 0 and d["lplc2_left"] > 0 and d["escape_gf"] == 50.0
    assert d["looming_left"] == pytest.approx(LOOM_HZ * (1 - 20.0 / 60.0), abs=0.01)  # threat from proximity when not expanding
    assert d["lplc2_left"] == pytest.approx(LPLC2_HZ * (1 - 20.0 / 60.0), abs=0.01)


def test_scary_object_on_the_right_drives_the_right_neurons():
    flee = {"bearing": -0.8, "dist": 20.0, "grow": 0.5, "fear": 1.0}
    d = BL.apply(_r("flee", forward=1.0, flee=flee), (1.0, 0.0))["drives"]
    assert d["looming_right"] > 0 and "looming_left" not in d


def test_brain_turn_sign_is_flipped_for_the_body():
    flee = {"bearing": 0.8, "dist": 20.0, "grow": 0.0, "fear": 1.0}
    # looming on the left made the brain turn RIGHT (+). Physics turn is + = LEFT, so the body turn must be negative.
    right = BL.apply(_r("flee", forward=1.0, flee=flee), (1.0, 0.11))
    assert right["steering"] == "brain" and right["turn"] == pytest.approx(-BL.TURN_GAIN * 0.11)
    left = BL.apply(_r("flee", forward=1.0, flee=flee), (1.0, -0.11))
    assert left["turn"] == pytest.approx(BL.TURN_GAIN * 0.11)
    assert BL.apply(_r("flee", forward=1.0, flee=flee), (1.0, 0.0))["turn"] == pytest.approx(0.0, abs=1e-9)


def test_weak_threat_sends_no_looming():
    flee = {"bearing": 0.3, "dist": 59.9, "grow": 0.0, "fear": 0.9}
    d = BL.apply(_r("flee", forward=1.0, flee=flee), (1.0, 0.0))["drives"]
    assert "looming_left" not in d and "looming_right" not in d
