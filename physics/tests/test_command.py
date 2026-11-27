import math

from app.command import MAX_DRIVE, CommandState, Motor, descending_drive


def test_zero_command_is_zero_drive():
    assert descending_drive(0, 0) == (0.0, 0.0)


def test_forward_drives_both_sides_equally():
    assert descending_drive(1, 0) == (MAX_DRIVE, MAX_DRIVE)


def test_positive_turn_strengthens_right_side():
    left, right = descending_drive(1, 0.5)
    assert right > left


def test_inputs_are_clamped_and_nonfinite_is_stop():
    assert descending_drive(5, -9) == descending_drive(1, -1)
    assert descending_drive(math.nan, 0.3) == (0.0, 0.0)
    assert descending_drive(math.inf, 0) == (0.0, 0.0)


def test_priority_manual_then_brain_then_hold():
    c = CommandState(stale_s=1.0)
    assert c.effective(0.0) == (Motor(), "hold")

    c.set_brain_connected(True)
    c.set_brain(1, 0, now=10.0)
    assert c.effective(10.5) == (Motor(1.0, 0.0), "brain")
    assert c.effective(11.5) == (Motor(), "hold")  # stale brain frame

    c.set_manual(0.5, 0.25)
    assert c.effective(11.5) == (Motor(0.5, 0.25), "manual")
    c.release_manual()
    assert c.effective(11.5) == (Motor(), "hold")


def test_disconnect_drops_brain_command():
    c = CommandState()
    c.set_brain_connected(True)
    c.set_brain(1, 0, now=0.0)
    c.set_brain_connected(False)
    assert c.effective(0.1) == (Motor(), "hold")
    assert c.status(0.1)["brain_connected"] is False
