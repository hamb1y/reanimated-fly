"""Put the brain in the loop (BRAIN_IN_LOOP, default on).

What the brain decides here, and what is still a rule (also shown in the UI):
- SPEED: the behaviour asks for a forward speed; it is turned into a firing rate for the P9 walking neurons
  (calibrated on the brain: 30/60/100/200/400 Hz gives forward 0.08/0.15/0.24/0.52/0.98). The fly then walks at the
  forward output the BRAIN produced, not at the requested value.
- FLEEING: the scary object stimulates the looming (LC4) and collision (LPLC2) neurons on the side it is on. The
  connectome answers with the OPPOSITE turn neurons (measured with the walking neurons on: looming on the left gives a
  brain turn of +0.05/+0.11/+0.11 at threat 0.3/0.67/1.0, on the right -0.08/-0.09/-0.11; + = right), and the fly turns
  by the brain's own turn output. Only the gain and the sign flip into the body convention are rules.
- Everything else (steering toward food and mates, the wander, the choice between fear / lust / hunger, eating) is
  still a scripted rule. Without a connected brain, the scripted command passes through unchanged.
"""
from __future__ import annotations

from .behavior import LOOM_HZ, LPLC2_HZ, SCARY_RANGE
from .world import clamp

GO_HZ_PER_UNIT = 410.0  # P9 rate per unit of requested forward speed (measured calibration, see above)
TURN_GAIN = 4.0  # the brain's turn output is small (about +-0.11 for one-sided looming at full threat); scaled up to steer
# NB: an earlier version subtracted a 'bias of +0.17' from the brain's turn. That number was an artifact of a measurement
# taken while other groups were being stimulated; with only the walking neurons on the bias is about -0.04. It was removed.
LOOM_GROUPS = {"left": ("looming_left", "lplc2_left"), "right": ("looming_right", "lplc2_right")}


def apply(r: dict, motor: tuple[float, float] | None) -> dict:
    """r: Behavior.tick() result. motor: (forward, turn) from a fresh brain frame, or None. Returns the adjusted dict."""
    out = dict(r)
    drives = dict(r["drives"])
    go = clamp(float(r["forward"]), 0.0, 1.0)
    flee = r.get("flee")
    if flee is not None:
        # threat = proximity or expansion, whichever is larger; shown to the brain as looming on the object's side
        near = clamp(1.0 - flee["dist"] / SCARY_RANGE, 0.0, 1.0)
        threat = flee["fear"] * max(flee["grow"], near)
        for g in ("looming", "lplc2"):
            drives.pop(g, None)
        loom_g, lplc_g = LOOM_GROUPS["left" if flee["bearing"] > 0 else "right"]
        if threat * LOOM_HZ >= 0.5:
            drives[loom_g] = round(LOOM_HZ * threat, 2)
            drives[lplc_g] = round(LPLC2_HZ * threat, 2)
    if go > 0.02:
        drives["p9"] = round(go * GO_HZ_PER_UNIT, 1)
    out["drives"] = drives
    out["steering"] = "rule"
    if motor is None:
        return out
    bf, bt = motor
    out["forward"] = clamp(bf, 0.0, 1.0)
    if flee is not None:
        out["turn"] = clamp(-TURN_GAIN * bt, -1.0, 1.0)  # brain turn: + = right; physics turn: + = left
        out["steering"] = "brain"
    return out
