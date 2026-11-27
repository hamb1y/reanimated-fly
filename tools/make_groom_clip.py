#!/usr/bin/env python
"""Build physics/app/data/groom_clip.npz from the NeuroMechFly (v1) recorded grooming kinematics.

Source: https://github.com/NeLy-EPFL/NeuroMechFly  (Apache License 2.0)
        data/joint_tracking/grooming/fly1/df3d/joint_angles__180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl
Paper:  Lobato-Rios et al. 2022, Nature Methods 19, 620-627, doi:10.1038/s41592-022-01466-7
        (joint angles from DeepFly3D pose estimation of a tethered fly, then inverse kinematics on the NMF v1 model)

What this does (and nothing else):
  1. read the two front legs (LF, RF) x 7 angles (radians, 2 kHz in the repo, which is 100 fps data interpolated)
  2. keep every 20th sample (100 Hz) and smooth with a Savitzky-Golay filter (window 11 = 0.11 s, order 3). The roll
     angles are noisy (about 1000 deg/s of frame-to-frame jitter that the 3D pose cannot resolve); this is the only
     change to the recording.
  3. choose the segment [t0, t1] in the active head-grooming part (1.4 s .. 3.9 s) whose end best matches its start
     (position and velocity), so it can be looped. The residual jump is closed at run time by a short crossfade.

The pickle is read with a restricted unpickler (numpy / pandas / builtins only), because pickles can run code.

    python tools/make_groom_clip.py SRC_DIR OUT.npz
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
from scipy.signal import savgol_filter

KEYS = ["ThC_yaw", "ThC_pitch", "ThC_roll", "CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch"]
SRC_NAME = "joint_angles__180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl"
FPS = 100.0
ACTIVE = (1.4, 3.9)  # seconds: the foreleg sweeps over the head here (see physics/README.md)
MIN_LEN, MAX_LEN = 1.5, 2.6


class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("numpy", "pandas", "builtins", "collections", "_codecs") and name not in (
            "eval", "exec", "open", "__import__", "compile",
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def main(src_dir: str, out: str) -> None:
    with open(f"{src_dir}/{SRC_NAME}", "rb") as f:
        ang = _Safe(f).load()
    full = np.stack(
        [np.stack([np.asarray(ang[f"{s}F_leg"][k], dtype=float) for k in KEYS], 1) for s in "LR"], 1
    )  # (N, 2 legs, 7)
    x = full[::20]  # 100 Hz
    flat = savgol_filter(x.reshape(len(x), -1), 11, 3, axis=0)
    t = np.arange(len(x)) / FPS
    best = None
    for i in range(int(ACTIVE[0] * FPS), int(ACTIVE[1] * FPS)):
        for j in range(i + int(MIN_LEN * FPS), min(len(x) - 1, int(ACTIVE[1] * FPS), i + int(MAX_LEN * FPS))):
            jump = np.abs(flat[i] - flat[j]).max()
            vel = np.abs((flat[i + 1] - flat[i]) - (flat[j] - flat[j - 1])).max() * FPS
            cost = jump + 0.03 * vel  # rad, and 0.03 s of velocity mismatch
            if best is None or cost < best[0]:
                best = (cost, i, j, jump, vel)
    _, i, j, jump, vel = best
    seg = flat[i:j].reshape(-1, 2, 7).astype(np.float32)
    np.savez_compressed(
        out,
        angles=seg,  # (n, 2 legs [L, R], 7 [KEYS]) radians, data convention (see groom.py)
        fps=np.float32(FPS),
        keys=np.array(KEYS),
        source=np.array(f"NeLy-EPFL/NeuroMechFly data/joint_tracking/grooming/fly1/df3d/{SRC_NAME}"),
        license=np.array("Apache-2.0"),
        t0=np.float32(t[i]),
        t1=np.float32(t[j]),
        closure_jump_deg=np.float32(np.degrees(jump)),
        closure_vel_mismatch_deg_s=np.float32(np.degrees(vel)),
    )
    print(f"segment {t[i]:.2f}..{t[j]:.2f} s ({len(seg)} samples at {FPS:.0f} Hz); end/start jump {np.degrees(jump):.1f} deg, "
          f"velocity mismatch {np.degrees(vel):.0f} deg/s")
    print("ranges (deg) L:", np.round(np.degrees(seg[:, 0].min(0))), np.round(np.degrees(seg[:, 0].max(0))))
    print("ranges (deg) R:", np.round(np.degrees(seg[:, 1].min(0))), np.round(np.degrees(seg[:, 1].max(0))))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
