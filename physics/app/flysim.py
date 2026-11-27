"""MuJoCo fly (NeuroMechFly v2 via flygym 2.1.1) with the tripod CPG walking controller.

Fidelity levels are (physics timestep, physics steps per controller tick). The controller
runs once per tick (tick = steps * timestep, 2 ms at level 0) and holds its joint targets
for the physics steps in between. Level 0 is the default. Higher levels are cheaper and are
used only as a fallback when the host cannot keep up; see README.md for measurements.
"""

from __future__ import annotations

import mujoco
import numpy as np
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

from . import collide, obstacles
from .command import descending_drive
from .groom import Groomer

FLY_NAME = "nmf"
SPAWN_POS = (0.0, 0.0, 0.8)  # mm, as in the flygym turning-controller tutorial
# (physics timestep s, physics steps per controller tick)
FIDELITY_LEVELS = ((2e-4, 10), (2.5e-4, 8))


class FlySim:
    """Single fly, flat ground, position-controlled legs, CPG + correction controller."""

    def __init__(self, level: int = 0) -> None:
        self.fly = make_locomotion_fly(name=FLY_NAME, add_adhesion=True)
        self.world = FlatGroundWorld()
        self.world.add_fly(
            self.fly,
            list(SPAWN_POS),
            Rotation3D("quat", [1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )
        # obstacles (app/collide.py): kitchen props as static geoms, plus a mocap pool for user-placed solids
        self.props_raw, self.props = obstacles.load_props()
        collide.add_prop_geoms(self.world.mjcf_root, self.props)
        collide.add_pool(self.world.mjcf_root)
        dt, self.control_steps = FIDELITY_LEVELS[level]
        self.level = level
        self.sim = Simulation(self.world, timestep=dt)
        self.n_fly_collide = collide.enable_fly_contacts(self.sim.mj_model, FLY_NAME)
        self.pool = collide.Pool(self.sim.mj_model)
        self.preprogrammed = PreprogrammedSteps()
        self.dof_order = self.fly.get_actuated_jointdofs_order("position")
        self.controller = HybridTurningController(
            timestep=self.control_dt,
            preprogrammed_steps=self.preprogrammed,
            output_dof_order=self.dof_order,
        )
        self.groom = Groomer(self.dof_order)  # replayed foreleg grooming; idle unless requested (app/groom.py)
        self._scene: dict | None = None
        self.reset()

    @property
    def timestep(self) -> float:
        return float(self.sim.timestep)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.control_steps

    @property
    def sim_time(self) -> float:
        return float(self.sim.mj_data.time)

    @property
    def body_names(self) -> list[str]:
        m = self.sim.mj_model
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}" for i in range(m.nbody)]

    def set_level(self, level: int) -> None:
        """Switch fidelity live (keeps the body state). Call from the sim thread only."""
        if level == self.level:
            return
        dt, k = FIDELITY_LEVELS[level]
        self.sim.mj_model.opt.timestep = dt
        self.control_steps = k
        self.controller.timestep = k * dt
        self.controller.cpg_network.timestep = k * dt
        self.level = level

    def reset(self) -> None:
        self.groom.reset()
        self.sim.reset()
        self.controller.reset(seed=0)
        initial = LocomotionAction(
            joint_angles=self.preprogrammed.default_pose_by_dof_order(self.dof_order),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        apply_locomotion_action(self.sim, FLY_NAME, initial)
        self.sim.warmup()
        self.pool.apply(self.sim.mj_model, self.sim.mj_data)  # the keyframe reset parks the pool; put user solids back

    def advance(self, forward: float, turn: float, n_ticks: int) -> None:
        """Advance n_ticks controller ticks under a constant motor command.

        Uses raw mj_step: flygym's Simulation.step is only mj_step, so this is the same physics
        with less Python overhead per physics step.
        """
        signal = np.asarray(descending_drive(forward, turn), dtype=float)
        m, d = self.sim.mj_model, self.sim.mj_data
        k = self.control_steps
        self.pool.apply(m, d)  # user-placed solids (set from the world loop via pool.set)
        for _ in range(n_ticks):
            obs = HybridControllerObservation.from_sim(self.sim, FLY_NAME)
            engaged = self.groom.engaged
            action = self.controller.step(np.zeros(2) if engaged else signal, obs)  # grooming freezes the walk command
            if engaged:
                angles, adh = self.groom.apply(action.joint_angles, action.adhesion_onoff, self.control_dt)
                action = LocomotionAction(joint_angles=angles, adhesion_onoff=adh)
            apply_locomotion_action(self.sim, FLY_NAME, action)
            for _ in range(k):
                mujoco.mj_step(m, d)

    def pose(self) -> dict:
        """Copy of world-frame body positions (mm) and quaternions (wxyz) by body id."""
        d = self.sim.mj_data
        return {
            "t": float(d.time),
            "pos": np.array(d.xpos, dtype=float),
            "quat": np.array(d.xquat, dtype=float),
        }

    def body_index(self, name: str) -> int:
        return self.body_names.index(name)

    def scene(self) -> dict:
        if self._scene is None:
            self._scene = self._build_scene()
        return self._scene

    def _build_scene(self) -> dict:
        m = self.sim.mj_model
        bodies = [
            {"id": i, "name": self.body_names[i], "parent": int(m.body_parentid[i])}
            for i in range(m.nbody)
        ]
        geoms = []
        for g in range(m.ngeom):
            gtype = int(m.geom_type[g])
            is_mesh = gtype == int(mujoco.mjtGeom.mjGEOM_MESH)
            geoms.append(
                {
                    "id": g,
                    "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "",
                    "body": int(m.geom_bodyid[g]),
                    "type": mujoco.mjtGeom(gtype).name.removeprefix("mjGEOM_").lower(),
                    "size": [float(v) for v in m.geom_size[g]],
                    "pos": [float(v) for v in m.geom_pos[g]],
                    "quat": [float(v) for v in m.geom_quat[g]],
                    "rgba": [float(v) for v in m.geom_rgba[g]],
                    "group": int(m.geom_group[g]),
                    "mesh_id": int(m.geom_dataid[g]) if is_mesh else None,
                }
            )
        return {
            "units": "mm",
            "quat_order": "wxyz",
            "timestep": self.timestep,
            "fly": FLY_NAME,
            "bodies": bodies,
            "geoms": geoms,
            "meshes": [
                {"id": i, "vertices": int(m.mesh_vertnum[i]), "faces": int(m.mesh_facenum[i])}
                for i in range(m.nmesh)
            ],
        }

    def mesh(self, mesh_id: int) -> dict | None:
        m = self.sim.mj_model
        if not 0 <= mesh_id < m.nmesh:
            return None
        va, vn = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
        fa, fn = int(m.mesh_faceadr[mesh_id]), int(m.mesh_facenum[mesh_id])
        faces = np.asarray(m.mesh_face[fa : fa + fn], dtype=np.int64)
        if faces.size and faces.max() >= vn:  # tolerate global vertex indices
            faces = faces - va
        return {
            "id": mesh_id,
            "vertices": np.round(m.mesh_vert[va : va + vn], 5).tolist(),
            "faces": faces.tolist(),
        }
