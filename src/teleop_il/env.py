"""Tabletop pick-and-lift environment: SO-101 arm, one cube, a floor acting as the table.

Public interface (seam A): reset(seed), step(joint_targets), success(), lift_height(),
place_cube(position), snapshot(), restore(snapshot). Success is read from MuJoCo state only.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

CONTROL_HZ = 20
CONTROL_DT = 1.0 / CONTROL_HZ
EPISODE_SECONDS = 15.0
CUBE_HALF = 0.02                      # 4 cm cube
LIFT_THRESHOLD = 0.05                 # cube bottom more than 5 cm above the table
LIFT_HOLD_SECONDS = 1.0               # for at least 1 s, continuously
SPAWN_X = (0.17, 0.30)                # metres forward of the arm base; near edge proven by the expert (ticket 06), was 0.10
SPAWN_Y = (-0.15, 0.15)               # metres sideways
HOME_ARM = (0.0734, -0.2715, 0.1862, 1.6561, 1.673)   # IK-derived: gripper at (0.20, 0, 0.09), pointing straight down
HOME_GRIPPER = 0.7277
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "mujoco_menagerie" / "robotstudio_so101"


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def build_model(with_cube: bool = True) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ASSET_DIR / "so101.xml"))
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    if not with_cube:
        return spec.compile()
    cube = spec.worldbody.add_body(name="cube", pos=[0.2, 0.0, CUBE_HALF])
    cube.add_freejoint()
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  friction=[1, 0.03, 0.003], contype=2, conaffinity=1, solref=[0.01, 1],
                  rgba=[0, 1, 0, 1])
    return spec.compile()


class Env:
    def __init__(self) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self._substeps = round(CONTROL_DT / self.model.opt.timestep)
        assert abs(self._substeps * self.model.opt.timestep - CONTROL_DT) < 1e-9
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self._arm_qadr = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_qadr = int(self.model.jnt_qposadr[self.model.body_jntadr[cube_body]])
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        self._home = np.array([*HOME_ARM, HOME_GRIPPER])
        self._cube_dofadr = int(self.model.body_dofadr[cube_body])
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._hold_needed = round(LIFT_HOLD_SECONDS / CONTROL_DT)
        self._hold_steps = 0
        self._succeeded = False

    def ee_position(self) -> np.ndarray:
        """Where the gripper frame really is now, read from the simulation (not the commanded target)."""
        return self.data.site_xpos[self._ee_site].copy()

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper joint-target limits (5 arm joints plus the gripper)."""
        return self._ctrl_range[:, 0].copy(), self._ctrl_range[:, 1].copy()

    def _obs(self) -> dict:
        return {
            "joint_pos": self.data.qpos[self._arm_qadr].copy(),
            "cube_pose": self.data.qpos[self._cube_qadr:self._cube_qadr + 7].copy(),
            "time": float(self.data.time),
        }

    def reset(self, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        x = rng.uniform(*SPAWN_X)
        y = rng.uniform(*SPAWN_Y)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qadr] = self._home
        self.data.ctrl[:] = self._home
        self.data.qpos[self._cube_qadr:self._cube_qadr + 7] = [x, y, CUBE_HALF, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self._hold_steps = 0
        self._succeeded = False
        return self._obs()

    def step(self, action) -> dict:
        action = np.asarray(action, dtype=float)
        assert action.shape == (6,), "action is 5 arm joint targets plus the gripper target"
        self.data.ctrl[:] = np.clip(action, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self._hold_steps = self._hold_steps + 1 if self.lift_height() > LIFT_THRESHOLD else 0
        if self._hold_steps >= self._hold_needed:
            self._succeeded = True
        return self._obs()

    def success(self) -> bool:
        """True once the cube bottom has stayed more than 5 cm above the table for 1 s in a row."""
        return self._succeeded

    def lift_height(self) -> float:
        """Height of the lowest point of the cube above the table, read from MuJoCo state."""
        a = self._cube_qadr
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[a + 3:a + 7])
        half_extent_z = CUBE_HALF * np.abs(rot.reshape(3, 3)[2]).sum()
        return float(self.data.qpos[a + 2] - half_extent_z)

    def place_cube(self, position) -> None:
        """Put the cube upright at `position` (x, y, z of its centre) with zero velocity."""
        a = self._cube_qadr
        self.data.qpos[a:a + 7] = [*position, 1, 0, 0, 0]
        self.data.qvel[self._cube_dofadr:self._cube_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> dict:
        """Full simulator state plus success progress; restore() makes later steps replay identically."""
        d = self.data
        return {
            "time": float(d.time),
            "qpos": d.qpos.copy(),
            "qvel": d.qvel.copy(),
            "ctrl": d.ctrl.copy(),
            "qacc_warmstart": d.qacc_warmstart.copy(),
            "hold_steps": self._hold_steps,
            "succeeded": self._succeeded,
        }

    def restore(self, snapshot: dict) -> None:
        d = self.data
        d.time = snapshot["time"]
        d.qpos[:] = snapshot["qpos"]
        d.qvel[:] = snapshot["qvel"]
        d.ctrl[:] = snapshot["ctrl"]
        d.qacc_warmstart[:] = snapshot["qacc_warmstart"]
        self._hold_steps = snapshot["hold_steps"]
        self._succeeded = snapshot["succeeded"]
        mujoco.mj_forward(self.model, d)
