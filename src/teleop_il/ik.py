"""Arm controller: end-effector position target in, joint targets out (differential IK with mink)."""
from __future__ import annotations

import math

import mink
import mujoco
import numpy as np

from teleop_il.env import HOME_ARM, HOME_GRIPPER, build_model

GRIPPER_OPEN = 1.2          # joint value, about 10 cm between the jaw tips (measured in MuJoCo)
GRIPPER_CLOSED = -0.175     # joint lower limit, jaw tips touching
_IK_DT = 0.01


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _align_axis_down(rotation: np.ndarray) -> np.ndarray:
    """Smallest rotation that turns the gripper approach axis (frame x) to straight down, other axes follow."""
    axis_now, down = rotation[:, 0], np.array([0.0, 0.0, -1.0])
    v = np.cross(axis_now, down)
    s, c = np.linalg.norm(v), float(axis_now @ down)
    if s < 1e-12:
        return rotation.copy()
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return (np.eye(3) + s * k + (1 - c) * (k @ k)) @ rotation


class ArmController:
    """Keeps the gripper position at the target, pointing roughly down (soft orientation weight)."""

    def __init__(self, orientation_cost: float = 0.02) -> None:
        self._model = build_model(with_cube=False)
        data = mujoco.MjData(self._model)
        self._home_q = np.array([*HOME_ARM, HOME_GRIPPER])
        data.qpos[:] = self._home_q
        mujoco.mj_forward(self._model, data)
        site = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._home_position = data.site_xpos[site].copy()
        self._home_rotation = _align_axis_down(data.site_xmat[site].reshape(3, 3).copy())
        self._home_yaw = math.atan2(self._home_position[1], self._home_position[0])
        self._task = mink.FrameTask("gripperframe", "site", position_cost=1.0,
                                    orientation_cost=orientation_cost, lm_damping=1.0)
        self._limits = [mink.ConfigurationLimit(self._model)]
        self.reset()

    @property
    def home_position(self) -> np.ndarray:
        return self._home_position.copy()

    @property
    def home_joints(self) -> np.ndarray:
        return self._home_q.copy()

    def reset(self, joints=None) -> None:
        """Start the IK from the home pose, or from `joints` (6 values) when another controller had the arm."""
        q = self._home_q.copy() if joints is None else np.asarray(joints, dtype=float).copy()
        self._cfg = mink.Configuration(self._model, q)

    def position(self) -> np.ndarray:
        """Gripper-frame position of the controller's current joint values (forward kinematics)."""
        return self._cfg.get_transform_frame_to_world("gripperframe", "site").translation().copy()

    def track(self, target_position, iterations: int = 8) -> np.ndarray:
        """Move the internal configuration towards the target; return the 6 joint values (gripper untouched)."""
        target = np.asarray(target_position, dtype=float)
        yaw = math.atan2(target[1], target[0]) - self._home_yaw      # the pan joint turns the whole arm
        pose = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(_rot_z(yaw) @ self._home_rotation), target)
        self._task.set_target(pose)
        for _ in range(iterations):
            velocity = mink.solve_ik(self._cfg, [self._task], _IK_DT, "daqp", 1e-3, limits=self._limits)
            self._cfg.integrate_inplace(velocity, _IK_DT)
        return self._cfg.q.copy()
