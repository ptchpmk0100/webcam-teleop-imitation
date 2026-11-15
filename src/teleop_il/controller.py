"""Policy-path controller (PRD-v2): a gripper-frame target position in, 6 joint targets out.

The state policy v2 outputs where the gripper frame should be next; this controller turns that into joint targets by
damped least squares on the gripper site's Jacobian, starting from the MEASURED joints (no memory between calls), with
a soft orientation term that keeps the gripper pointing down and turned with the pan joint, as the teleop and the
expert do (`ik.ArmController`). The gripper itself is open or closed, nothing in between.

It is written with plain MuJoCo calls (kinematics, site Jacobian, quaternion error) so the C++ rollout can do exactly
the same (ticket 24). Teleop keeps using `ik.ArmController` (mink); only the policy path uses this.

Ticket 19 screen (2026-11-15, scratch probes): the 100 human demos were replayed with the target "recorded gripper
position one step ahead", IK from the measured joints, gripper open/closed from the recorded target. Orientation weight
0.02 (ArmController's value) lifted 88 of 100; 0.1 lifted 69; 0.005 lifted 93, and all 77 demos that did not push the
cube before the first close; 0.002 and 0.001 lifted 86 and 87. Pushed demos replay 10 to 16 of 23 whatever the
setting (contact chaos). Gains above 1 on the tracking error overshoot: the servos are force-limited (2.94 N m at
kp 998), so any error above about 3 mrad already saturates them.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from teleop_il.env import ARM_JOINTS, HOME_ARM, HOME_GRIPPER, build_model
from teleop_il.ik import GRIPPER_CLOSED, GRIPPER_OPEN, _align_axis_down, _rot_z

ITERATIONS = 8
ORIENTATION_WEIGHT = 0.005          # position first, pointing down is soft; replay screen below picked it

DAMPING = 1e-4                      # added to the normal equations' diagonal (rad^-2 scale)


class GripperController:
    def __init__(self, iterations: int = ITERATIONS, orientation_weight: float = ORIENTATION_WEIGHT,
                 damping: float = DAMPING) -> None:
        self._iterations, self._w, self._damping = iterations, orientation_weight, damping
        m = self._model = build_model(with_cube=False)
        self._data = mujoco.MjData(m)
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[:5]]
        self._qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self._dofadr = np.array([m.jnt_dofadr[j] for j in joints])
        self._site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._lo = m.actuator_ctrlrange[:5, 0].copy()
        self._hi = m.actuator_ctrlrange[:5, 1].copy()
        self._jac_p = np.zeros((3, m.nv))
        self._jac_r = np.zeros((3, m.nv))
        home = self._kinematics(np.array(HOME_ARM))
        self._home_rotation = _align_axis_down(self._data.site_xmat[self._site].reshape(3, 3).copy())
        self._home_yaw = math.atan2(home[1], home[0])
        _ = HOME_GRIPPER

    def _kinematics(self, arm: np.ndarray) -> np.ndarray:
        d = self._data
        d.qpos[self._qadr] = arm
        mujoco.mj_kinematics(self._model, d)
        mujoco.mj_comPos(self._model, d)
        return d.site_xpos[self._site].copy()

    def gripper_position(self, joints) -> np.ndarray:
        """Where the gripper frame is with the arm at `joints` (forward kinematics; at least the 5 arm joints)."""
        return self._kinematics(np.asarray(joints, dtype=float)[:5])

    def target_rotation(self, target) -> np.ndarray:
        """Pointing straight down, turned about the vertical by the target's bearing from the base (as the pan
        joint turns the arm)."""
        yaw = math.atan2(float(target[1]), float(target[0])) - self._home_yaw
        return _rot_z(yaw) @ self._home_rotation

    def _limited_step(self, q: np.ndarray, jac: np.ndarray, err: np.ndarray) -> np.ndarray:
        """One damped least-squares step. A joint that the step would push past its limit is put on the limit and
        left out of a re-solve, so the other joints take over its share (plain clipping lost that share: moving 2 cm
        up from home stopped 2 cm short)."""
        free = np.ones(5, bool)
        base = q.copy()                       # frozen joints sit on their limit here, free ones at the start value
        for _ in range(5):
            remaining = err - jac @ (base - q)                    # what the free joints still have to do
            j = jac[:, free]
            new = base.copy()
            new[free] += np.linalg.solve(j.T @ j + self._damping * np.eye(int(free.sum())), j.T @ remaining)
            over = free & ((new < self._lo) | (new > self._hi))
            if not over.any():
                return new
            base[over] = np.clip(new[over], self._lo[over], self._hi[over])
            free &= ~over
            if not free.any():
                return base
        return np.clip(new, self._lo, self._hi)

    def command(self, joints, target, closed: bool) -> np.ndarray:
        """Joint targets (5 arm joints plus the gripper) that put the gripper frame at `target` (world metres), from
        the measured `joints` (at least the 5 arm joints; a sixth value is ignored)."""
        q = np.clip(np.asarray(joints, dtype=float)[:5].copy(), self._lo, self._hi)
        target = np.asarray(target, dtype=float)
        goal_quat = np.empty(4)
        mujoco.mju_mat2Quat(goal_quat, self.target_rotation(target).ravel())
        quat, quat_conj, err_quat, err_rot = np.empty(4), np.empty(4), np.empty(4), np.empty(3)
        d = self._data
        for _ in range(self._iterations):
            x = self._kinematics(q)
            mujoco.mju_mat2Quat(quat, d.site_xmat[self._site])
            mujoco.mju_negQuat(quat_conj, quat)
            mujoco.mju_mulQuat(err_quat, goal_quat, quat_conj)          # rotation from now to the goal, world frame
            mujoco.mju_quat2Vel(err_rot, err_quat, 1.0)
            mujoco.mj_jacSite(self._model, d, self._jac_p, self._jac_r, self._site)
            jac = np.vstack([self._jac_p[:, self._dofadr], self._w * self._jac_r[:, self._dofadr]])
            err = np.r_[target - x, self._w * err_rot]
            q = self._limited_step(q, jac, err)
        return np.r_[q, GRIPPER_CLOSED if closed else GRIPPER_OPEN]
