"""Seam for the policy-path controller (PRD-v2): a gripper-frame target position in, 6 joint targets out.

Judged through MuJoCo's own forward kinematics on a fresh model, never through the controller's internals.
"""
import itertools

import mujoco
import numpy as np
import pytest

from teleop_il.controller import GripperController
from teleop_il.env import ARM_JOINTS, HOME_ARM, HOME_GRIPPER, SPAWN_X, SPAWN_Y, build_model
from teleop_il.ik import GRIPPER_CLOSED, GRIPPER_OPEN

_model = build_model(with_cube=False)
_data = mujoco.MjData(_model)
_site = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
_qadr = [_model.jnt_qposadr[mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS]
HOME = np.array([*HOME_ARM, HOME_GRIPPER])


def gripper_frame(joints):
    """Where the gripper frame is, and its approach axis, if the arm stood at `joints` (forward kinematics)."""
    _data.qpos[:] = 0
    _data.qpos[_qadr] = joints
    mujoco.mj_kinematics(_model, _data)
    return _data.site_xpos[_site].copy(), _data.site_xmat[_site].reshape(3, 3)[:, 0].copy()


@pytest.mark.parametrize("move", [(0.02, 0, 0), (-0.02, 0, 0), (0, 0.02, 0), (0, -0.02, 0), (0, 0, 0.02),
                                  (0, 0, -0.02)])
def test_a_two_centimetre_move_from_home_puts_the_gripper_on_the_target(move):
    start, _ = gripper_frame(HOME)
    target = start + np.array(move)
    q = GripperController().command(HOME, target, closed=False)
    reached, _ = gripper_frame(q)
    assert np.linalg.norm(reached - target) < 1e-3


@pytest.mark.parametrize("x, y, z", list(itertools.product(SPAWN_X, SPAWN_Y, (0.025, 0.09))))
def test_repeated_commands_reach_every_corner_of_the_spawn_rectangle_pointing_down(x, y, z):
    controller, q, target = GripperController(), HOME.copy(), np.array([x, y, z])
    for _ in range(30):
        q = controller.command(q, target, closed=False)
    reached, approach = gripper_frame(q)
    assert np.linalg.norm(reached - target) < 1e-3
    assert approach[2] < -0.9, f"approach axis {approach} is not pointing down"


def test_an_unreachable_target_gives_arm_joint_targets_inside_the_actuator_limits():
    model = build_model()
    q = HOME.copy()
    for _ in range(10):
        q = GripperController().command(q, np.array([1.0, 1.0, 1.0]), closed=True)
    lo, hi = model.actuator_ctrlrange[:5, 0], model.actuator_ctrlrange[:5, 1]
    assert np.all(q[:5] >= lo) and np.all(q[:5] <= hi)


@pytest.mark.parametrize("closed, expected", [(True, GRIPPER_CLOSED), (False, GRIPPER_OPEN)])
def test_the_gripper_target_is_fully_open_or_fully_closed(closed, expected):
    start, _ = gripper_frame(HOME)
    assert GripperController().command(HOME, start, closed=closed)[5] == expected
