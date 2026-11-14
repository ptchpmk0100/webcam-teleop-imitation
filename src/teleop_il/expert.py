"""Scripted pick-and-lift expert (ticket 06): the ceiling for the task and the proof of the spawn rectangle.

The SO-101 gripper is asymmetric: the fixed jaw sits about 2 cm to one side of the gripper frame and the moving jaw
swings in from the other side. Aiming the frame at the cube centre lands the fixed jaw on top of the cube, so the
expert offsets the approach along the jaw-closing axis (found by a probe: 8 of 9 seeds lifted 10 cm that way).
"""
from __future__ import annotations

import math
import sys

import numpy as np

from teleop_il.ik import GRIPPER_CLOSED, GRIPPER_OPEN, ArmController

HOVER_Z = 0.09                      # gripper-frame height while travelling (the vertical pose reaches this high)
GRASP_Z = 0.025                     # gripper-frame height at the grasp (cube centre is at 0.02)
LIFT_Z = 0.12
JAW_OFFSET = 0.010                  # metres from the cube centre against the fixed jaw, along the closing axis
APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, LIFT_STEPS = 40, 45, 15, 30     # 20 Hz control steps per phase


def grasp_point(cube_xyz, z: float = GRASP_Z) -> np.ndarray:
    """Where the gripper frame goes to grasp a cube at `cube_xyz`: offset against the fixed jaw along the closing axis,
    which the pan joint turns with the arm (+y at home)."""
    cx, cy = float(cube_xyz[0]), float(cube_xyz[1])
    yaw = math.atan2(cy, cx)
    closing_axis = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    offset = -JAW_OFFSET * closing_axis
    return np.array([cx + offset[0], cy + offset[1], z])


class ExpertPolicy:
    """Reads only the observation (joint positions and cube pose), never the simulator's internals."""

    def __init__(self) -> None:
        self._arm = ArmController()
        self.reset()

    def reset(self) -> None:
        self._arm.reset()
        self._step = 0
        self._cube = None

    def _hand_target(self, z: float) -> np.ndarray:
        return grasp_point(self._cube, z)

    def __call__(self, obs):
        if self._cube is None:
            self._cube = np.array(obs["cube_pose"][:3], float)
        t = self._step
        self._step += 1
        if t < APPROACH_STEPS:
            target, grip = self._hand_target(HOVER_Z), GRIPPER_OPEN
        elif t < APPROACH_STEPS + DESCEND_STEPS:
            frac = (t - APPROACH_STEPS + 1) / DESCEND_STEPS
            target, grip = self._hand_target(HOVER_Z + frac * (GRASP_Z - HOVER_Z)), GRIPPER_OPEN
        elif t < APPROACH_STEPS + DESCEND_STEPS + CLOSE_STEPS:
            target, grip = self._hand_target(GRASP_Z), GRIPPER_CLOSED
        else:
            frac = min((t - APPROACH_STEPS - DESCEND_STEPS - CLOSE_STEPS + 1) / LIFT_STEPS, 1.0)
            target, grip = self._hand_target(GRASP_Z + frac * (LIFT_Z - GRASP_Z)), GRIPPER_CLOSED
        q = self._arm.track(target, 8)
        q[5] = grip
        return q


def check(policy, seeds=None, required_rate: float = 0.9) -> int:
    """Run the policy through the evaluation harness; exit status 0 only if enough seeds succeed (judged from MuJoCo state)."""
    from teleop_il.env import Env
    from teleop_il.evaluation import HELD_OUT_SEEDS, evaluate

    seeds = HELD_OUT_SEEDS if seeds is None else tuple(seeds)
    env = Env()
    result = evaluate(policy, seeds=seeds, env=env)
    ok = result.successes >= required_rate * result.n
    print(f"expert: {result.successes}/{result.n} seeds succeeded (95% CI {result.ci_low:.2f}-{result.ci_high:.2f}), "
          f"required {required_rate:.0%}: {'PASS' if ok else 'FAIL'}")
    for seed, success in zip(seeds, result.per_seed):
        if not success:
            cube = env.reset(seed)["cube_pose"]
            print(f"  failed seed {seed}: cube at x={cube[0]:.3f} y={cube[1]:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check(ExpertPolicy()))
