"""Grasp assist (shared autonomy, redesign grill Q2/Q13 and the Q14 fallback), tested in the real physics.

The human brings the gripper near the cube and closes it; the assist aligns, descends, closes, then hands back and the
human lifts. Success is read from MuJoCo state (env.success()), never from the assist's own report.
"""
import math

import numpy as np
import pytest

from helpers import hand
from teleop_il.assist import AssistedTeleop
from teleop_il.env import CUBE_HALF, Env
from teleop_il.teleop import HandFrame, Teleop

DT = 0.05                                   # one frame per 20 Hz control step, so env.step keeps pace exactly
HOME = np.array([0.2, 0.0, 0.09])           # where the gripper starts (documented home pose)


class Session:
    """A fake human: puts the hand where it maps to a wanted gripper position (gains 0.6 y, 0.5 z; keys for x)."""

    def __init__(self, cube_xy, seed=0):
        self.env = Env()
        self.cube = self.env.reset(seed)["cube_pose"][:3]      # cube_xy None: keep the seed's spawn position
        if cube_xy is not None:
            self.env.place_cube([*cube_xy, CUBE_HALF])
        self.at = AssistedTeleop(Teleop(), ee=self.env.ee_position, cube=self.env.cube_position)
        self.t = 0.0
        self.outputs = []
        self.hold(HOME, False, 0.3)                            # engage with the hand at the reference, then move

    def frame(self, xyz, closed):
        x, y, z = xyz
        u, v = 0.5 + (y - HOME[1]) / 0.6, 0.5 - (z - HOME[2]) / 0.5
        f = hand(self.t, u=u, v=v, close_gripper=closed)
        return HandFrame(f.t, f.image_landmarks, f.world_landmarks, True, closed, forward_m=x - HOME[0])

    def hold(self, xyz, closed, seconds):
        for _ in range(round(seconds / DT)):
            self.t += DT
            out = self.at.update(self.frame(xyz, closed))
            self.outputs.append(out)
            self.env.step(out.joint_targets)
        return self.outputs[-1]

    def fired(self):
        return self.outputs[-1].info["assist_fired"]


def pick(s, near, hover_z=0.07):
    """Hover near the cube, close, wait for the assist, then lift the hand 10 cm and hold it there."""
    s.hold([*near, hover_z], False, 1.5)
    s.hold([*near, hover_z], True, 2.5)
    s.hold([*near, hover_z + 0.10], True, 2.5)


def test_closing_within_5_cm_of_the_cube_grasps_it_and_raising_the_hand_lifts_it():
    s = Session(cube_xy=(0.23, 0.03))
    pick(s, near=(0.20, 0.0))                         # 3.6 cm from the cube centre, never aligned by the human
    assert s.fired() == 1
    assert s.env.success()


def test_closing_8_cm_away_does_not_fire_the_assist_and_the_cube_stays():
    s = Session(cube_xy=(0.28, 0.0))
    pick(s, near=(0.20, 0.0))
    assert s.fired() == 0
    assert s.env.lift_height() < 0.005


def test_moving_into_the_zone_with_the_gripper_already_closed_fires_the_assist():
    s = Session(cube_xy=(0.27, 0.0))                   # 7 cm from where the gripper is when it closes
    s.hold([0.20, 0.10, 0.07], True, 2.0)             # closed, then moving further away (more than 10 cm)
    assert s.fired() == 0
    s.hold([0.25, 0.01, 0.07], True, 2.5)             # still closed, now within the zone
    s.hold([0.25, 0.01, 0.17], True, 2.5)
    assert s.fired() == 1
    assert s.env.success()


def test_the_assist_fires_once_per_close_and_again_after_the_gripper_is_opened():
    s = Session(cube_xy=(0.23, 0.0))
    s.hold([0.21, 0.0, 0.07], True, 4.0)              # assist runs, hands back, gripper still closed near the cube
    assert s.fired() == 1
    s.hold([0.21, 0.0, 0.07], False, 1.5)             # open: the cube is dropped, the assist re-arms
    s.hold([0.21, 0.0, 0.07], True, 3.0)
    assert s.fired() == 2


def test_handing_control_back_does_not_make_the_arm_jump():
    s = Session(cube_xy=(0.23, 0.02))
    pick(s, near=(0.20, 0.0))
    active = [o.info["assist"] == "active" for o in s.outputs]
    end = max(i for i, a in enumerate(active) if a)
    q = np.array([o.joint_targets[:5] for o in s.outputs])
    jump = np.abs(q[end + 1] - q[end]).max()
    assert jump < 0.05                                # radians in one 50 ms frame


def test_the_human_has_no_control_while_the_assist_runs():
    s = Session(cube_xy=(0.23, 0.0))
    s.hold([0.21, 0.0, 0.07], False, 1.0)
    s.hold([0.21, 0.0, 0.07], True, 0.2)
    assert s.outputs[-1].info["assist"] == "active"
    before = s.outputs[-1].joint_targets.copy()
    s.hold([0.21, 0.12, 0.20], True, 0.1)             # a big hand move during the assist is ignored
    assert s.outputs[-1].info["assist"] == "active"
    assert np.abs(s.outputs[-1].joint_targets[:5] - before[:5]).max() < 0.3


def test_the_output_says_when_the_gripper_is_in_the_zone_so_the_screen_can_show_it():
    s = Session(cube_xy=(0.23, 0.0))
    s.hold([0.20, 0.10, 0.07], False, 1.5)
    assert s.outputs[-1].info["assist"] == "off"
    s.hold([0.22, 0.0, 0.07], False, 1.5)
    assert s.outputs[-1].info["assist"] == "ready"
    assert s.outputs[-1].info["grasp_distance_m"] < 0.05


@pytest.mark.parametrize("seed", range(10))
def test_a_sloppy_approach_anywhere_in_the_spawn_rectangle_still_lifts_the_cube(seed):
    """Hover up to 4.5 cm off in any direction, gripper 5 to 10 cm above the table (the cube top is at 4), close, raise."""
    rng = np.random.default_rng(1000 + seed)
    s = Session(cube_xy=None, seed=seed)
    angle, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.0, 0.045)
    near = (s.cube[0] + r * math.cos(angle), s.cube[1] + r * math.sin(angle))
    pick(s, near=near, hover_z=float(rng.uniform(0.05, 0.10)))
    assert s.fired() == 1
    assert s.env.success()
