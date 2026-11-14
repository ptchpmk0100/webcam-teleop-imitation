"""Seam B: teleop core, driven by synthetic landmark clips. Expected motion comes from the documented gains."""
import numpy as np
import pytest

from helpers import PALM, Clock, fk, hand, pinhole_hand
from teleop_il.ik import ArmController
from teleop_il.teleop import Teleop

# Documented mapping (metres of end-effector motion per unit of hand motion):
#   hand left-right in the image: 0.6 m per image width, towards the user's right = arm -y  (u decreases -> y decreases)
#   hand up-down in the image:    0.5 m per image height, up = arm +z
#   hand distance (inverse depth): 0.3 m per 100 percent change, closer = arm +x
#   (gains lowered from 0.8/0.8/0.5 after Khoi's session log: the hand used 0.5 of the image height, the arm only has 0.23 m)


def settle(tp, clock, n=40, **kw):
    out = None
    for _ in range(n):
        out = tp.update(hand(clock.t(), **kw))
    return out


def engaged_teleop():
    tp, clock = Teleop(), Clock()
    base = settle(tp, clock, n=3, engage=False)           # arm sits at home before engaging
    start, _ = fk(base.joint_targets)
    settle(tp, clock, n=3)                                 # engage with the hand at the reference position
    return tp, clock, start


def test_arm_does_not_move_before_the_clutch_is_engaged():
    tp, clock = Teleop(), Clock()
    first = settle(tp, clock, n=1, engage=False, u=0.5)
    later = settle(tp, clock, n=10, engage=False, u=0.2, v=0.8)
    assert np.allclose(first.joint_targets, later.joint_targets)


def test_hand_moving_right_by_a_tenth_of_the_image_moves_the_arm_6_cm_to_its_right():
    tp, clock, start = engaged_teleop()
    out = settle(tp, clock, u=0.4)                         # u decreases = user's right
    pos, _ = fk(out.joint_targets)
    assert pos - start == pytest.approx([0.0, -0.06, 0.0], abs=0.005)


def test_hand_moving_up_by_a_tenth_of_the_image_lifts_the_arm_5_cm():
    tp, clock, start = engaged_teleop()
    out = settle(tp, clock, v=0.4)                         # v decreases = up in the image
    pos, _ = fk(out.joint_targets)
    assert pos - start == pytest.approx([0.0, 0.0, 0.05], abs=0.005)


def test_hand_20_percent_closer_moves_the_arm_6_cm_forward():
    tp, clock, start = engaged_teleop()
    out = settle(tp, clock, size=0.12)                     # closer to the camera
    pos, _ = fk(out.joint_targets)
    assert pos - start == pytest.approx([0.06, 0.0, 0.0], abs=0.005)


def test_releasing_the_clutch_freezes_the_arm_and_reengaging_does_not_jump():
    tp, clock, start = engaged_teleop()
    moved = settle(tp, clock, u=0.4)
    frozen = settle(tp, clock, n=20, engage=False, u=0.7, v=0.2)      # hand wanders, clutch released
    assert np.allclose(frozen.joint_targets, moved.joint_targets, atol=1e-6)
    again = settle(tp, clock, n=1, u=0.7, v=0.2)                      # clutch pressed at a new hand position
    assert np.allclose(fk(again.joint_targets)[0], fk(moved.joint_targets)[0], atol=0.002)


@pytest.mark.parametrize("x", [0.10, 0.15, 0.20, 0.25, 0.30])
@pytest.mark.parametrize("y", [-0.15, -0.075, 0.0, 0.075, 0.15])
def test_the_arm_reaches_the_whole_spawn_rectangle_at_table_height_pointing_down(x, y):
    arm = ArmController()
    q = arm.track([x, y, 0.03], iterations=600)
    pos, tilt = fk(q)
    assert np.linalg.norm(pos - [x, y, 0.03]) < 0.003
    assert tilt <= 12.0                                   # measured worst case 10.7 degrees at the far corners


# ---- pinch to gripper: closes below 3 cm, opens above 5 cm, stays put in between ----

OPEN, CLOSED = 1.2, -0.175                                 # gripper joint values (jaw gap about 10 cm / touching)


def gripper_after(tp, clock, pinch_m):
    return tp.update(hand(clock.t(), pinch_m=pinch_m)).joint_targets[5]


def test_pinch_closes_below_3_cm_and_opens_above_5_cm():
    tp, clock = Teleop(), Clock()
    assert gripper_after(tp, clock, 0.08) == pytest.approx(OPEN)
    assert gripper_after(tp, clock, 0.04) == pytest.approx(OPEN)      # between the thresholds: still open
    assert gripper_after(tp, clock, 0.025) == pytest.approx(CLOSED)
    assert gripper_after(tp, clock, 0.04) == pytest.approx(CLOSED)    # between the thresholds: still closed
    assert gripper_after(tp, clock, 0.06) == pytest.approx(OPEN)


def test_pinch_does_not_flicker_around_the_closing_threshold():
    tp, clock = Teleop(), Clock()
    gripper_after(tp, clock, 0.08)
    seen = [gripper_after(tp, clock, p) for p in (0.029, 0.031, 0.029, 0.031, 0.029, 0.031)]
    assert seen == [pytest.approx(CLOSED)] * 6


# ---- tracking loss: hold the arm; flag the episode when the hand is gone for more than 0.3 s ----

def drop_hand(tp, clock, frames, engage=True):
    from helpers import no_hand
    out = None
    for _ in range(frames):
        out = tp.update(no_hand(clock.t(), engage=engage))
    return out


def test_a_short_dropout_holds_the_arm_and_does_not_flag_the_episode():
    tp, clock, _ = engaged_teleop()
    before = settle(tp, clock, u=0.4)
    out = drop_hand(tp, clock, frames=8)                   # 0.27 s
    assert np.array_equal(out.joint_targets, before.joint_targets)
    assert out.tracking_lost is False and out.flagged is False


def test_a_dropout_longer_than_0_3_s_flags_the_episode_and_keeps_holding():
    tp, clock, _ = engaged_teleop()
    before = settle(tp, clock, u=0.4)
    out = drop_hand(tp, clock, frames=11)                  # 0.37 s
    assert out.tracking_lost is True and out.flagged is True
    assert np.array_equal(out.joint_targets, before.joint_targets)


def test_when_the_hand_returns_the_arm_does_not_jump_and_the_flag_stays_until_reset():
    tp, clock, _ = engaged_teleop()
    before = settle(tp, clock, u=0.4)
    drop_hand(tp, clock, frames=11)
    back = tp.update(hand(clock.t(), u=0.8, v=0.2))        # hand reappears somewhere else
    assert np.allclose(fk(back.joint_targets)[0], fk(before.joint_targets)[0], atol=0.002)
    assert back.tracking_lost is False and back.flagged is True
    tp.reset()
    assert settle(tp, clock, n=1, engage=False).flagged is False


def test_a_dropout_while_the_clutch_is_released_is_not_a_problem():
    tp, clock, _ = engaged_teleop()
    settle(tp, clock, n=3, engage=False)
    out = drop_hand(tp, clock, frames=20, engage=False)
    assert out.flagged is False and out.tracking_lost is False


# ---- smoothing: depth and pixel noise must not shake the arm ----

def test_hand_jitter_is_smoothed_to_less_than_half():
    rng = np.random.default_rng(0)
    tp, clock, _ = engaged_teleop()
    ys, zs = [], []
    for i in range(120):
        out = tp.update(hand(clock.t(), u=0.5 + rng.normal(0, 0.004), v=0.5 + rng.normal(0, 0.004)))
        if i >= 60:
            pos, _ = fk(out.joint_targets)
            ys.append(pos[1]); zs.append(pos[2])
    raw_sigma = 0.6 * 0.004                                # unfiltered arm jitter would be 2.4 mm
    assert np.std(ys) < 0.5 * raw_sigma and np.std(zs) < 0.5 * raw_sigma


def test_a_steady_hand_motion_is_followed_with_little_lag():
    tp, clock, start = engaged_teleop()
    for i in range(1, 31):                                 # u from 0.5 to 0.4 over one second
        out = tp.update(hand(clock.t(), u=0.5 - 0.1 * i / 30))
    pos, _ = fk(out.joint_targets)
    assert abs((pos - start)[1] - (-0.06)) < 0.015         # within 1.5 cm of where the hand now is


def test_the_flag_survives_releasing_the_clutch_after_a_long_dropout():
    tp, clock, _ = engaged_teleop()
    drop_hand(tp, clock, frames=11)                        # flagged while the clutch is held
    out = drop_hand(tp, clock, frames=5, engage=False)     # clutch released while the hand is still missing
    assert out.flagged is True


def test_a_hand_far_outside_the_range_keeps_the_arm_inside_the_workspace_box():
    tp, clock, _ = engaged_teleop()
    out = settle(tp, clock, n=60, u=0.1, v=0.0)            # 0.4 of the image left and up: asks for y = -0.32 m, z = +0.4 m
    pos, _ = fk(out.joint_targets)
    assert -0.18 <= pos[1] <= 0.18                         # box is y in [-0.17, 0.17]
    assert pos[2] <= 0.26                                  # box is z up to 0.25


# ---- reported by Khoi at G5: the arm lagged the hand and forward/back moved when the hand only tilted ----

def test_a_quick_hand_move_is_followed_within_100_ms():
    tp, clock, start = engaged_teleop()
    for _ in range(3):                                     # three frames = 100 ms after the hand jumps 0.1 image widths
        out = tp.update(hand(clock.t(), u=0.4))
    pos, _ = fk(out.joint_targets)
    assert (pos - start)[1] < -0.9 * 0.06                  # at least 90 percent of the 6 cm


def pinhole_engaged(**kw):
    tp, clock = Teleop(), Clock()
    for _ in range(3):                                     # clutch released, hand already at its place
        tp.update(pinhole_hand(clock.t(), engage=False, **kw))
    out = None
    for _ in range(40):
        out = tp.update(pinhole_hand(clock.t(), **kw))
    return tp, clock, fk(out.joint_targets)[0]


def test_moving_the_flat_hand_from_50_cm_to_45_cm_from_the_camera_moves_the_arm_3_cm_forward():
    tp, clock, start = pinhole_engaged(distance=0.5)
    for _ in range(40):
        out = tp.update(pinhole_hand(clock.t(), distance=0.45))
    pos, _ = fk(out.joint_targets)
    assert (pos - start)[0] == pytest.approx(0.0333, abs=0.008)    # 0.5 / 0.45 - 1 = 11.1 percent, times 0.3 m


@pytest.mark.parametrize("pitch,yaw", [(45, 0), (0, 45), (-40, 30), (-45, 0)])
def test_tilting_the_hand_moves_the_arm_only_as_far_as_the_palm_really_moved_towards_the_camera(pitch, yaw):
    """A tilt about the wrist brings the palm centre slightly nearer or further; apart from that the arm must not move.
    The old cue (image length of one segment) wrongly moved the arm 7 to 14 cm for these tilts."""
    tp, clock, start = pinhole_engaged(distance=0.5)
    for _ in range(60):
        out = tp.update(pinhole_hand(clock.t(), distance=0.5, pitch_deg=pitch, yaw_deg=yaw))
    pos, _ = fk(out.joint_targets)
    a, b = np.radians(pitch), np.radians(yaw)               # palm centre depth from the 3D model, not from the code
    rot = (np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
           @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]))
    centre = 0.5 + np.mean([(rot @ np.array(p))[2] for p in PALM.values()])
    assert (pos - start)[0] == pytest.approx(0.3 * (0.5 / centre - 1), abs=0.015)


# ---- gripper key: the pinch can be hidden from the camera, so a key closes the gripper too ----

def test_the_gripper_key_closes_the_gripper_even_when_the_pinch_is_open():
    tp, clock = Teleop(), Clock()
    out = tp.update(hand(clock.t(), pinch_m=0.08, close_gripper=True))
    assert out.joint_targets[5] == pytest.approx(CLOSED)


def test_releasing_the_gripper_key_opens_it_again_when_the_pinch_is_open():
    tp, clock = Teleop(), Clock()
    tp.update(hand(clock.t(), pinch_m=0.08, close_gripper=True))
    out = tp.update(hand(clock.t(), pinch_m=0.08, close_gripper=False))
    assert out.joint_targets[5] == pytest.approx(OPEN)


def test_a_pinch_still_closes_the_gripper_without_the_key():
    tp, clock = Teleop(), Clock()
    out = tp.update(hand(clock.t(), pinch_m=0.02, close_gripper=False))
    assert out.joint_targets[5] == pytest.approx(CLOSED)


def test_the_output_reports_the_pinch_gap_and_the_target_for_the_live_overlay_and_log():
    tp, clock, _ = engaged_teleop()
    out = tp.update(hand(clock.t(), pinch_m=0.037, u=0.4))
    assert out.info["pinch_gap_m"] == pytest.approx(0.037)
    assert out.info["target"].shape == (3,) and out.info["raw"].shape == (3,) and out.info["filtered"].shape == (3,)


# ---- from Khoi's session log: depth spikes, a target pinned at the workspace edge, and the near-base region ----

def test_a_two_frame_depth_spike_does_not_move_the_arm_forward():
    """The log showed the raw depth cue jumping 5 -> 11 -> 15 -> 5 within three frames (palm edge-on to the camera)."""
    tp, clock, start = pinhole_engaged(distance=0.5)
    before, worst = None, 0.0
    for i in range(12):
        frame = pinhole_hand(clock.t())
        if i in (5, 6):                                    # two frames where the world palm collapses to a fifth of its size
            frame = type(frame)(t=frame.t, image_landmarks=frame.image_landmarks, engage=True,
                                world_landmarks=frame.world_landmarks * 0.2)
        out = tp.update(frame)
        if i == 4:
            before = fk(out.joint_targets)[0]
        elif i > 4:
            worst = max(worst, abs(fk(out.joint_targets)[0][0] - before[0]))
    assert worst < 0.01


def test_a_hand_that_overshoots_the_arms_range_can_come_back_without_a_dead_zone():
    tp, clock, start = engaged_teleop()
    settle(tp, clock, n=40, u=-0.2)                        # far past the right edge of the arm's range (y box is -0.17)
    out = settle(tp, clock, n=40, u=-0.1)                  # hand returns 0.1 image widths: the arm must follow at once
    pos, _ = fk(out.joint_targets)
    assert pos[1] == pytest.approx(-0.17 + 0.6 * 0.1, abs=0.01)


def test_the_arm_is_never_asked_to_go_closer_to_its_base_than_the_expert_can_grasp():
    tp, clock, _ = engaged_teleop()
    out = settle(tp, clock, n=60, size=0.03)               # hand much further from the camera: asks for a far-back x
    pos, _ = fk(out.joint_targets)
    assert pos[0] >= 0.149                                 # expert-proven spawn starts at 0.17; teleop box starts at 0.15
