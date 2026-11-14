"""Seam A: the Env interface. Expected values are literals from the PRD, not recomputed from the code."""
import numpy as np
import pytest

from teleop_il.env import Env


@pytest.fixture(scope="module")
def env():
    return Env()


def cube_xy(obs):
    return np.array(obs["cube_pose"][:2])


def test_same_seed_gives_same_cube_position(env):
    a = cube_xy(env.reset(seed=7))
    b = cube_xy(env.reset(seed=7))
    assert np.array_equal(a, b)


def test_different_seeds_give_different_cube_positions(env):
    a = cube_xy(env.reset(seed=1))
    b = cube_xy(env.reset(seed=2))
    assert not np.array_equal(a, b)


def test_cube_spawns_inside_the_agreed_rectangle(env):
    for seed in range(50):
        x, y = cube_xy(env.reset(seed=seed))
        assert 0.10 <= x <= 0.30 and -0.15 <= y <= 0.15


def test_cube_rests_on_the_table_after_reset(env):
    obs = env.reset(seed=3)
    for _ in range(10):
        obs = env.step(obs["joint_pos"])
    assert abs(obs["cube_pose"][2] - 0.02) < 0.002   # centre of a 4 cm cube resting on the floor


def test_one_step_advances_time_by_one_twentieth_of_a_second(env):
    obs0 = env.reset(seed=4)
    obs1 = env.step(obs0["joint_pos"])
    assert obs1["time"] - obs0["time"] == pytest.approx(0.05, abs=1e-9)



# ---- success check: cube bottom more than 5 cm above the table, continuously for 1 s (20 steps at 20 Hz) ----

def hold_cube_up(env, obs, clearance, steps):
    """Keep the cube `clearance` metres above the table for `steps` control steps (kinematic hold).
    The cube is held in a table corner, clear of the arm's hover pose."""
    for _ in range(steps):
        env.place_cube([0.12, 0.14, 0.02 + clearance])
        obs = env.step(obs["joint_pos"])
    return obs


def test_a_cube_resting_on_the_table_is_not_a_success(env):
    obs = env.reset(seed=5)
    for _ in range(40):                                  # 2 s
        obs = env.step(obs["joint_pos"])
    assert env.lift_height() == pytest.approx(0.0, abs=0.002)
    assert env.success() is False


def test_lift_height_is_the_gap_between_cube_bottom_and_table(env):
    env.reset(seed=5)
    env.place_cube([0.2, 0.0, 0.07])
    assert env.lift_height() == pytest.approx(0.05, abs=1e-6)


def test_cube_held_above_5_cm_for_a_full_second_is_a_success(env):
    obs = env.reset(seed=6)
    obs = hold_cube_up(env, obs, clearance=0.10, steps=19)
    assert env.success() is False                        # 0.95 s is not enough
    hold_cube_up(env, obs, clearance=0.10, steps=1)
    assert env.success() is True                         # exactly 1.0 s


# The cube is placed, then physics runs for 0.05 s before the check, so it falls 0.5*9.81*0.05**2 = 1.2 cm first.
# These two cases straddle the 5 cm line after that fall: 6.6 cm placed ends near 5.4 cm, 5.6 cm placed near 4.4 cm.

def test_a_cube_held_just_above_5_cm_is_a_success(env):
    obs = env.reset(seed=6)
    hold_cube_up(env, obs, clearance=0.066, steps=20)
    assert env.success() is True


def test_a_cube_held_just_below_5_cm_is_not_a_success(env):
    obs = env.reset(seed=6)
    hold_cube_up(env, obs, clearance=0.056, steps=40)
    assert env.success() is False


def test_an_interrupted_hold_does_not_count(env):
    obs = env.reset(seed=6)
    obs = hold_cube_up(env, obs, clearance=0.10, steps=18)
    obs = hold_cube_up(env, obs, clearance=0.0, steps=1)   # cube back on the table
    obs = hold_cube_up(env, obs, clearance=0.10, steps=19)
    assert env.success() is False                        # 19 consecutive steps after the break
    hold_cube_up(env, obs, clearance=0.10, steps=1)
    assert env.success() is True


def test_success_stays_true_once_reached_and_clears_on_reset(env):
    obs = env.reset(seed=6)
    obs = hold_cube_up(env, obs, clearance=0.10, steps=20)
    for _ in range(10):                                  # cube falls back down
        obs = env.step(obs["joint_pos"])
    assert env.success() is True
    env.reset(seed=6)
    assert env.success() is False


# ---- snapshot and restore ----

def run(env, obs, n):
    for _ in range(n):
        obs = env.step(obs["joint_pos"])
    return obs


def test_restoring_a_snapshot_replays_identically(env):
    obs = run(env, env.reset(seed=8), 5)
    snap = env.snapshot()
    first = run(env, obs, 10)
    env.restore(snap)
    second = run(env, obs, 10)
    assert np.array_equal(first["joint_pos"], second["joint_pos"])
    assert np.array_equal(first["cube_pose"], second["cube_pose"])
    assert first["time"] == second["time"]


def test_a_snapshot_also_restores_the_success_progress(env):
    obs = hold_cube_up(env, env.reset(seed=9), clearance=0.10, steps=19)
    snap = env.snapshot()
    hold_cube_up(env, obs, clearance=0.10, steps=1)
    assert env.success() is True
    env.restore(snap)
    assert env.success() is False
    hold_cube_up(env, obs, clearance=0.10, steps=1)
    assert env.success() is True


# ---- commanded joint targets move the arm ----

def test_the_arm_follows_a_joint_target_within_one_second(env):
    obs = env.reset(seed=10)
    target = obs["joint_pos"].copy()
    target[0] = 0.5                                      # shoulder pan, radians
    target[5] = 0.2                                      # gripper
    for _ in range(20):
        obs = env.step(target)
    assert obs["joint_pos"][0] == pytest.approx(0.5, abs=0.03)
    assert obs["joint_pos"][5] == pytest.approx(0.2, abs=0.03)


def test_targets_outside_the_joint_range_are_clipped(env):
    obs = env.reset(seed=10)
    target = obs["joint_pos"].copy()
    target[0] = 10.0                                     # far beyond the 1.92 rad limit
    for _ in range(20):
        obs = env.step(target)
    assert obs["joint_pos"][0] <= 1.92 + 1e-6
