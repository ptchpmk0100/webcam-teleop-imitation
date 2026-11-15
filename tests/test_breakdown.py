"""Ticket 18 (slimmed): why each rollout of a policy failed, read from the MuJoCo state of the rollout."""
import numpy as np
import pytest

from teleop_il.breakdown import CLASSES, breakdown, classify
from teleop_il.dataset import record_episode
from teleop_il.env import Env
from teleop_il.evaluation import HoldPolicy, evaluate
from teleop_il.expert import ExpertPolicy
from teleop_il.policy_v2 import PolicyV2, torch_actions, train

T = 50
FAR, NEAR = np.full(T, 0.10), np.r_[np.full(T // 2, 0.10), np.full(T - T // 2, 0.005)]
OPEN, CLOSED_LATE = np.zeros(T, bool), np.r_[np.zeros(T - 10, bool), np.ones(10, bool)]
ON_TABLE, RAISED = np.zeros(T), np.r_[np.zeros(T - 5), np.full(5, 0.03)]


def test_a_successful_rollout_is_lifted():
    assert classify(NEAR, CLOSED_LATE, RAISED, success=True) == "lifted"


def test_a_gripper_that_never_came_near_the_cube_is_never_near():
    assert classify(FAR, CLOSED_LATE, ON_TABLE, success=False) == "never near"


def test_a_gripper_that_came_near_but_never_closed_there():
    assert classify(NEAR, OPEN, ON_TABLE, success=False) == "near, never closed"


def test_closing_far_from_the_cube_does_not_count_as_a_grasp_attempt():
    closed_early = np.r_[np.ones(10, bool), np.zeros(T - 10, bool)]
    assert classify(NEAR, closed_early, ON_TABLE, success=False) == "near, never closed"


def test_a_close_at_the_cube_that_did_not_raise_it_is_a_missed_grasp():
    assert classify(NEAR, CLOSED_LATE, ON_TABLE, success=False) == "closed at the cube, not raised"


def test_a_cube_raised_but_not_held_high_enough_for_long_enough_is_dropped_or_short():
    assert classify(NEAR, CLOSED_LATE, RAISED, success=False) == "raised, dropped or held too short"


def test_every_class_is_listed_once():
    assert len(CLASSES) == len(set(CLASSES)) == 5


@pytest.fixture(scope="module")
def cloned():
    env = Env()
    model = train([record_episode(env, ExpertPolicy(), seed=s, source="expert") for s in range(5)], epochs=150,
                  seed=0).model
    return lambda: PolicyV2(lambda x: torch_actions(model, x))


def test_the_lifted_count_equals_the_evaluation_harness_on_the_same_seeds(cloned):
    seeds = tuple(range(8))
    rows = breakdown(cloned(), seeds)
    assert [s for s, _ in rows] == list(seeds)
    assert [c == "lifted" for _, c in rows] == list(evaluate(cloned(), seeds=seeds).per_seed)


def test_the_do_nothing_policy_is_never_near():
    assert {c for _, c in breakdown(HoldPolicy(), range(10_000, 10_005))} == {"never near"}
