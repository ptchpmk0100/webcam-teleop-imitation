"""Seam A, evaluation harness. Expected values are literals (Wilson interval tables, PRD numbers)."""
import numpy as np
import pytest

from teleop_il.env import Env
from teleop_il.evaluation import HELD_OUT_SEEDS, MAX_STEPS, HoldPolicy, RandomPolicy, evaluate, null_checks, wilson_interval


class StubEnv:
    """Stands in for Env: success on odd seeds. Counts steps so the episode length can be checked."""

    def __init__(self, succeed_when=lambda seed: seed % 2 == 1, success_after=1):
        self.succeed_when, self.success_after = succeed_when, success_after
        self.steps_taken = 0
        self._seed = None
        self._t = 0

    action_bounds = (np.zeros(6), np.ones(6))

    def reset(self, seed):
        self._seed, self._t = seed, 0
        self.steps_taken = 0
        return {"joint_pos": np.zeros(6), "cube_pose": np.zeros(7), "time": 0.0}

    def step(self, action):
        self._t += 1
        self.steps_taken += 1
        return {"joint_pos": np.zeros(6), "cube_pose": np.zeros(7), "time": self._t * 0.05}

    def success(self):
        return self.succeed_when(self._seed) and self._t >= self.success_after


class BrokenEnv(Env):
    """A broken harness input: always reports success. null_checks must catch this."""

    def success(self):
        return True


def zero_policy(obs):
    return np.zeros(6)


def test_wilson_interval_for_no_successes_in_100():
    lo, hi = wilson_interval(0, 100)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.0370, abs=0.0005)


def test_wilson_interval_for_half_in_100():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.404, abs=0.001)
    assert hi == pytest.approx(0.596, abs=0.001)


def test_wilson_interval_for_all_in_100_reaches_one():
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo == pytest.approx(0.963, abs=0.001)


def test_there_are_100_distinct_held_out_seeds_apart_from_demo_seeds():
    assert len(HELD_OUT_SEEDS) == 100 and len(set(HELD_OUT_SEEDS)) == 100
    assert min(HELD_OUT_SEEDS) >= 10_000          # demos and training use seeds below 10000


def test_evaluate_counts_successes_per_seed():
    res = evaluate(zero_policy, seeds=[1, 2, 3, 4, 5], env=StubEnv())
    assert res.successes == 3 and res.n == 5
    assert res.rate == pytest.approx(0.6)
    assert res.per_seed == (True, False, True, False, True)


def test_an_episode_without_success_runs_the_full_15_seconds():
    env = StubEnv(succeed_when=lambda seed: False)
    evaluate(zero_policy, seeds=[2], env=env)
    assert MAX_STEPS == 300
    assert env.steps_taken == 300


def test_an_episode_ends_as_soon_as_it_succeeds():
    env = StubEnv(succeed_when=lambda seed: True, success_after=7)
    evaluate(zero_policy, seeds=[1], env=env)
    assert env.steps_taken == 7


def test_evaluation_is_deterministic_for_a_fixed_seed_list():
    env = Env()
    seeds = HELD_OUT_SEEDS[:5]
    policy = RandomPolicy(*env.action_bounds, seed=0)
    a = evaluate(policy, seeds=seeds, env=env)
    b = evaluate(policy, seeds=seeds, env=env)
    assert a == b


def test_null_checks_pass_on_the_real_env():
    assert null_checks(Env()) == []


def test_null_checks_fail_on_an_env_that_always_reports_success():
    problems = null_checks(BrokenEnv())
    assert problems, "a harness that always reports success must be caught"


def test_null_checks_allow_a_random_policy_at_most_2_successes_in_100():
    two = StubEnv(succeed_when=lambda seed: seed in (10_000, 10_001))
    problems = null_checks(two)
    assert not any("random" in p for p in problems)      # 2 of 100 is allowed for random actions
    assert any("do-nothing" in p for p in problems)      # but never for the do-nothing policy


def test_null_checks_flag_a_random_policy_with_3_successes_in_100():
    three = StubEnv(succeed_when=lambda seed: seed in (10_000, 10_001, 10_002))
    assert any("random" in p for p in null_checks(three))
