"""Evaluation harness (seam A): run any policy over fixed held-out seeds and score it from MuJoCo state."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HELD_OUT_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_100))   # 100 fixed seeds; demos and training use seeds below 10000
# PRD-v2: every experiment and every choice between variants is scored here; the held-out seeds only at the named
# checkpoints (tickets 21, 22, 23, 25), so choosing the best variant does not flatter the headline number.
DEV_SEEDS: tuple[int, ...] = tuple(range(20_000, 20_100))
MAX_STEPS = 300                                                   # 15 s at 20 Hz
NULL_HOLD_MAX_SUCCESSES = 0                                       # do-nothing must score 0 of 100
NULL_RANDOM_MAX_SUCCESSES = 2                                     # random actions at most 2 of 100


@dataclass(frozen=True)
class EvalResult:
    successes: int
    n: int
    rate: float
    ci_low: float
    ci_high: float
    per_seed: tuple[bool, ...]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 percent Wilson score interval for a binomial proportion."""
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(policy, seeds=None, env=None) -> EvalResult:
    """Run `policy` (callable obs -> 6 joint targets) once per seed; success is read from the env's MuJoCo state."""
    from teleop_il.env import Env

    seeds = HELD_OUT_SEEDS if seeds is None else tuple(seeds)
    env = Env() if env is None else env
    outcomes = []
    for seed in seeds:
        obs = env.reset(seed)
        if hasattr(policy, "reset"):
            policy.reset()
        for _ in range(MAX_STEPS):
            obs = env.step(policy(obs))
            if env.success():
                break
        outcomes.append(bool(env.success()))
    n, k = len(outcomes), sum(outcomes)
    lo, hi = wilson_interval(k, n)
    return EvalResult(successes=k, n=n, rate=k / n, ci_low=lo, ci_high=hi, per_seed=tuple(outcomes))


class HoldPolicy:
    """Do nothing: keep the arm at the pose it started the episode in."""

    def __init__(self) -> None:
        self._start = None

    def reset(self) -> None:
        self._start = None

    def __call__(self, obs):
        if self._start is None:
            self._start = np.array(obs["joint_pos"])
        return self._start


class RandomPolicy:
    """Uniform random joint targets within the joint limits. Reseeded on reset(), so evaluation is deterministic."""

    def __init__(self, low, high, seed: int = 0) -> None:
        self._low, self._high, self._seed = np.asarray(low), np.asarray(high), seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def __call__(self, obs):
        return self._rng.uniform(self._low, self._high)


def null_checks(env=None) -> list[str]:
    """Return a list of problems; empty means the harness is trustworthy (do-nothing 0/100, random at most 2/100)."""
    from teleop_il.env import Env

    env = Env() if env is None else env
    problems = []
    hold = evaluate(HoldPolicy(), env=env)
    if hold.successes > NULL_HOLD_MAX_SUCCESSES:
        problems.append(f"do-nothing policy succeeded {hold.successes}/{hold.n} (allowed {NULL_HOLD_MAX_SUCCESSES})")
    rand = evaluate(RandomPolicy(*env.action_bounds, seed=0), env=env)
    if rand.successes > NULL_RANDOM_MAX_SUCCESSES:
        problems.append(f"random policy succeeded {rand.successes}/{rand.n} (allowed {NULL_RANDOM_MAX_SUCCESSES})")
    return problems


if __name__ == "__main__":
    import sys

    found = null_checks()
    print("null checks: " + ("PASS" if not found else "FAIL: " + "; ".join(found)))
    sys.exit(1 if found else 0)
