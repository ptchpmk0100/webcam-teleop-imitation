"""Seam A for the scripted expert: it is judged only by the evaluation harness, from MuJoCo state."""
import pytest

from teleop_il.env import Env
from teleop_il.evaluation import HELD_OUT_SEEDS, HoldPolicy, evaluate
from teleop_il.expert import ExpertPolicy, check


def test_the_expert_picks_and_lifts_the_cube_on_most_of_the_first_held_out_seeds():
    result = evaluate(ExpertPolicy(), seeds=HELD_OUT_SEEDS[:12])
    assert result.successes >= 11, f"failed seeds: {[s for s, ok in zip(HELD_OUT_SEEDS[:12], result.per_seed) if not ok]}"


def test_the_expert_is_deterministic_for_a_fixed_seed_list():
    seeds = HELD_OUT_SEEDS[:4]
    assert evaluate(ExpertPolicy(), seeds=seeds).per_seed == evaluate(ExpertPolicy(), seeds=seeds).per_seed


def test_the_expert_check_fails_loudly_for_a_policy_that_does_nothing(capsys):
    assert check(HoldPolicy(), seeds=HELD_OUT_SEEDS[:5], required_rate=0.9) == 1
    assert "FAIL" in capsys.readouterr().out


def test_the_expert_check_passes_and_names_the_rate_for_a_good_expert(capsys):
    assert check(ExpertPolicy(), seeds=HELD_OUT_SEEDS[:12], required_rate=0.9) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "/12" in out


def test_the_check_lists_failed_seeds_with_the_cube_position_so_the_spawn_rectangle_can_be_tightened(capsys):
    check(HoldPolicy(), seeds=HELD_OUT_SEEDS[:3], required_rate=0.9)
    out = capsys.readouterr().out
    assert f"seed {HELD_OUT_SEEDS[0]}" in out and "cube at x=" in out
