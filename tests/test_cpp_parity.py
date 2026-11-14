"""Ticket 13: the C++ rollout, linked against the same MuJoCo build, matches the Python physics on the same sequences."""
import numpy as np
import pytest

from teleop_il.cpp_parity import Outcome, build, compare, run_cpp, run_python, sequences

SEEDS = (10_000, 10_001, 10_002)


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    return build(tmp_path_factory.mktemp("build"))


@pytest.fixture(scope="module")
def seqs():
    return sequences(expert_seeds=SEEDS, hold_seeds=SEEDS[:1], wild_seeds=(10_010, 10_011), relift_seeds=(10_003,))


def test_cpp_and_python_agree_on_cube_height_and_success_for_the_same_sequences(exe, seqs, tmp_path):
    py = run_python(seqs)
    cpp = run_cpp(exe, seqs, tmp_path / "run")
    assert compare(py, cpp, required=len(seqs)) == []
    assert [o.success for o in py] == [True] * len(SEEDS) + [False] * 4   # expert lifts; hold, wild, relift do not
    assert max(abs(a.cube_z - b.cube_z) for a, b in zip(py, cpp)) < 0.005


def test_a_cpp_rollout_with_different_physics_is_caught(exe, seqs, tmp_path):
    """9 physics steps per control step instead of 10: the comparison must fail loudly."""
    cpp = run_cpp(exe, seqs, tmp_path / "run", substeps=9)
    assert compare(run_python(seqs), cpp, required=len(seqs)) != []


def test_compare_reports_height_and_success_mismatches_by_seed():
    py = [Outcome(1, True, 0.12, 0.10), Outcome(2, False, 0.02, 0.0)]
    cpp = [Outcome(1, True, 0.13, 0.11), Outcome(2, True, 0.02, 0.0)]
    problems = compare(py, cpp, required=2)
    assert any("seed 1" in p and "height" in p for p in problems)
    assert any("success" in p for p in problems)


def test_compare_refuses_results_for_different_seeds():
    assert compare([Outcome(1, True, 0.1, 0.1)], [Outcome(2, True, 0.1, 0.1)], required=1) != []


def test_the_run_folder_is_new_and_holds_the_scene_inputs_and_results(exe, seqs, tmp_path):
    run_cpp(exe, seqs, tmp_path / "run")
    assert {p.name for p in (tmp_path / "run").iterdir()} >= {"scene.mjb", "sequences.txt", "cpp_results.txt"}
    with pytest.raises(FileExistsError):
        run_cpp(exe, seqs, tmp_path / "run")


def test_sequences_carry_the_cube_position_of_each_seed():
    from teleop_il.env import Env
    s = sequences(expert_seeds=(10_005,), hold_seeds=())
    assert np.allclose(s[0].cube_xy, Env().reset(10_005)["cube_pose"][:2])


def test_two_short_lifts_add_up_to_more_than_a_second_above_the_threshold_but_do_not_count(seqs):
    from teleop_il.env import Env
    relift = seqs[-1]
    env = Env(); env.reset(relift.seed)
    above = 0
    for a in relift.actions:
        env.step(a)
        above += env.lift_height() > 0.05
    assert above >= 20 and not env.success()


def test_wild_sequences_ask_for_joint_targets_beyond_the_limits(seqs):
    from teleop_il.env import Env
    lo, hi = Env().action_bounds
    wild = np.vstack([s.actions for s in seqs[len(SEEDS) + 1:len(SEEDS) + 3]])
    assert (wild < lo).any() and (wild > hi).any()
