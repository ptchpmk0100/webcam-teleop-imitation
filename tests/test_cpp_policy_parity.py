"""Ticket 14: the C++ executable runs the ONNX policy with ONNX Runtime and agrees with PyTorch and with Python."""
import numpy as np
import pytest

from teleop_il.cpp_parity import build
from teleop_il.cpp_policy_parity import cpp_infer, cpp_policy_rollout, python_policy_rollout
from teleop_il.dataset import record_episode
from teleop_il.env import Env
from teleop_il.expert import ExpertPolicy
from teleop_il.policy import OnnxPolicy, episode_observations, export_onnx, torch_actions, train

SEEDS = tuple(range(10_000, 10_008))


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    folder = build(tmp_path_factory.mktemp("build"))
    return folder.parent / "policy_rollout"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    env = Env()
    # 60 episodes: with 20 the policy lifted none of SEEDS, and the comparison must include lifts. Training is seeded
    # but not bit-identical across CPUs (on the CI machine training seed 0 lifted none), so take the first training
    # seed whose policy lifts at least one of SEEDS; which policy is compared does not matter for parity.
    episodes = [record_episode(env, ExpertPolicy(), seed=s, source="expert") for s in range(60)]
    for i, training_seed in enumerate(range(4)):
        result = train(episodes, epochs=150, seed=training_seed)
        path = tmp_path_factory.mktemp(f"onnx{i}") / "policy.onnx"
        export_onnx(result.model, path)
        if any(o.success for o in python_policy_rollout(OnnxPolicy(path), SEEDS)):
            break
    return result.model, path, np.vstack([episode_observations(e) for e in episodes[:3]])


def test_cpp_onnx_runtime_matches_pytorch_within_1e_4_on_recorded_observations(exe, trained, tmp_path):
    model, onnx_file, x = trained
    out = cpp_infer(exe, onnx_file, x, tmp_path / "run")
    assert out.shape == (len(x), 10, 6)
    assert np.abs(out - torch_actions(model, x)).max() <= 1e-4


def test_cpp_and_python_policy_rollouts_agree_on_success_and_cube_height(exe, trained, tmp_path):
    _, onnx_file, _ = trained
    py = python_policy_rollout(OnnxPolicy(onnx_file), SEEDS)
    cpp = cpp_policy_rollout(exe, onnx_file, SEEDS, tmp_path / "run")
    assert [o.seed for o in cpp] == list(SEEDS)
    assert [o.success for o in py] == [o.success for o in cpp]
    assert max(abs(a.cube_z - b.cube_z) for a, b in zip(py, cpp)) < 0.005
    assert any(o.success for o in py)                             # the comparison includes lifts, not only failures


def test_the_cpp_do_nothing_policy_never_lifts(exe, tmp_path):
    out = cpp_policy_rollout(exe, "hold", SEEDS, tmp_path / "run")
    assert not any(o.success for o in out)
    assert all(abs(o.cube_z - 0.02) < 1e-3 for o in out)          # the cube was not even touched


def test_the_cpp_random_policy_moves_the_arm_but_does_not_lift(exe, tmp_path):
    out = cpp_policy_rollout(exe, "random:0", SEEDS, tmp_path / "run")
    assert sum(o.success for o in out) == 0


def test_the_python_rollout_scores_like_the_evaluation_harness(trained):
    from teleop_il.evaluation import evaluate
    _, onnx_file, _ = trained
    py = python_policy_rollout(OnnxPolicy(onnx_file), SEEDS)
    assert [o.success for o in py] == list(evaluate(OnnxPolicy(onnx_file), seeds=SEEDS).per_seed)


def test_each_cpp_run_writes_a_new_folder(exe, tmp_path):
    cpp_policy_rollout(exe, "hold", SEEDS[:1], tmp_path / "run")
    with pytest.raises(FileExistsError):
        cpp_policy_rollout(exe, "hold", SEEDS[:1], tmp_path / "run")
