"""Ticket 24: the C++ rollout runs policy v2 (controller, ONNX contract v2, chunk blending) like Python."""
import numpy as np
import pytest

from teleop_il.controller import GripperController
from teleop_il.cpp_parity import build
from teleop_il.cpp_policy_parity import python_policy_rollout
from teleop_il.cpp_policy_v2_parity import cpp_controller, cpp_infer_v2, cpp_policy_v2_rollout, recorded_cases
from teleop_il.dataset import record_episode
from teleop_il.env import Env
from teleop_il.expert import ExpertPolicy
from teleop_il.policy_v2 import OnnxPolicyV2, PolicyV2, episode_features, export_onnx, torch_actions, train

TRAINING_SEEDS = tuple(range(5))


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    return build(tmp_path_factory.mktemp("build")).parent / "policy_v2_rollout"


@pytest.fixture(scope="module")
def episodes():
    env = Env()
    return [record_episode(env, ExpertPolicy(), seed=s, source="expert") for s in TRAINING_SEEDS]


@pytest.fixture(scope="module")
def trained(episodes, tmp_path_factory):
    model = train(episodes, epochs=150, seed=0).model
    path = export_onnx(model, tmp_path_factory.mktemp("onnx") / "policy.onnx")
    return model, path


def test_the_cpp_controller_matches_python_within_1e_9_on_recorded_states(exe, episodes, tmp_path):
    joints, targets, closed = recorded_cases(episodes[:2])
    cmd, ee = cpp_controller(exe, joints, targets, closed, tmp_path / "run")
    ctl = GripperController()
    assert np.abs(cmd - np.array([ctl.command(q, t, c) for q, t, c in zip(joints, targets, closed)])).max() <= 1e-9
    assert np.abs(ee - np.array([ctl.gripper_position(q) for q in joints])).max() <= 1e-9
    assert closed.any() and not closed.all()                  # both gripper commands are compared


def test_cpp_onnx_runtime_matches_pytorch_within_1e_4_on_the_v2_contract(exe, trained, episodes, tmp_path):
    model, path = trained
    x = np.vstack([episode_features(e) for e in episodes[:2]])
    out = cpp_infer_v2(exe, path, x, tmp_path / "run")
    assert out.shape == (len(x), 20, 4)
    assert np.abs(out - torch_actions(model, x)).max() <= 1e-4


def test_cpp_and_python_policy_v2_rollouts_agree_on_success_and_cube_height(exe, trained, tmp_path):
    _, path = trained
    py = python_policy_rollout(PolicyV2(OnnxPolicyV2(path).actions), TRAINING_SEEDS)
    cpp = cpp_policy_v2_rollout(exe, path, TRAINING_SEEDS, tmp_path / "run")
    assert [o.success for o in py] == [o.success for o in cpp]
    assert max(abs(a.cube_z - b.cube_z) for a, b in zip(py, cpp)) < 0.005
    assert sum(o.success for o in py) >= 4                    # the comparison includes lifts, not only failures


def test_the_cpp_null_policies_do_not_lift(exe, tmp_path):
    seeds = tuple(range(10_000, 10_008))
    assert not any(o.success for o in cpp_policy_v2_rollout(exe, "hold", seeds, tmp_path / "hold"))
    assert not any(o.success for o in cpp_policy_v2_rollout(exe, "random:0", seeds, tmp_path / "random"))
