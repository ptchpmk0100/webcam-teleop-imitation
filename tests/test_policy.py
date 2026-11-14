"""Ticket 09: state policy (MLP, action chunk of 10), trained locally on scripted-expert episodes, exported to ONNX.

Frozen thresholds (board): training loss below 10 percent of its start on a 5-episode set; ONNX output within 1e-4 of
PyTorch on recorded observations.
"""
import numpy as np
import pytest

from teleop_il.dataset import Episode, record_episode
from teleop_il.env import Env
from teleop_il.expert import ExpertPolicy
from teleop_il.policy import (CHUNK, OBS_SIZE, OnnxPolicy, chunk_targets, episode_observations, export_onnx,
                              torch_actions, train)


@pytest.fixture(scope="module")
def episodes():
    env, expert = Env(), ExpertPolicy()
    return [record_episode(env, expert, seed=s, source="expert") for s in range(5)]


@pytest.fixture(scope="module")
def trained(episodes):
    return train(episodes, epochs=300, seed=0)


@pytest.fixture(scope="module")
def onnx_file(trained, tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "policy.onnx"
    export_onnx(trained.model, path)
    return path


def test_the_chunk_for_a_step_is_the_next_10_joint_targets_padded_with_the_last_one():
    n = 13
    targets = np.arange(n * 6, dtype=float).reshape(n, 6)
    e = Episode(seed=0, source="test", success=True, flagged=False, assist_fired=0, initial_state={},
                time=np.zeros(n), qpos=np.zeros((n, 13)), qvel=np.zeros((n, 12)), joint_targets=targets,
                cube_pose=np.zeros((n, 7)), tracking_lost=np.zeros(n, bool), assist_active=np.zeros(n, bool))
    chunks = chunk_targets(e)
    assert chunks.shape == (n, CHUNK, 6)
    assert np.array_equal(chunks[0], targets[:10])
    assert np.array_equal(chunks[8, :5], targets[8:13]) and np.array_equal(chunks[8, 5:], np.repeat(targets[-1:], 5, 0))


def test_observations_are_joint_positions_and_cube_pose_like_the_env_reports_them(episodes):
    env = Env()
    obs = env.reset(0)
    x = episode_observations(episodes[0])
    assert x.shape == (episodes[0].steps, OBS_SIZE)
    assert np.allclose(x[0], np.r_[obs["joint_pos"], obs["cube_pose"]])


def test_training_loss_falls_below_10_percent_of_its_start_on_5_episodes(trained):
    assert trained.losses[-1] < 0.1 * trained.losses[0]


def test_the_trained_policy_reproduces_the_recorded_joint_targets_closely(episodes, trained):
    x = np.vstack([episode_observations(e) for e in episodes])
    y = np.vstack([e.joint_targets for e in episodes])
    pred = torch_actions(trained.model, x)[:, 0]
    assert np.abs(pred - y).mean() < 0.05                         # radians, first action of the chunk


def test_onnx_output_matches_pytorch_within_1e_4_on_recorded_observations(episodes, trained, onnx_file):
    x = np.vstack([episode_observations(e) for e in episodes]).astype(np.float32)
    ref = torch_actions(trained.model, x)
    out = OnnxPolicy(onnx_file).actions(x)
    assert out.shape == (len(x), CHUNK, 6)
    assert np.abs(out - ref).max() <= 1e-4


def test_the_onnx_file_has_the_interface_the_cpp_side_expects(onnx_file):
    import onnx
    m = onnx.load(str(onnx_file))
    (inp,), (out,) = m.graph.input, m.graph.output
    dims = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
    assert inp.name == "observation" and dims(inp)[1:] == [OBS_SIZE]
    assert out.name == "actions" and dims(out)[1:] == [CHUNK, 6]


def test_the_onnx_policy_drives_the_env_from_its_observation_dict(onnx_file):
    env, policy = Env(), OnnxPolicy(onnx_file)
    obs = env.reset(0)
    a, b = policy(obs), policy(obs)
    assert a.shape == (6,) and np.array_equal(a, b)
    env.step(a)


def test_training_is_reproducible_with_the_same_seed(episodes):
    a = train(episodes[:2], epochs=5, seed=1).losses
    b = train(episodes[:2], epochs=5, seed=1).losses
    assert a == b
