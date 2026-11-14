"""Ticket 15: image policy (small CNN with spatial softmax + MLP on image and joints, 10-step action chunks), trained
locally on scripted-expert episodes and their rendered images, exported to ONNX.

Frozen thresholds (board): training loss below 10 percent of its start on a 5-episode set; ONNX within 1e-4 of PyTorch.
"""
import numpy as np
import pytest

from teleop_il.dataset import DatasetWriter, read_dataset, record_episode
from teleop_il.env import CUBE_HALF, IMAGE_SIZE, Env
from teleop_il.expert import ExpertPolicy
from teleop_il.image_policy import (ImageOnnxPolicy, episode_joints, export_image_onnx, image_torch_actions,
                                    train_image)
from teleop_il.replay import read_images, render_dataset


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("img")
    env, w = Env(), DatasetWriter(root / "ds")
    for seed in range(5):
        w.add(record_episode(env, ExpertPolicy(), seed=seed, source="expert"))
    w.close()
    render_dataset(root / "ds", root / "images")
    ds = read_dataset(root / "ds")
    images = [read_images(root / "images", e["file"]) for e in ds.manifest["episodes"]]
    return ds.episodes, images


@pytest.fixture(scope="module")
def trained(data):
    episodes, images = data
    return train_image(episodes, images, epochs=10, seed=0)


@pytest.fixture(scope="module")
def onnx_file(trained, tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "image_policy.onnx"
    export_image_onnx(trained.model, path)
    return path


def test_training_loss_falls_below_10_percent_of_its_start_on_5_episodes(trained):
    assert trained.losses[-1] < 0.1 * trained.losses[0]


def test_onnx_output_matches_pytorch_within_1e_4_on_recorded_images_and_joints(data, trained, onnx_file):
    episodes, images = data
    img = np.vstack(images[:2]).astype(np.float32)
    joints = np.vstack([episode_joints(e) for e in episodes[:2]])
    ref = image_torch_actions(trained.model, img, joints)
    out = ImageOnnxPolicy(onnx_file, render=None).actions(img, joints)
    assert out.shape == (len(img), 10, 6)
    assert np.abs(out - ref).max() <= 1e-4


def test_the_onnx_file_takes_a_raw_image_and_the_joints_and_returns_the_chunk(onnx_file):
    import onnx
    m = onnx.load(str(onnx_file))
    dims = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
    inputs = {v.name: dims(v)[1:] for v in m.graph.input}
    assert inputs == {"image": [*IMAGE_SIZE, 3], "joints": [6]}
    (out,) = m.graph.output
    assert out.name == "actions" and dims(out)[1:] == [10, 6]


def test_the_image_policy_drives_the_env_through_its_camera(onnx_file):
    env = Env()
    policy = ImageOnnxPolicy(onnx_file, render=env.render)
    obs = env.reset(0)
    a, b = policy(obs), policy(obs)
    assert a.shape == (6,) and np.allclose(a, b, atol=1e-4)       # renders differ by GPU noise only (ticket 08)


def test_the_image_policy_sees_the_cube_only_through_the_image(onnx_file):
    env = Env()
    policy = ImageOnnxPolicy(onnx_file, render=env.render)
    obs = env.reset(0)
    lied = {**obs, "cube_pose": obs["cube_pose"] + np.array([0.1, 0.1, 0, 0, 0, 0, 0])}
    before = policy(obs)
    assert np.allclose(before, policy(lied), atol=1e-4)          # the cube pose in the observation is ignored
    env.place_cube([0.28, -0.12, CUBE_HALF])                      # moving the real cube changes the picture
    assert np.abs(policy(obs) - before).max() > 1e-3


def test_training_is_reproducible_with_the_same_seed(data):
    episodes, images = data
    a = train_image(episodes[:1], images[:1], epochs=1, seed=3).losses
    b = train_image(episodes[:1], images[:1], epochs=1, seed=3).losses
    assert a == b


def test_the_trained_image_policy_reproduces_the_recorded_joint_targets(data, trained):
    episodes, images = data
    pred = image_torch_actions(trained.model, np.vstack(images).astype(np.float32),
                               np.vstack([episode_joints(e) for e in episodes]))[:, 0]
    err = np.abs(pred - np.vstack([e.joint_targets for e in episodes])).mean()
    assert err < 0.05                                             # radians, first action of the chunk
