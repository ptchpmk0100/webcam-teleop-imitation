"""Ticket 08: replay a stored episode and render camera images into a new folder; inputs stay unchanged."""
import hashlib

import numpy as np
import pytest

from teleop_il.dataset import DatasetWriter, Episode, read_dataset, record_episode
from teleop_il.env import Env
from teleop_il.expert import ExpertPolicy
from teleop_il.ik import GRIPPER_OPEN
from teleop_il.replay import read_images, render_dataset, replay_actions

H, W = 96, 128


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    env, expert = Env(), ExpertPolicy()
    folder = tmp_path_factory.mktemp("data") / "expert"
    w = DatasetWriter(folder)
    for seed in (5, 6):
        w.add(record_episode(env, expert, seed=seed, source="expert"))
    w.close()
    return folder


@pytest.fixture(scope="module")
def rendered(dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("render") / "images"
    render_dataset(dataset, out, height=H, width=W)
    return out


def test_replaying_an_expert_episode_reproduces_its_success_and_its_recorded_cube_path(dataset):
    for e in read_dataset(dataset).episodes:
        r = replay_actions(e)
        assert r.success == e.success
        drift = np.linalg.norm(r.cube_pose[:, :3] - e.cube_pose[:, :3], axis=1)
        assert drift.max() < 0.005                                  # every step, against what was recorded


def test_replay_really_runs_the_actions_a_gripper_that_never_closes_does_not_lift(dataset):
    e = read_dataset(dataset).episodes[0]
    targets = e.joint_targets.copy()
    targets[:, 5] = GRIPPER_OPEN
    r = replay_actions(Episode(**{**e.__dict__, "joint_targets": targets}))
    assert not r.success and r.final_cube_pose[2] < 0.03


def test_one_image_per_recorded_step_in_the_requested_size(dataset, rendered):
    for entry, e in zip(read_dataset(dataset).manifest["episodes"], read_dataset(dataset).episodes):
        images = read_images(rendered, entry["file"])
        assert images.shape == (e.steps, H, W, 3) and images.dtype == np.uint8


def test_rendered_images_are_not_blank_and_show_the_green_cube_and_the_motion(dataset, rendered):
    entry = read_dataset(dataset).manifest["episodes"][0]
    images = read_images(rendered, entry["file"]).astype(int)
    first, last = images[0], images[-1]
    assert first.std() > 10                                         # not a flat colour
    green = (first[..., 1] > 120) & (first[..., 0] < 80) & (first[..., 2] < 80)
    assert green.sum() > 20                                         # the cube is in view
    assert np.abs(first - last).mean() > 2                          # the arm moved and the cube went up


def test_images_follow_the_recorded_state_not_a_fresh_simulation(dataset, tmp_path):
    """Human demos cannot be re-simulated exactly (the live loop steps physics in 5 ms slices on camera timing), so
    images must come from each step's stored state: frozen states give identical frames."""
    e = read_dataset(dataset).episodes[0]
    frozen = Episode(**{**e.__dict__, "qpos": np.repeat(e.qpos[:1], e.steps, axis=0)})
    folder = tmp_path / "frozen"
    w = DatasetWriter(folder); w.add(frozen); w.close()
    render_dataset(folder, tmp_path / "img", height=H, width=W)
    images = read_images(tmp_path / "img", "episode_00000.npz")
    assert same_image(images[0], images[-1])


def same_image(a, b):
    """Equal up to GPU noise: rendering one state twice differs by at most 1 level on about 20 pixels (measured), a
    one-substep-stale arm by up to 166 levels on 400 to 500 pixels."""
    return np.abs(a.astype(int) - b.astype(int)).max() <= 2


def test_rendering_writes_a_new_folder_and_leaves_the_dataset_unchanged(dataset, tmp_path):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir()}
    render_dataset(dataset, tmp_path / "img", height=H, width=W)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir()}
    assert before == after
    with pytest.raises(FileExistsError):
        render_dataset(dataset, tmp_path / "img", height=H, width=W)


def test_the_images_record_which_dataset_they_were_rendered_from(dataset, rendered):
    import json
    m = json.loads((rendered / "images_manifest.json").read_text())
    source = hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
    assert m["source_manifest_sha256"] == source and m["camera"] == "policy" and m["size"] == [H, W]


def test_the_env_camera_at_evaluation_shows_exactly_what_the_training_images_show(dataset, rendered):
    """Training images come from render_dataset, evaluation images from Env.render: a mismatch would score the image
    policy on a view it never trained on."""
    e = read_dataset(dataset).episodes[0]
    entry = read_dataset(dataset).manifest["episodes"][0]
    env = Env()
    env.restore(e.initial_state)
    for action in e.joint_targets[:60]:                            # step 60: the arm is moving (at 40 it is still)
        env.step(action)
    images = read_images(rendered, entry["file"])
    assert same_image(env.render(H, W), images[60])
