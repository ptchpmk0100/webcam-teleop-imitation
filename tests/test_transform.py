"""Seam for transformed copies (PRD-v2, ticket 22): a dataset folder in, a NEW folder of copies out. Each copy moves a
demo's gripper path to a new cube position, is replayed through the policy-path controller, and is kept only if the
cube lifts (MuJoCo state). Copies use seeds below 10000 that no demo, development or held-out run uses."""
import json
import math

import numpy as np
import pytest

from teleop_il.convert import gate
from teleop_il.dataset import DatasetWriter, read_dataset, record_episode
from teleop_il.env import Env
from teleop_il.evaluation import DEV_SEEDS, HELD_OUT_SEEDS
from teleop_il.expert import ExpertPolicy
from teleop_il.transform import copy_seeds, generate, train_copies, transform_path


def _rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def test_copy_seeds_stay_below_10000_and_avoid_demo_dev_and_held_out_seeds():
    demo_seeds = {1000, 1001, 1005}
    seeds = copy_seeds(500, exclude=demo_seeds)
    assert len(seeds) == len(set(seeds)) == 500
    assert all(s < 10_000 for s in seeds)
    assert not set(seeds) & (demo_seeds | set(DEV_SEEDS) | set(HELD_OUT_SEEDS))
    with pytest.raises(ValueError):
        copy_seeds(10_000, exclude=set())


def test_after_the_ramp_the_path_keeps_its_offset_from_the_cube_turned_with_the_bearing():
    rng = np.random.default_rng(0)
    path = np.c_[rng.uniform(0.1, 0.3, (30, 2)), rng.uniform(0.02, 0.1, 30)]
    c0, c1 = np.array([0.22, -0.05, 0.02]), np.array([0.27, 0.11, 0.02])
    out = transform_path(path, c0, c1, ramp_end=10)
    dtheta = math.atan2(c1[1], c1[0]) - math.atan2(c0[1], c0[0])
    np.testing.assert_allclose(out[0], path[0], atol=1e-12)                 # starts where the arm starts
    np.testing.assert_allclose(out[:, 2], path[:, 2], atol=1e-12)           # heights unchanged
    for t in range(10, 30):
        np.testing.assert_allclose(out[t, :2] - c1[:2], _rot(dtheta) @ (path[t, :2] - c0[:2]), atol=1e-12)
    np.testing.assert_allclose(transform_path(path, c0, c0, ramp_end=10), path, atol=1e-12)


@pytest.fixture(scope="module")
def expert_folder(tmp_path_factory):
    env, expert = Env(), ExpertPolicy()
    w = DatasetWriter(tmp_path_factory.mktemp("src") / "d")
    for s in (3, 4):
        w.add(record_episode(env, expert, seed=s, source="expert"))
    w.close()
    return w._folder


def test_every_kept_copy_lifts_names_its_source_and_its_labels_replay(expert_folder, tmp_path):
    out = generate(expert_folder, tmp_path / "copies", copies_per_demo=2, workers=1)
    ds, src = read_dataset(out), read_dataset(expert_folder)
    info = ds.manifest["transform"]
    assert info["source"]["folder"] == str(expert_folder)
    assert info["counts"]["tried"] == 4 and info["counts"]["kept"] == len(ds.episodes) >= 1
    by_file = {e["file"]: e for e in src.manifest["episodes"]}
    for entry, e in zip(ds.manifest["episodes"], ds.episodes):
        copy = info["copies"][entry["file"]]
        assert copy["source_file"] in by_file and copy["source_seed"] == by_file[copy["source_file"]]["seed"]
        assert e.success and e.source == "transformed"
        assert e.seed < 10_000 and e.seed not in {3, 4}
        assert e.seed not in set(DEV_SEEDS) | set(HELD_OUT_SEEDS)
    assert gate(out, required=1.0) == 0                          # the v2 labels of every copy reproduce its lift


def test_only_copies_of_training_demos_are_added_to_training():
    manifest = {"transform": {"copies": {"episode_00000.npz": {"source_file": "a.npz", "source_seed": 1},
                                         "episode_00001.npz": {"source_file": "b.npz", "source_seed": 2},
                                         "episode_00002.npz": {"source_file": "a.npz", "source_seed": 1}}}}
    assert train_copies(manifest, train_files=["a.npz"]) == ["episode_00000.npz", "episode_00002.npz"]
    assert json.dumps(train_copies(manifest, train_files=[])) == "[]"
