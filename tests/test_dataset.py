"""Seam C, dataset: write and read round trip, loud manifest validation, split by episode (PRD testing decisions)."""
import json

import numpy as np
import pytest

from teleop_il.dataset import DatasetError, DatasetWriter, Episode, read_dataset, record_episode, split
from teleop_il.env import Env
from teleop_il.expert import ExpertPolicy

ARRAYS = ("time", "qpos", "qvel", "joint_targets", "cube_pose", "tracking_lost", "assist_active")


@pytest.fixture(scope="module")
def expert_episodes():
    env, expert = Env(), ExpertPolicy()
    return [record_episode(env, expert, seed=s, source="expert") for s in (3, 4)]


def write(folder, episodes):
    w = DatasetWriter(folder)
    for e in episodes:
        w.add(e)
    w.close()
    return folder


def test_a_recorded_expert_episode_is_a_success_with_one_row_per_20_hz_step(expert_episodes):
    e = expert_episodes[0]
    assert e.success and e.seed == 3 and e.source == "expert"
    assert len(e.time) == len(e.qpos) == len(e.joint_targets) == len(e.cube_pose)
    assert np.allclose(np.diff(e.time), 0.05)
    assert e.cube_pose[-1][2] - e.cube_pose[0][2] > 0.05          # the cube really went up during the episode


def test_write_then_read_gives_back_exactly_the_same_episodes(tmp_path, expert_episodes):
    back = read_dataset(write(tmp_path / "ds", expert_episodes)).episodes
    assert len(back) == 2
    for a, b in zip(expert_episodes, back):
        assert (a.seed, a.source, a.success, a.flagged, a.assist_fired) == (b.seed, b.source, b.success, b.flagged,
                                                                             b.assist_fired)
        for name in ARRAYS:
            assert np.array_equal(getattr(a, name), getattr(b, name)), name
            assert getattr(a, name).dtype == getattr(b, name).dtype, name
        for key, value in a.initial_state.items():
            assert np.array_equal(value, b.initial_state[key]), key


def test_the_initial_state_restores_the_simulation_to_where_the_episode_started(tmp_path, expert_episodes):
    e = read_dataset(write(tmp_path / "ds", expert_episodes[:1])).episodes[0]
    env = Env()
    env.reset(99)
    env.restore(e.initial_state)
    assert np.array_equal(env.data.qpos, e.qpos[0])


def test_a_dataset_folder_is_never_written_twice(tmp_path, expert_episodes):
    write(tmp_path / "ds", expert_episodes[:1])
    with pytest.raises(FileExistsError):
        DatasetWriter(tmp_path / "ds")


def test_an_episode_with_mismatched_array_lengths_is_refused_when_written(tmp_path, expert_episodes):
    e = expert_episodes[0]
    bad = Episode(**{**e.__dict__, "joint_targets": e.joint_targets[:-1]})
    with pytest.raises(DatasetError, match="joint_targets"):
        DatasetWriter(tmp_path / "ds").add(bad)


# ---- malformed manifests fail loudly, each with a message that names the problem ----

def corrupt_manifest(folder, change):
    path = folder / "manifest.json"
    m = json.loads(path.read_text())
    change(m)
    path.write_text(json.dumps(m))


@pytest.mark.parametrize("change, message", [
    (lambda m: m.pop("format_version"), "format_version"),
    (lambda m: m.update(format_version=99), "format_version"),
    (lambda m: m["episodes"][0].pop("sha256"), "sha256"),
    (lambda m: m["episodes"][0].update(steps=1), "steps"),
    (lambda m: m["episodes"][0].update(file="missing.npz"), "missing.npz"),
    (lambda m: m["episodes"][0].update(seed=12345), "seed"),
    (lambda m: m["episodes"].append(dict(m["episodes"][0])), "twice"),
])
def test_a_malformed_manifest_fails_loudly(tmp_path, expert_episodes, change, message):
    folder = write(tmp_path / "ds", expert_episodes)
    corrupt_manifest(folder, change)
    with pytest.raises(DatasetError, match=message):
        read_dataset(folder)


def test_an_episode_file_changed_after_writing_fails_its_checksum(tmp_path, expert_episodes):
    folder = write(tmp_path / "ds", expert_episodes)
    name = json.loads((folder / "manifest.json").read_text())["episodes"][1]["file"]
    data = bytearray((folder / name).read_bytes())
    data[len(data) // 2] ^= 0xFF
    (folder / name).write_bytes(bytes(data))
    with pytest.raises(DatasetError, match="sha256"):
        read_dataset(folder)


def test_a_folder_without_a_manifest_fails_loudly(tmp_path):
    (tmp_path / "ds").mkdir()
    with pytest.raises(DatasetError, match="manifest"):
        read_dataset(tmp_path / "ds")


# ---- split by episode: 80/20, and two recordings of the same cube position always land on the same side ----

def fake_index(seeds):
    return [{"file": f"episode_{i:05d}.npz", "seed": s} for i, s in enumerate(seeds)]


def test_train_and_validation_share_no_episode_and_together_cover_all():
    index = fake_index(range(100))
    train, val = split(index, val_fraction=0.2, seed=0)
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(e["file"] for e in index)
    assert len(val) == 20


def test_retries_of_the_same_cube_position_never_straddle_the_split():
    index = fake_index([s // 2 for s in range(100)])            # every seed recorded twice
    train, val = split(index, val_fraction=0.2, seed=0)
    seed_of = {e["file"]: e["seed"] for e in index}
    assert not {seed_of[f] for f in train} & {seed_of[f] for f in val}


def test_the_split_is_the_same_every_time_for_the_same_seed_and_differs_for_another():
    index = fake_index(range(50))
    assert split(index, seed=0) == split(index, seed=0)
    assert split(index, seed=0) != split(index, seed=1)


# ---- merging record sessions into one dataset (ticket 10): a new folder, sources untouched ----

def test_merging_sessions_gives_one_valid_dataset_with_every_episode_unchanged(tmp_path, expert_episodes):
    import hashlib
    from teleop_il.dataset import merge_datasets
    a = write(tmp_path / "a", expert_episodes[:1])
    b = write(tmp_path / "b", expert_episodes[1:])
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for d in (a, b) for p in d.iterdir()}
    merged = read_dataset(merge_datasets([a, b], tmp_path / "all"))
    assert [e.seed for e in merged.episodes] == [3, 4]
    for x, y in zip(expert_episodes, merged.episodes):
        assert np.array_equal(x.qpos, y.qpos) and np.array_equal(x.joint_targets, y.joint_targets)
    assert [s["manifest_sha256"] for s in merged.manifest["sources"]] == [
        hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest() for d in (a, b)]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for d in (a, b) for p in d.iterdir()}


def test_merging_refuses_an_existing_folder_and_a_broken_source(tmp_path, expert_episodes):
    from teleop_il.dataset import merge_datasets
    a = write(tmp_path / "a", expert_episodes[:1])
    merge_datasets([a], tmp_path / "all")
    with pytest.raises(FileExistsError):
        merge_datasets([a], tmp_path / "all")
    (tmp_path / "broken").mkdir()
    with pytest.raises(DatasetError):
        merge_datasets([a, tmp_path / "broken"], tmp_path / "all2")
    assert not (tmp_path / "all2").exists()                     # nothing half-written
