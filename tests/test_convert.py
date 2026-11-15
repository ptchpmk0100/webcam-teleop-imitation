"""Seam for dataset conversion (PRD-v2): a dataset folder in, a new dataset folder out, judged by replaying it in the
simulator through the policy-path controller (lifts counted from MuJoCo state)."""
import dataclasses
import hashlib
import json

import numpy as np
import pytest

from teleop_il.convert import build, clean, gate, pushed_before_grasp
from teleop_il.dataset import DatasetWriter, read_dataset, record_episode
from teleop_il.env import Env
from teleop_il.expert import ExpertPolicy
from teleop_il.ik import GRIPPER_CLOSED, GRIPPER_OPEN

SEEDS = (3, 4, 5)


@pytest.fixture(scope="module")
def expert_episodes():
    env, expert = Env(), ExpertPolicy()
    return [record_episode(env, expert, seed=s, source="expert") for s in SEEDS]


def write(folder, episodes):
    w = DatasetWriter(folder)
    for e in episodes:
        w.add(e)
    w.close()
    return folder


def never_closes(e):
    targets = e.joint_targets.copy()
    targets[:, 5] = GRIPPER_OPEN
    return dataclasses.replace(e, joint_targets=targets)


def test_scripted_expert_demos_replay_to_a_lift_through_the_controller(expert_episodes, tmp_path, capsys):
    assert gate(write(tmp_path / "d", expert_episodes), required=1.0) == 0
    assert "3/3" in capsys.readouterr().out


def test_the_gate_fails_loudly_when_a_demo_does_not_replay(expert_episodes, tmp_path, capsys):
    data = write(tmp_path / "d", [*expert_episodes[:2], never_closes(expert_episodes[2])])
    assert gate(data, required=0.95) == 1
    out = capsys.readouterr().out
    assert "2/3" in out and "FAIL" in out
    assert gate(data, required=0.6) == 0


def test_the_gate_can_keep_only_the_demos_that_replay_in_a_new_folder(expert_episodes, tmp_path):
    data = write(tmp_path / "d", [expert_episodes[0], never_closes(expert_episodes[1])])
    gate(data, required=0.0, keep=tmp_path / "kept")
    kept = read_dataset(tmp_path / "kept")
    assert [e.seed for e in kept.episodes] == [SEEDS[0]]
    assert kept.manifest["replay_gate"]["replayed"] == 1 and kept.manifest["replay_gate"]["episodes"] == 2


def test_cleaning_drops_an_episode_the_assist_grasped_at_its_first_row(expert_episodes):
    e = expert_episodes[0]
    active = e.assist_active.copy()
    active[:30] = True
    assert clean(dataclasses.replace(e, assist_active=active)) is None


def test_cleaning_drops_a_stale_closed_gripper_first_row_and_starts_from_the_next_recorded_state(expert_episodes):
    e = expert_episodes[0]
    targets = e.joint_targets.copy()
    targets[0, 5] = GRIPPER_CLOSED
    out = clean(dataclasses.replace(e, joint_targets=targets))
    assert out.steps == e.steps - 1
    assert out.time[0] == e.time[1] and out.initial_state["time"] == e.time[1]
    np.testing.assert_array_equal(out.initial_state["qpos"], e.qpos[1])
    np.testing.assert_array_equal(out.joint_targets, e.joint_targets[1:])


def test_cleaning_keeps_a_normal_episode_unchanged(expert_episodes):
    e = expert_episodes[0]
    out = clean(e)
    assert out.steps == e.steps
    np.testing.assert_array_equal(out.qpos, e.qpos)


def test_a_demo_that_pushed_the_cube_before_the_first_close_is_recognised(expert_episodes):
    e = expert_episodes[0]
    assert not pushed_before_grasp(e)
    pose = e.cube_pose.copy()
    first_close = int(np.argmax(e.joint_targets[:, 5] < 0.4))
    pose[first_close // 2:first_close, 0] += 0.01
    assert pushed_before_grasp(dataclasses.replace(e, cube_pose=pose))


def test_build_cleans_sessions_into_a_new_folder_and_leaves_the_sources_unchanged(expert_episodes, tmp_path):
    a = write(tmp_path / "a", expert_episodes[:2])
    stale = expert_episodes[2].joint_targets.copy()
    stale[0, 5] = GRIPPER_CLOSED
    active = expert_episodes[1].assist_active.copy()
    active[0] = True
    b = write(tmp_path / "b", [dataclasses.replace(expert_episodes[2], joint_targets=stale),
                               dataclasses.replace(expert_episodes[1], assist_active=active)])
    before = [hashlib.sha256((f / "manifest.json").read_bytes()).hexdigest() for f in (a, b)]
    build([a, b], tmp_path / "out")
    after = [hashlib.sha256((f / "manifest.json").read_bytes()).hexdigest() for f in (a, b)]
    assert before == after
    out = read_dataset(tmp_path / "out")
    assert [e.seed for e in out.episodes] == [SEEDS[0], SEEDS[1], SEEDS[2]]
    assert out.manifest["cleaning"]["counts"] == {"in": 4, "dropped_auto_grasp": 1, "dropped_stale_row0": 1, "kept": 3}
    assert [s["folder"] for s in out.manifest["cleaning"]["sources"]] == [str(a), str(b)]
    with pytest.raises(FileExistsError):
        build([a], tmp_path / "out")
    json.dumps(out.manifest)
