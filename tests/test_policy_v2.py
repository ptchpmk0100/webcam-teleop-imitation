"""State policy v2 (PRD-v2): cube-relative observation, gripper-move chunk plus open/closed, chunk blending, executed
through the policy-path controller. Judged through the simulator and the evaluation harness."""
import mujoco
import numpy as np
import pytest

from teleop_il.dataset import record_episode
from teleop_il.env import Env
from teleop_il.evaluation import DEV_SEEDS, HELD_OUT_SEEDS
from teleop_il.expert import ExpertPolicy
from teleop_il.policy_v2 import (CHUNK, OBS_SIZE, OnnxPolicyV2, PolicyV2, episode_features, episode_labels,
                                 export_onnx, torch_actions, train)


@pytest.fixture(scope="module")
def expert_episodes():
    env, expert = Env(), ExpertPolicy()
    return [record_episode(env, expert, seed=s, source="expert") for s in range(5)]


@pytest.fixture(scope="module")
def trained(expert_episodes):
    return train(expert_episodes, epochs=150, seed=0)


def test_development_seeds_are_one_hundred_and_overlap_neither_demo_nor_held_out_seeds():
    assert len(DEV_SEEDS) == 100 == len(set(DEV_SEEDS))
    assert not set(DEV_SEEDS) & set(HELD_OUT_SEEDS)
    assert min(DEV_SEEDS) >= 10_000


def test_the_first_move_label_is_where_the_gripper_really_went_in_the_next_control_step(expert_episodes):
    e = expert_episodes[0]
    labels = episode_labels(e)
    env = Env()
    scratch = mujoco.MjData(env.model)

    def gripper_now():            # site positions after mj_step are one substep stale; recompute from the state
        scratch.qpos[:] = env.data.qpos
        mujoco.mj_kinematics(env.model, scratch)
        return scratch.site_xpos[env._ee_site].copy()

    env.restore(e.initial_state)
    for t in range(40):
        before = gripper_now()
        env.step(e.joint_targets[t])
        np.testing.assert_allclose(before + labels[t, 0, :3], gripper_now(), atol=1e-9)


def test_the_gripper_label_is_closed_exactly_where_the_demo_commanded_it_closed(expert_episodes):
    e = expert_episodes[0]
    labels = episode_labels(e)
    np.testing.assert_array_equal(labels[:, 0, 3] > 0.5, e.joint_targets[:, 5] < 0.4)


def test_labels_and_features_have_one_row_per_recorded_step(expert_episodes):
    e = expert_episodes[1]
    assert episode_features(e).shape == (e.steps, OBS_SIZE)
    assert episode_labels(e).shape == (e.steps, CHUNK, 4)


def test_training_loss_falls_below_ten_percent_of_its_start_on_five_episodes(trained):
    assert trained.losses[-1] < 0.1 * trained.losses[0]


def test_the_onnx_export_matches_pytorch_on_recorded_observations(trained, expert_episodes, tmp_path):
    x = np.vstack([episode_features(e) for e in expert_episodes])
    path = export_onnx(trained.model, tmp_path / "p.onnx")
    np.testing.assert_allclose(OnnxPolicyV2(path).actions(x), torch_actions(trained.model, x), atol=1e-4)


def test_a_policy_cloned_from_five_expert_demos_lifts_the_cube_on_its_own_training_positions(trained):
    from teleop_il.evaluation import evaluate
    result = evaluate(PolicyV2(lambda x: torch_actions(trained.model, x)), seeds=range(5))
    assert result.successes >= 4


def test_the_policy_is_deterministic(trained):
    from teleop_il.evaluation import evaluate
    policy = PolicyV2(lambda x: torch_actions(trained.model, x))
    assert evaluate(policy, seeds=DEV_SEEDS[:3]).per_seed == evaluate(policy, seeds=DEV_SEEDS[:3]).per_seed
