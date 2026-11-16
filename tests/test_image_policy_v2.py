"""Ticket 16: image policy v2 (camera image + arm state -> the v2 gripper-move chunk and closed logit).

Thresholds from the board: training loss below 10 percent of its start on a 5-episode set; ONNX within 1e-4 of PyTorch.
The policy must never see the cube pose: the rollout test hands it an observation without one.
"""
import numpy as np
import pytest

from teleop_il import image_policy_v2 as ip
from teleop_il import policy_v2 as pv
from teleop_il.dataset import DatasetWriter, read_dataset, record_episode
from teleop_il.env import IMAGE_SIZE, Env
from teleop_il.expert import ExpertPolicy
from teleop_il.replay import read_images, render_dataset
from teleop_il.train_image_v2 import episode_arrays, prepare


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("imgv2")
    env, w = Env(), DatasetWriter(root / "ds")
    for seed in range(5):
        w.add(record_episode(env, ExpertPolicy(), seed=seed, source="expert"))
    w.close()
    render_dataset(root / "ds", root / "images")
    return root


@pytest.fixture(scope="module")
def arrays(data):
    ds = read_dataset(data / "ds")
    rows = [episode_arrays(e, read_images(data / "images", m["file"]), pv.CHUNK)
            for m, e in zip(ds.manifest["episodes"], ds.episodes)]
    return (np.concatenate([r[0] for r in rows]), np.vstack([r[1] for r in rows]),
            np.concatenate([r[2] for r in rows]))


@pytest.fixture(scope="module")
def trained(arrays):
    model, losses = ip.train(*arrays, epochs=300, seed=0, shift=0, batch_size=32, channels=8, device="cpu")   # overfit check: no augmentation, small CNN
    return model, losses


def test_chunk_and_image_size_match_the_state_policy_and_the_environment():
    assert ip.CHUNK == pv.CHUNK and ip.ACTION_SIZE == pv.ACTION_SIZE
    assert ip.IMAGE_HW == IMAGE_SIZE


def test_arrays_have_one_row_per_step_and_the_proprio_is_joints_then_gripper_position(arrays):
    images, proprio, labels = arrays
    assert images.shape[1:] == (*IMAGE_SIZE, 3) and images.dtype == np.uint8
    assert proprio.shape == (len(images), ip.PROPRIO_SIZE)
    assert labels.shape == (len(images), pv.CHUNK, 4)
    c = pv._controller()
    np.testing.assert_allclose(proprio[7, 6:], c.gripper_position(proprio[7, :6]), atol=1e-6)


def test_random_shift_keeps_shape_is_identity_without_padding_and_moves_pixels():
    import torch
    x = torch.zeros(4, 96, 128, 3)
    x[:, 40, 60] = 255.0
    assert torch.equal(ip.random_shift(x, 0), x)
    y = ip.random_shift(x, 4, generator=torch.Generator().manual_seed(1))
    assert y.shape == x.shape
    where = [tuple(torch.nonzero(y[i, ..., 0])[0].tolist()) for i in range(4)]
    assert all(abs(r - 40) <= 4 and abs(c - 60) <= 4 for r, c in where) and len(set(where)) > 1


def test_training_loss_falls_below_ten_percent_on_a_five_episode_set(trained):
    _, losses = trained
    assert losses[-1] < 0.10 * losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}"


def test_onnx_matches_pytorch_within_1e_4(trained, arrays, tmp_path):
    model, _ = trained
    images, proprio, _ = arrays
    path = ip.export_onnx(model, tmp_path / "p.onnx")
    sel = slice(0, 64)
    expected = ip.torch_actions(model, images[sel], proprio[sel])
    got = ip.OnnxImagePolicyV2(path).actions(images[sel], proprio[sel])
    assert got.shape == (64, pv.CHUNK, 4)
    assert np.abs(got - expected).max() < 1e-4
    with pytest.raises(FileExistsError):
        ip.export_onnx(model, path)


def test_the_rollout_policy_runs_from_the_image_and_joints_only(trained, tmp_path):
    model, _ = trained
    path = ip.export_onnx(model, tmp_path / "p.onnx")
    env = Env()
    obs = env.reset(3)
    policy = ip.image_rollout_policy(path, env.render)
    action = policy({"joint_pos": obs["joint_pos"]})            # no cube pose in the observation
    assert action.shape == (6,) and np.all(np.isfinite(action))


def test_state_dict_round_trip_restores_the_normalisation(trained, tmp_path):
    import torch
    model, _ = trained
    torch.save(model.state_dict(), tmp_path / "policy.pt")
    again = ip.load_model(tmp_path / "policy.pt", channels=8)
    for name in ("p_mean", "p_std", "y_mean", "y_std"):
        assert torch.equal(getattr(model, name), getattr(again, name))


def test_prepare_writes_train_and_validation_arrays_with_the_same_split_as_the_state_policy(data, tmp_path):
    out = prepare(data / "ds", data / "images", tmp_path / "arrays")
    with np.load(out / "train.npz") as tr, np.load(out / "val.npz") as va:
        assert len(tr["images"]) == len(tr["proprio"]) == len(tr["labels"]) > 0
        assert len(va["images"]) > 0
    with pytest.raises(FileExistsError):
        prepare(data / "ds", data / "images", tmp_path / "arrays")
