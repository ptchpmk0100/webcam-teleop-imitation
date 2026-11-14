"""State policy (ticket 09): an MLP from the state to a chunk of the next 10 joint targets, trained by behaviour
cloning, exported to ONNX for the C++ rollout.

Contract with the C++ side (ticket 14): the ONNX graph takes "observation", float32 (batch, 13) = 6 joint positions
(5 arm joints and the gripper, as Env reports them) followed by the cube pose (position and quaternion), and returns
"actions", float32 (batch, 10, 6) = absolute joint targets for this step and the next 9. Normalisation is inside the
graph, so the C++ side feeds raw state. At run time the policy re-plans every step and executes the chunk's first
action; the rest of the chunk is a training target that smooths what the network learns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHUNK = 10
ACTION_SIZE = 6
OBS_SIZE = 13
STD_FLOOR = 1e-2                 # features that barely vary (the cube quaternion on the table) are not blown up
_ARM_QADR = None


def _arm_qadr():
    global _ARM_QADR
    if _ARM_QADR is None:
        from teleop_il.env import Env
        _ARM_QADR = Env()._arm_qadr.copy()
    return _ARM_QADR


def observation(obs: dict) -> np.ndarray:
    """The policy's input from an Env observation dict."""
    return np.r_[obs["joint_pos"], obs["cube_pose"]].astype(np.float32)


def episode_observations(e) -> np.ndarray:
    """(T, 13) policy inputs for every recorded step of an episode (state before that step's action)."""
    return np.hstack([e.qpos[:, _arm_qadr()], e.cube_pose]).astype(np.float32)


def chunk_targets(e, chunk: int = CHUNK) -> np.ndarray:
    """(T, chunk, 6): for each step, its joint targets and the next chunk-1, padded with the episode's last one."""
    a = np.asarray(e.joint_targets)
    padded = np.vstack([a, np.repeat(a[-1:], chunk - 1, axis=0)])
    return np.stack([padded[t:t + chunk] for t in range(len(a))])


def _model_class():
    import torch
    from torch import nn

    class StateMLP(nn.Module):
        def __init__(self, x_mean, x_std, y_mean, y_std, hidden: int = 256) -> None:
            super().__init__()
            for name, value in (("x_mean", x_mean), ("x_std", x_std), ("y_mean", y_mean), ("y_std", y_std)):
                self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
            self.net = nn.Sequential(nn.Linear(OBS_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, CHUNK * ACTION_SIZE))

        def normalized(self, obs):
            return self.net((obs - self.x_mean) / self.x_std).view(-1, CHUNK, ACTION_SIZE)

        def forward(self, obs):                     # raw state in, raw joint targets out
            return self.normalized(obs) * self.y_std + self.y_mean

    return StateMLP


@dataclass
class TrainResult:
    model: object
    losses: list                 # losses[0] before any update, then the mean training loss after each epoch


def train(episodes, epochs: int = 200, seed: int = 0, batch_size: int = 256, lr: float = 1e-3,
          hidden: int = 256) -> TrainResult:
    """Behaviour cloning on the episodes' recorded steps: mean squared error on normalised action chunks."""
    import torch
    torch.manual_seed(seed)
    x = np.vstack([episode_observations(e) for e in episodes])
    y = np.vstack([chunk_targets(e) for e in episodes]).astype(np.float32)
    x_mean, x_std = x.mean(0), np.maximum(x.std(0), STD_FLOOR)
    y_mean, y_std = y.reshape(-1, ACTION_SIZE).mean(0), np.maximum(y.reshape(-1, ACTION_SIZE).std(0), STD_FLOOR)
    model = _model_class()(x_mean, x_std, y_mean, y_std, hidden)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    yn = (yt - model.y_mean) / model.y_std
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        losses = [float(((model.normalized(xt) - yn) ** 2).mean())]
    for _ in range(epochs):
        order = torch.randperm(len(xt), generator=gen)
        total, n = 0.0, 0
        for i in range(0, len(xt), batch_size):
            idx = order[i:i + batch_size]
            loss = ((model.normalized(xt[idx]) - yn[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(idx), n + len(idx)
        losses.append(total / n)
    model.eval()
    return TrainResult(model=model, losses=losses)


def torch_actions(model, x) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.as_tensor(np.asarray(x, np.float32))).numpy()


def export_onnx(model, path) -> Path:
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; exports go to new files")
    batch = torch.export.Dim("batch", min=1, max=4096)
    program = torch.onnx.export(model, (torch.zeros(2, OBS_SIZE),), input_names=["observation"],
                                output_names=["actions"], dynamic_shapes=({0: batch},), dynamo=True)
    program.save(str(path))
    return path


class OnnxPolicy:
    """Runs the exported policy with ONNX Runtime; callable on an Env observation dict like any harness policy."""

    def __init__(self, path) -> None:
        import onnxruntime as ort
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def actions(self, x) -> np.ndarray:
        return self._session.run(["actions"], {"observation": np.asarray(x, np.float32)})[0]

    def __call__(self, obs) -> np.ndarray:
        return self.actions(observation(obs)[None])[0, 0].astype(np.float64)
