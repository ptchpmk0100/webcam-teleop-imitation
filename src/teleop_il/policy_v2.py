"""State policy v2 (PRD-v2, ticket 20): what the gripper should do next, relative to the cube.

Observation (6 values, built from the Env observation only: joint positions and cube pose; the gripper position comes
from forward kinematics of the joints): gripper frame minus the cube's grasp point (3; the grasp point moves with the
cube, so it stays meaningful while lifting), gripper height above the table, cube bottom height, gripper joint
(opening).

Action chunk (CHUNK steps of 4 values): where the gripper frame should be after each of the next CHUNK control steps,
relative to where it is now (metres), and a "closed" logit. Labels come from the demo itself: the gripper positions the
arm achieved, and the recorded open/closed command (see `convert`).

Execution: every step the policy predicts a chunk; the targets that all recent chunks predicted for the next step are
blended (ACT-style temporal ensembling, weights exp(-m * i) with i = 0 for the oldest chunk), the closed probability is
blended the same way and passed through hysteresis, and the policy-path controller turns the target into joint targets.

ONNX contract v2 (ticket 24): "observation" float32 (batch, 6) -> "actions" float32 (batch, CHUNK, 4), normalisation
inside the graph; blending, hysteresis and the controller live in the rollout loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from teleop_il.env import ARM_JOINTS, CUBE_HALF
from teleop_il.expert import GRASP_Z, grasp_point

CHUNK = 20                         # ticket 20 screen, 80 expert demos, dev seeds: chunk 10 gave 82, 20 gave 100 (3 of 3
                                   # training seeds), 30 gave 100
OBS_SIZE = 6
ACTION_SIZE = 4
ENSEMBLE_M = 0.01                  # ACT's default
CLOSE_ABOVE, OPEN_BELOW = 0.6, 0.4  # hysteresis on the blended closed probability
_X_FLOOR, _Y_FLOOR = 1e-3, 1e-4    # std floors: features in metres / radians, moves in metres
_CONTROLLER = None


def _controller():
    global _CONTROLLER
    if _CONTROLLER is None:
        from teleop_il.controller import GripperController
        _CONTROLLER = GripperController()
    return _CONTROLLER


def features(joint_pos, cube_pose, ee) -> np.ndarray:
    cube = np.asarray(cube_pose, float)
    goal = grasp_point(cube[:3], z=float(cube[2]) + GRASP_Z - CUBE_HALF)
    ee = np.asarray(ee, float)
    return np.r_[ee - goal, ee[2], cube[2] - CUBE_HALF, float(joint_pos[5])].astype(np.float32)


def _arm_rows(e) -> np.ndarray:
    from teleop_il.env import Env
    if not hasattr(_arm_rows, "qadr"):
        _arm_rows.qadr = Env()._arm_qadr.copy()
    return np.asarray(e.qpos)[:, _arm_rows.qadr]


def _path(e) -> np.ndarray:
    c = _controller()
    return np.array([c.gripper_position(q) for q in _arm_rows(e)])


def episode_features(e) -> np.ndarray:
    joints, path = _arm_rows(e), _path(e)
    return np.array([features(joints[t], e.cube_pose[t], path[t]) for t in range(e.steps)], np.float32)


def episode_labels(e, chunk: int = CHUNK) -> np.ndarray:
    """(T, chunk, 4): moves to the achieved gripper positions after 1..chunk steps, and closed (0/1) at the step."""
    from teleop_il.convert import gripper_closed
    path, closed = _path(e), gripper_closed(e).astype(np.float32)
    T = e.steps
    out = np.empty((T, chunk, ACTION_SIZE), np.float32)
    for t in range(T):
        future = np.minimum(np.arange(t + 1, t + 1 + chunk), T - 1)
        now = np.minimum(np.arange(t, t + chunk), T - 1)
        out[t, :, :3] = path[future] - path[t]
        out[t, :, 3] = closed[now]
    return out


def _model_class():
    import torch
    from torch import nn

    class PolicyMLP(nn.Module):
        def __init__(self, x_mean, x_std, y_mean, y_std, chunk: int, hidden: int) -> None:
            super().__init__()
            for name, value in (("x_mean", x_mean), ("x_std", x_std), ("y_mean", y_mean), ("y_std", y_std)):
                self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
            self.chunk = chunk
            self.net = nn.Sequential(nn.Linear(OBS_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, chunk * ACTION_SIZE))

        def normalized(self, obs):
            return self.net((obs - self.x_mean) / self.x_std).view(-1, self.chunk, ACTION_SIZE)

        def forward(self, obs):            # raw observation in; moves in metres and the closed logit out
            out = self.normalized(obs)
            return torch.cat([out[..., :3] * self.y_std + self.y_mean, out[..., 3:]], dim=-1)

    return PolicyMLP


@dataclass
class TrainResult:
    model: object
    losses: list                 # losses[0] before any update, then the mean training loss after each epoch


def train(episodes, epochs: int = 200, seed: int = 0, chunk: int = CHUNK, hidden: int = 256, lr: float = 1e-3,
          batch_size: int = 256, x=None, y=None) -> TrainResult:
    """Behaviour cloning: L1 on normalised moves plus binary cross-entropy on the closed logit."""
    import torch
    torch.manual_seed(seed)
    x = np.vstack([episode_features(e) for e in episodes]) if x is None else x
    y = np.vstack([episode_labels(e, chunk) for e in episodes]) if y is None else y
    moves = y[..., :3].reshape(-1, 3)
    model = _model_class()(x.mean(0), np.maximum(x.std(0), _X_FLOOR), moves.mean(0),
                           np.maximum(moves.std(0), _Y_FLOOR), chunk, hidden)
    xt, yt = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    target_moves = (yt[..., :3] - model.y_mean) / model.y_std
    bce = torch.nn.BCEWithLogitsLoss()

    def loss_of(idx):
        out = model.normalized(xt[idx])
        return (out[..., :3] - target_moves[idx]).abs().mean() + bce(out[..., 3], yt[idx][..., 3])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        losses = [float(loss_of(torch.arange(len(xt))))]
    for _ in range(epochs):
        order = torch.randperm(len(xt), generator=gen)
        total, n = 0.0, 0
        for i in range(0, len(xt), batch_size):
            idx = order[i:i + batch_size]
            loss = loss_of(idx)
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


class OnnxPolicyV2:
    """The exported network under ONNX Runtime: `actions(x)` like `torch_actions`."""

    def __init__(self, path) -> None:
        import onnxruntime as ort
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def actions(self, x) -> np.ndarray:
        return self._session.run(["actions"], {"observation": np.asarray(x, np.float32)})[0]


class PolicyV2:
    """Harness policy (Env observation dict -> 6 joint targets) around a network `actions_fn` ((B, 6) -> (B, H, 4)).

    `execute`: "ensemble" blends all recent chunks every step; an integer k re-plans every k steps and plays the
    chunk's targets open loop in between.

    `observe(obs, joints, ee)` returns the tuple of batched inputs for `actions_fn`; the default is the state policy's
    cube-relative features. The image policy (image_policy_v2) passes the camera image and the arm state instead."""

    def __init__(self, actions_fn, m: float = ENSEMBLE_M, execute="ensemble", controller=None, observe=None) -> None:
        self._actions, self._m, self._execute = actions_fn, m, execute
        self._observe = observe or (lambda obs, joints, ee: (features(joints, obs["cube_pose"], ee)[None],))
        self._controller = _controller() if controller is None else controller
        self.reset()

    def reset(self) -> None:
        self._t = 0
        self._chunks = []            # (start step, absolute targets (H, 3), closed probabilities (H,))
        self._closed = False

    def __call__(self, obs) -> np.ndarray:
        joints = np.asarray(obs["joint_pos"], float)
        ee = self._controller.gripper_position(joints)
        t = self._t
        replan = self._execute == "ensemble" or not self._chunks or t - self._chunks[-1][0] >= int(self._execute)
        if replan:
            out = np.asarray(self._actions(*self._observe(obs, joints, ee))[0], float)
            self._chunks.append((t, ee + out[:, :3], 1.0 / (1.0 + np.exp(-out[:, 3]))))
        horizon = len(self._chunks[-1][1])
        self._chunks = [c for c in self._chunks if t - c[0] < horizon]
        if self._execute == "ensemble":
            w = np.exp(-self._m * np.arange(len(self._chunks)))          # i = 0 is the oldest chunk
            target = sum(wi * c[1][t - c[0]] for wi, c in zip(w, self._chunks)) / w.sum()
            p = float(sum(wi * c[2][t - c[0]] for wi, c in zip(w, self._chunks)) / w.sum())
        else:
            start, targets, probs = self._chunks[-1]
            target, p = targets[t - start], float(probs[t - start])
        if self._closed and p < OPEN_BELOW:
            self._closed = False
        elif not self._closed and p > CLOSE_ABOVE:
            self._closed = True
        self._t += 1
        return self._controller.command(joints, target, self._closed)


_ = ARM_JOINTS
