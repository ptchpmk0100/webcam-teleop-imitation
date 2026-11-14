"""Image policy (ticket 15): a small CNN with a spatial-softmax keypoint layer reads the policy camera image, an MLP
combines its keypoints with the joint positions and predicts a chunk of the next 10 joint targets.

The cube pose is NOT an input: the network has to find the cube in the picture. Images come from replay.render_dataset
for training and from Env.render at evaluation (the same camera, checked equal in ticket 08).

ONNX contract: inputs "image" float32 (batch, 96, 128, 3), raw RGB 0..255 as rendered, and "joints" float32 (batch, 6);
output "actions" float32 (batch, 10, 6) absolute joint targets. Pixel scaling and joint/action normalisation are inside
the graph. At run time the chunk's first action is executed, like the state policy.
"""
from __future__ import annotations

import numpy as np

from teleop_il.env import IMAGE_SIZE
from teleop_il.policy import ACTION_SIZE, CHUNK, STD_FLOOR, TrainResult, _arm_qadr, chunk_targets

JOINTS = 6


def episode_joints(e) -> np.ndarray:
    """(T, 6) joint positions for every recorded step (state before that step's action)."""
    return e.qpos[:, _arm_qadr()].astype(np.float32)


def _model_class():
    import torch
    from torch import nn

    class SpatialSoftmax(nn.Module):
        """Each feature map becomes the expected (x, y) of its softmax: where in the image that feature is."""

        def __init__(self, height: int, width: int) -> None:
            super().__init__()
            self.register_buffer("xs", torch.linspace(-1.0, 1.0, width).reshape(1, 1, 1, width))
            self.register_buffer("ys", torch.linspace(-1.0, 1.0, height).reshape(1, 1, height, 1))

        def forward(self, f):
            b, c, h, w = f.shape
            p = torch.softmax(f.reshape(b, c, h * w), dim=-1).reshape(b, c, h, w)
            return torch.cat([(p * self.xs).sum(dim=(2, 3)), (p * self.ys).sum(dim=(2, 3))], dim=1)

    class ImagePolicy(nn.Module):
        def __init__(self, j_mean, j_std, y_mean, y_std, channels: int = 32, hidden: int = 256) -> None:
            super().__init__()
            for name, value in (("j_mean", j_mean), ("j_std", j_std), ("y_mean", y_mean), ("y_std", y_std)):
                self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
            h, w = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
            self.cnn = nn.Sequential(
                nn.Conv2d(3, channels, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                SpatialSoftmax(h, w))
            self.head = nn.Sequential(nn.Linear(2 * channels + JOINTS, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, CHUNK * ACTION_SIZE))

        def normalized(self, image, joints):
            x = image.permute(0, 3, 1, 2) / 255.0 - 0.5
            z = torch.cat([self.cnn(x), (joints - self.j_mean) / self.j_std], dim=1)
            return self.head(z).view(-1, CHUNK, ACTION_SIZE)

        def forward(self, image, joints):           # raw image and joints in, raw joint targets out
            return self.normalized(image, joints) * self.y_std + self.y_mean

    return ImagePolicy


def train_image(episodes, images, epochs: int = 50, seed: int = 0, batch_size: int = 64, lr: float = 1e-3,
                threads: int | None = None) -> TrainResult:
    """Behaviour cloning on (image, joints) -> normalised action chunk. `images[i]` is (T, 96, 128, 3) uint8 for
    `episodes[i]`, as replay.read_images returns it."""
    import torch
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    img = torch.from_numpy(np.vstack(images))                      # uint8, converted per batch to save memory
    joints = np.vstack([episode_joints(e) for e in episodes])
    y = np.vstack([chunk_targets(e) for e in episodes]).astype(np.float32)
    if len(img) != len(joints):
        raise ValueError(f"{len(img)} images for {len(joints)} recorded steps")
    j_mean, j_std = joints.mean(0), np.maximum(joints.std(0), STD_FLOOR)
    y_mean, y_std = y.reshape(-1, ACTION_SIZE).mean(0), np.maximum(y.reshape(-1, ACTION_SIZE).std(0), STD_FLOOR)
    model = _model_class()(j_mean, j_std, y_mean, y_std)
    jt, yt = torch.from_numpy(joints), torch.from_numpy(y)
    yn = (yt - model.y_mean) / model.y_std

    def full_loss():
        with torch.no_grad():
            total = 0.0
            for i in range(0, len(img), 256):
                total += float(((model.normalized(img[i:i + 256].float(), jt[i:i + 256]) - yn[i:i + 256]) ** 2)
                               .mean()) * len(jt[i:i + 256])
            return total / len(img)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    losses = [full_loss()]
    for _ in range(epochs):
        order = torch.randperm(len(img), generator=gen)
        total = 0.0
        for i in range(0, len(img), batch_size):
            idx = order[i:i + batch_size]
            loss = ((model.normalized(img[idx].float(), jt[idx]) - yn[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(img))
    model.eval()
    return TrainResult(model=model, losses=losses)


def image_torch_actions(model, images, joints) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.as_tensor(np.asarray(images, np.float32)),
                     torch.as_tensor(np.asarray(joints, np.float32))).numpy()


def export_image_onnx(model, path):
    import torch
    from pathlib import Path
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; exports go to new files")
    batch = torch.export.Dim("batch", min=1, max=4096)
    example = (torch.zeros(2, *IMAGE_SIZE, 3), torch.zeros(2, JOINTS))
    program = torch.onnx.export(model, example, input_names=["image", "joints"], output_names=["actions"],
                                dynamic_shapes=({0: batch}, {0: batch}), dynamo=True)
    program.save(str(path))
    return path


class ImageOnnxPolicy:
    """Runs the exported image policy with ONNX Runtime. `render` returns the current policy-camera image (Env.render);
    the observation dict supplies only the joint positions."""

    def __init__(self, path, render) -> None:
        import onnxruntime as ort
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._render = render

    def actions(self, images, joints) -> np.ndarray:
        return self._session.run(["actions"], {"image": np.asarray(images, np.float32),
                                               "joints": np.asarray(joints, np.float32)})[0]

    def __call__(self, obs) -> np.ndarray:
        image = self._render()[None].astype(np.float32)
        joints = np.asarray(obs["joint_pos"], np.float32)[None]
        return self.actions(image, joints)[0, 0].astype(np.float64)
