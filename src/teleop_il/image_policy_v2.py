"""Image policy v2 (ticket 16): the v2 state-policy design with a camera image instead of the cube pose.

Same output as `policy_v2`: a chunk of CHUNK gripper moves (metres, relative to where the gripper is now) and a
"closed" logit per step, executed by the same rollout (chunk blending, hysteresis, policy-path controller). The network
has to find the cube in the picture: its inputs are the policy-camera image and the arm's own state, never the cube pose.

ONNX contract: "image" float32 (batch, 96, 128, 3), raw RGB 0..255 as rendered; "proprio" float32 (batch, 9) = the six
arm joint positions followed by the gripper-frame position (forward kinematics of the joints, computed in the rollout
exactly as the state policy does); output "actions" float32 (batch, CHUNK, 4). Normalisation is inside the graph.

This file needs only numpy and torch to train, so a notebook on Kaggle or Colab can run it on its own:
    python image_policy_v2.py fit ARRAYS_DIR OUT_DIR [--epochs N] [--seed S]
`ARRAYS_DIR` holds train.npz and val.npz made by `python -m teleop_il.train_image_v2 prepare`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

IMAGE_HW = (96, 128)
PROPRIO_SIZE = 9
CHUNK = 20                         # same as policy_v2.CHUNK (a test checks they agree)
ACTION_SIZE = 4
_X_FLOOR, _Y_FLOOR = 1e-3, 1e-4


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

    class ImagePolicyV2(nn.Module):
        def __init__(self, p_mean, p_std, y_mean, y_std, chunk: int = CHUNK, channels: int = 32,
                     hidden: int = 256) -> None:
            super().__init__()
            for name, value in (("p_mean", p_mean), ("p_std", p_std), ("y_mean", y_mean), ("y_std", y_std)):
                self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
            self.chunk = chunk
            h, w = IMAGE_HW[0] // 2, IMAGE_HW[1] // 2
            self.cnn = nn.Sequential(
                nn.Conv2d(3, channels, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                SpatialSoftmax(h, w))
            self.head = nn.Sequential(nn.Linear(2 * channels + PROPRIO_SIZE, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, chunk * ACTION_SIZE))

        def normalized(self, image, proprio):
            x = image.permute(0, 3, 1, 2) / 255.0 - 0.5
            z = torch.cat([self.cnn(x), (proprio - self.p_mean) / self.p_std], dim=1)
            return self.head(z).view(-1, self.chunk, ACTION_SIZE)

        def forward(self, image, proprio):      # raw image and proprio in; moves in metres and the closed logit out
            out = self.normalized(image, proprio)
            return torch.cat([out[..., :3] * self.y_std + self.y_mean, out[..., 3:]], dim=-1)

    return ImagePolicyV2


def build_model(chunk: int = CHUNK, **kwargs):
    """A model with placeholder statistics; load_state_dict restores the real ones (they are buffers)."""
    return _model_class()(np.zeros(PROPRIO_SIZE), np.ones(PROPRIO_SIZE), np.zeros(3), np.ones(3), chunk, **kwargs)


def random_shift(images, pad: int, generator=None):
    """DrQ-style augmentation: each image moves by up to `pad` pixels in x and y (edge pixels repeat). (B,H,W,3)."""
    import torch
    if pad <= 0:
        return images
    b, h, w, _ = images.shape
    x = torch.nn.functional.pad(images.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode="replicate")
    dx = torch.randint(0, 2 * pad + 1, (b,), generator=generator)
    dy = torch.randint(0, 2 * pad + 1, (b,), generator=generator)
    out = torch.stack([x[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(b)])
    return out.permute(0, 2, 3, 1)


def train(images, proprio, labels, epochs: int = 100, seed: int = 0, chunk: int = CHUNK, batch_size: int = 128,
          lr: float = 1e-3, shift: int = 4, device: str | None = None, log=None, channels: int = 32):
    """Behaviour cloning: L1 on normalised moves plus binary cross-entropy on the closed logit (as policy_v2.train).
    images (N,96,128,3) uint8, proprio (N,9) float, labels (N,chunk,4) float. Returns (model on CPU, losses)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if not (len(images) == len(proprio) == len(labels)):
        raise ValueError(f"{len(images)} images, {len(proprio)} proprio rows, {len(labels)} labels")
    proprio, labels = np.asarray(proprio, np.float32), np.asarray(labels, np.float32)
    moves = labels[..., :3].reshape(-1, 3)
    model = build_model(chunk, channels=channels)
    for name, value in (("p_mean", proprio.mean(0)), ("p_std", np.maximum(proprio.std(0), _X_FLOOR)),
                        ("y_mean", moves.mean(0)), ("y_std", np.maximum(moves.std(0), _Y_FLOOR))):
        getattr(model, name).copy_(torch.as_tensor(value))
    model = model.to(device)
    img = torch.from_numpy(np.ascontiguousarray(images))
    on_gpu = device != "cpu" and img.numel() < 0.4 * torch.cuda.get_device_properties(0).total_memory   # pictures on the card if they fit
    if on_gpu:
        img = img.to(device)
    pt, yt = torch.from_numpy(proprio).to(device), torch.from_numpy(labels).to(device)
    target_moves = (yt[..., :3] - model.y_mean) / model.y_std
    bce = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)

    def batch_loss(idx, augment):
        x = img[idx.to(img.device)].to(device).float()
        if augment:
            x = random_shift(x, shift)
        out = model.normalized(x, pt[idx.to(device)])
        j = idx.to(device)
        return (out[..., :3] - target_moves[j]).abs().mean() + bce(out[..., 3], yt[j][..., 3])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.1)
    with torch.no_grad():
        first = torch.randperm(len(img), generator=gen)[:1024]
        losses = [float(batch_loss(first, False))]
    t0 = time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(img), generator=gen)
        total, n = 0.0, 0
        for i in range(0, len(img), batch_size):
            idx = order[i:i + batch_size]
            loss = batch_loss(idx, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(idx), n + len(idx)
        sched.step()
        losses.append(total / n)
        if log:
            log(f"epoch {epoch + 1}/{epochs} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)")
    model = model.cpu().eval()
    return model, losses


def torch_actions(model, images, proprio) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.as_tensor(np.asarray(images, np.float32)),
                     torch.as_tensor(np.asarray(proprio, np.float32))).numpy()


def export_onnx(model, path) -> Path:
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; exports go to new files")
    batch = torch.export.Dim("batch", min=1, max=4096)
    example = (torch.zeros(2, *IMAGE_HW, 3), torch.zeros(2, PROPRIO_SIZE))
    program = torch.onnx.export(model, example, input_names=["image", "proprio"], output_names=["actions"],
                                dynamic_shapes=({0: batch}, {0: batch}), dynamo=True)
    program.save(str(path))
    return path


def load_model(pt_path, chunk: int = CHUNK, channels: int = 32):
    import torch
    model = build_model(chunk, channels=channels)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    return model.eval()


class OnnxImagePolicyV2:
    """The exported network under ONNX Runtime: `actions(images, proprio)` like `torch_actions`."""

    def __init__(self, path) -> None:
        import onnxruntime as ort
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def actions(self, images, proprio) -> np.ndarray:
        return self._session.run(["actions"], {"image": np.asarray(images, np.float32),
                                               "proprio": np.asarray(proprio, np.float32)})[0]


def image_rollout_policy(path, render, m=None, execute="ensemble"):
    """Harness policy (Env observation -> 6 joint targets) for an exported image policy. `render` is Env.render: the
    policy sees the camera image and its own joints, never the cube pose."""
    from teleop_il.policy_v2 import ENSEMBLE_M, PolicyV2

    net = OnnxImagePolicyV2(path)

    def observe(obs, joints, ee):
        return (np.asarray(render(), np.float32)[None], np.r_[joints, ee].astype(np.float32)[None])

    return PolicyV2(net.actions, m=ENSEMBLE_M if m is None else m, execute=execute, observe=observe)


def _fit_main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="image_policy_v2.py fit")
    ap.add_argument("arrays", type=Path)
    ap.add_argument("out", type=Path, help="NEW folder for policy.pt and fit.json")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)
    import torch
    with np.load(args.arrays / "train.npz") as z:
        images, proprio, labels = z["images"], z["proprio"], z["labels"]
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"{len(images)} training frames; device {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    t0 = time.time()
    model, losses = train(images, proprio, labels, epochs=args.epochs, seed=args.seed, shift=args.shift,
                          batch_size=args.batch_size, log=lambda s: print(s, flush=True))
    torch.save(model.state_dict(), args.out / "policy.pt")
    with open(args.out / "fit.json", "x") as f:
        json.dump({"epochs": args.epochs, "seed": args.seed, "shift": args.shift, "frames": len(images),
                   "seconds": time.time() - t0, "losses": losses}, f)
    print(f"saved {args.out / 'policy.pt'}: loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        sys.exit(_fit_main(sys.argv[2:]))
    sys.exit("usage: python image_policy_v2.py fit ARRAYS_DIR OUT_DIR [--epochs N] [--seed S]")
