"""Train the image policy and export it (ticket 15 code; real training is ticket 16, on Colab or Kaggle).

  python -m teleop_il.train_image --expert-episodes 100            # record expert data, render images, train
  python -m teleop_il.train_image --data DATASET --images IMAGES   # existing dataset and its rendered images

Writes a new folder under runs/: image_policy.pt, image_policy.onnx, losses.json, eval.json (held-out score of the
ONNX policy through the env camera).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")                 # images are rendered offscreen
REPO = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="dataset folder; needs --images")
    src.add_argument("--expert-episodes", type=int, help="record this many scripted-expert episodes first")
    ap.add_argument("--images", type=Path, help="folder rendered from --data by teleop_il.replay")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seeds", type=int, default=100, help="how many of the 100 held-out seeds to score on")
    args = ap.parse_args(argv)
    if args.data and not args.images:
        ap.error("--data needs --images")

    import torch

    from teleop_il.dataset import read_dataset, split
    from teleop_il.env import Env
    from teleop_il.evaluation import HELD_OUT_SEEDS, evaluate
    from teleop_il.image_policy import (ImageOnnxPolicy, episode_joints, export_image_onnx, image_torch_actions,
                                        train_image)
    from teleop_il.policy import chunk_targets
    from teleop_il.replay import read_images, render_dataset
    from teleop_il.train_state import record_expert_dataset

    stamp = time.strftime("%Y%m%d_%H%M%S")
    data, images_dir = args.data, args.images
    if data is None:
        data = record_expert_dataset(REPO / "runs" / f"expert_data_{stamp}", args.expert_episodes)
        images_dir = render_dataset(data, REPO / "runs" / f"expert_images_{stamp}")
    ds = read_dataset(data)
    files = [e["file"] for e in ds.manifest["episodes"]]
    by_file = dict(zip(files, ds.episodes))
    train_files, val_files = split(ds.manifest["episodes"], val_fraction=0.2, seed=args.seed)
    keep = lambda fs: [f for f in fs if by_file[f].success and not by_file[f].flagged]
    train_files, val_files = keep(train_files), keep(val_files)

    out = REPO / "runs" / f"image_policy_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    result = train_image([by_file[f] for f in train_files], [read_images(images_dir, f) for f in train_files],
                         epochs=args.epochs, seed=args.seed)
    seconds = time.time() - t0
    errs = []
    for f in val_files:
        pred = image_torch_actions(result.model, read_images(images_dir, f).astype(np.float32),
                                   episode_joints(by_file[f]))
        errs.append(np.abs(pred - chunk_targets(by_file[f])).mean())
    val_err = float(np.mean(errs)) if errs else float("nan")
    torch.save(result.model.state_dict(), out / "image_policy.pt")
    export_image_onnx(result.model, out / "image_policy.onnx")
    with open(out / "losses.json", "x") as f:
        json.dump({"train_losses": result.losses, "val_mean_abs_error_rad": val_err, "data": str(data),
                   "images": str(images_dir), "train_episodes": len(train_files), "val_episodes": len(val_files),
                   "epochs": args.epochs, "seconds": seconds}, f)
    env = Env()
    score = evaluate(ImageOnnxPolicy(out / "image_policy.onnx", render=env.render),
                     seeds=HELD_OUT_SEEDS[:args.eval_seeds], env=env)
    with open(out / "eval.json", "x") as f:
        json.dump({"successes": score.successes, "n": score.n, "ci": [score.ci_low, score.ci_high],
                   "per_seed": list(score.per_seed)}, f)
    print(f"data {data} + images {images_dir} ({len(train_files)} train, {len(val_files)} validation episodes)")
    print(f"trained {args.epochs} epochs in {seconds:.0f} s: loss {result.losses[0]:.4f} -> {result.losses[-1]:.4f}; "
          f"validation mean abs error {val_err:.4f} rad")
    print(f"held-out score (ONNX image policy): {score.successes}/{score.n} "
          f"(95% CI {score.ci_low:.2f}-{score.ci_high:.2f})")
    print(f"run folder: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
