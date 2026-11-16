"""Image policy v2 (ticket 16): prepare arrays, train, export, score.

  python -m teleop_il.train_image_v2 prepare --data runs/v2_human_20261115 --images runs/v2_images_human_20261116 \\
        [--extra COPIES --extra-images COPY_IMAGES] --out runs/image_v2_arrays_X
  python -m teleop_il.train_image_v2 fit runs/image_v2_arrays_X --epochs 100          # local (CPU), or on Kaggle:
  python image_policy_v2.py fit ARRAYS OUT                                             # numpy + torch only
  python -m teleop_il.train_image_v2 score runs/image_v2_policy_X [--held-out]        # exports ONNX, scores 100 seeds

`prepare` uses the same 80/20 split and the same transformed-copy rule as train_v2, so image and state policies see the
same training demos. Scores are on the development seeds unless --held-out is given (checkpoints only). Every stage
writes a NEW folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[2]


def episode_proprio(e) -> np.ndarray:
    """(T, 9): arm joints then gripper-frame position, as the rollout computes them."""
    from teleop_il import policy_v2 as pv
    return np.hstack([pv._arm_rows(e), pv._path(e)]).astype(np.float32)


def episode_arrays(e, images, chunk: int):
    from teleop_il import policy_v2 as pv
    if len(images) != e.steps:
        raise ValueError(f"{len(images)} images for {e.steps} steps")
    return images, episode_proprio(e), pv.episode_labels(e, chunk)


def prepare(data: Path, images: Path, out: Path, extra: Path | None = None, extra_images: Path | None = None,
            chunk: int | None = None) -> Path:
    from teleop_il import policy_v2 as pv
    from teleop_il.dataset import MANIFEST, read_dataset, split
    from teleop_il.replay import read_images
    from teleop_il.transform import train_copies

    chunk = chunk or pv.CHUNK
    ds = read_dataset(data)
    by_file = dict(zip((e["file"] for e in ds.manifest["episodes"]), ds.episodes))
    train_files, val_files = split(ds.manifest["episodes"], val_fraction=0.2, seed=0)
    parts = {"train": [(by_file[f], read_images(images, f)) for f in train_files],
             "val": [(by_file[f], read_images(images, f)) for f in val_files]}
    if extra is not None:
        ex = read_dataset(extra)
        if ex.manifest["transform"]["source"]["manifest_sha256"] != hashlib.sha256((data / MANIFEST).read_bytes()).hexdigest():
            raise ValueError(f"{extra} was not made from {data}")
        ex_by_file = dict(zip((e["file"] for e in ex.manifest["episodes"]), ex.episodes))
        parts["train"] += [(ex_by_file[f], read_images(extra_images, f)) for f in train_copies(ex.manifest, set(train_files))]
    out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name, items in parts.items():
        rows = [episode_arrays(e, im, chunk) for e, im in items]
        with open(out / f"{name}.npz", "xb") as f:
            np.savez_compressed(f, images=np.concatenate([r[0] for r in rows]), proprio=np.vstack([r[1] for r in rows]),
                                labels=np.concatenate([r[2] for r in rows]))
        counts[name] = {"episodes": len(rows), "frames": int(sum(len(r[1]) for r in rows))}
    with open(out / "meta.json", "x") as f:
        json.dump({"data": str(data), "images": str(images), "extra": None if extra is None else str(extra),
                   "chunk": chunk, **counts}, f)
    return out


def _score_part(args):
    path, seeds, m, execute = args
    from teleop_il.env import Env
    from teleop_il.evaluation import evaluate
    from teleop_il.image_policy_v2 import image_rollout_policy
    env = Env()
    return list(evaluate(image_rollout_policy(path, env.render, m=m, execute=execute), seeds=seeds, env=env).per_seed)


def score(path, seeds, m, execute, workers):
    from multiprocessing import get_context
    parts = [seeds[i::workers] for i in range(workers)]
    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(_score_part, [(str(path), p, m, execute) for p in parts])
    outcome = {}
    for p, r in zip(parts, results):
        outcome.update(zip(p, r))
    return [outcome[s] for s in seeds]


def _validation(model, val, chunk):
    from teleop_il.image_policy_v2 import torch_actions
    n = len(val["proprio"])
    pred = np.concatenate([torch_actions(model, val["images"][i:i + 256], val["proprio"][i:i + 256])
                           for i in range(0, n, 256)])
    y = val["labels"]
    return {"val_move_mean_abs_mm": float(1000 * np.abs(pred[..., :3] - y[..., :3]).mean()),
            "val_first_move_error_mm": float(1000 * np.linalg.norm(pred[:, 0, :3] - y[:, 0, :3], axis=1).mean()),
            "val_gripper_accuracy": float(((pred[..., 3] > 0) == (y[..., 3] > 0.5)).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--extra", type=Path)
    p.add_argument("--extra-images", type=Path)
    p.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fit")
    f.add_argument("arrays", type=Path)
    f.add_argument("--epochs", type=int, default=100)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--shift", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("run", type=Path, help="folder with policy.pt (from fit, possibly trained elsewhere)")
    s.add_argument("--arrays", type=Path, help="arrays folder, for the validation error")
    s.add_argument("--held-out", action="store_true")
    s.add_argument("--m", type=float, default=None)
    s.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    s.add_argument("--n", type=int, default=100, help="score only the first n seeds (quick checks)")
    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        if bool(args.extra) != bool(args.extra_images):
            ap.error("--extra and --extra-images go together")
        out = prepare(args.data, args.images, args.out, args.extra, args.extra_images)
        print(f"arrays written to {out}: {(out / 'meta.json').read_text()}")
        return 0

    if args.cmd == "fit":
        from teleop_il import image_policy_v2 as ip
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return ip._fit_main([str(args.arrays), str(REPO / "runs" / f"image_v2_policy_{stamp}"), "--epochs",
                             str(args.epochs), "--seed", str(args.seed), "--shift", str(args.shift)])

    from teleop_il import image_policy_v2 as ip
    from teleop_il import policy_v2 as pv
    from teleop_il.evaluation import DEV_SEEDS, HELD_OUT_SEEDS, wilson_interval
    model = ip.load_model(args.run / "policy.pt")
    onnx = args.run / "policy.onnx"
    if not onnx.exists():
        ip.export_onnx(model, onnx)
    result = {}
    if args.arrays:
        with np.load(args.arrays / "val.npz") as z:
            result.update(_validation(model, {k: z[k] for k in z.files}, pv.CHUNK))
    seeds = list((HELD_OUT_SEEDS if args.held_out else DEV_SEEDS)[:args.n])
    m = pv.ENSEMBLE_M if args.m is None else args.m
    t0 = time.time()
    outcomes = score(onnx, seeds, m, "ensemble", args.workers)
    k, n = sum(outcomes), len(outcomes)
    lo, hi = wilson_interval(k, n)
    result.update({"seeds": "held-out" if args.held_out else "development", "successes": k, "n": n, "ci": [lo, hi],
                   "per_seed": outcomes, "m": m, "score_seconds": time.time() - t0})
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(args.run / f"score_{stamp}.json", "x") as f:
        json.dump(result, f)
    if "val_move_mean_abs_mm" in result:
        print(f"validation: move error {result['val_move_mean_abs_mm']:.2f} mm "
              f"(first step {result['val_first_move_error_mm']:.2f} mm), gripper {result['val_gripper_accuracy']:.1%}")
    print(f"{result['seeds']} seeds: {k}/{n} lifted (95% CI {lo:.2f}-{hi:.2f}); {args.run}/score_{stamp}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
