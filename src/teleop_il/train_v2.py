"""Train state policy v2 (PRD-v2) on a dataset folder, export ONNX, score it with the evaluation harness.

  python -m teleop_il.train_v2 --data runs/expert_data_20261114_235944 --required 0.9     # ticket 20 design gate
  python -m teleop_il.train_v2 --data runs/v2_human_replayed_20261115                     # ticket 21 (dev seeds)

Scores on the development seeds unless --held-out is given (held-out seeds only at the PRD-v2 checkpoints). The null
checks run in the same command. Writes a NEW folder runs/v2_policy_<time>/ (policy.pt, policy.onnx, run.json). Exit
status 1 if --required is given and the score is below it, or if a null check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _score_part(args):
    path, seeds, m, execute = args
    from teleop_il.evaluation import evaluate
    from teleop_il.policy_v2 import OnnxPolicyV2, PolicyV2
    return list(evaluate(PolicyV2(OnnxPolicyV2(path).actions, m=m, execute=execute), seeds=seeds).per_seed)


def score(path, seeds, m, execute, workers):
    """Per-seed outcomes of the exported policy, the seeds split over worker processes."""
    from multiprocessing import Pool
    parts = [seeds[i::workers] for i in range(workers)]
    with Pool(workers) as pool:
        results = pool.map(_score_part, [(str(path), p, m, execute) for p in parts])
    outcome = {}
    for p, r in zip(parts, results):
        outcome.update(zip(p, r))
    return [outcome[s] for s in seeds]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--train-episodes", type=int, default=None, help="use at most this many training episodes")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m", type=float, default=None, help="ensembling weight exponent")
    ap.add_argument("--execute", default="ensemble", help="'ensemble' or k (re-plan every k steps)")
    ap.add_argument("--held-out", action="store_true", help="score on the held-out seeds (checkpoints only)")
    ap.add_argument("--required", type=float, default=None)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(4)
    from teleop_il.dataset import read_dataset, split
    from teleop_il.evaluation import DEV_SEEDS, HELD_OUT_SEEDS, null_checks, wilson_interval
    from teleop_il import policy_v2 as pv

    chunk = args.chunk or pv.CHUNK
    m = pv.ENSEMBLE_M if args.m is None else args.m
    execute = args.execute if args.execute == "ensemble" else int(args.execute)
    ds = read_dataset(args.data)
    by_file = dict(zip((e["file"] for e in ds.manifest["episodes"]), ds.episodes))
    train_files, val_files = split(ds.manifest["episodes"], val_fraction=0.2, seed=0)
    train_eps = [by_file[f] for f in train_files][:args.train_episodes]
    val_eps = [by_file[f] for f in val_files]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = REPO / "runs" / f"v2_policy_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    result = pv.train(train_eps, epochs=args.epochs, seed=args.seed, chunk=chunk, hidden=args.hidden)
    seconds = time.time() - t0
    x_val = np.vstack([pv.episode_features(e) for e in val_eps])
    y_val = np.vstack([pv.episode_labels(e, chunk) for e in val_eps])
    pred = pv.torch_actions(result.model, x_val)
    val_move_mm = float(1000 * np.abs(pred[..., :3] - y_val[..., :3]).mean())
    val_first_mm = float(1000 * np.linalg.norm(pred[:, 0, :3] - y_val[:, 0, :3], axis=1).mean())
    val_grip = float(((pred[..., 3] > 0) == (y_val[..., 3] > 0.5)).mean())
    torch.save(result.model.state_dict(), out / "policy.pt")
    pv.export_onnx(result.model, out / "policy.onnx")

    seeds = list(HELD_OUT_SEEDS if args.held_out else DEV_SEEDS)
    outcomes = score(out / "policy.onnx", seeds, m, execute, workers=max(1, min(10, (os.cpu_count() or 2) - 1)))
    k, n = sum(outcomes), len(outcomes)
    lo, hi = wilson_interval(k, n)
    problems = null_checks()
    passed = (args.required is None or k >= args.required * n) and not problems
    run = {"data": str(args.data), "train_episodes": len(train_eps), "val_episodes": len(val_eps),
           "epochs": args.epochs, "chunk": chunk, "hidden": args.hidden, "seed": args.seed, "m": m,
           "execute": args.execute, "seconds": seconds, "losses": result.losses,
           "val_move_mean_abs_mm": val_move_mm, "val_first_move_error_mm": val_first_mm,
           "val_gripper_accuracy": val_grip, "seeds": "held-out" if args.held_out else "development",
           "successes": k, "n": n, "ci": [lo, hi], "per_seed": outcomes, "null_check_problems": problems,
           "required": args.required}
    with open(out / "run.json", "x") as f:
        json.dump(run, f)
    print(f"data {args.data}: {len(train_eps)} train, {len(val_eps)} validation episodes; chunk {chunk}, "
          f"execute {args.execute}, m {m}")
    print(f"trained {args.epochs} epochs in {seconds:.0f} s: loss {result.losses[0]:.3f} -> {result.losses[-1]:.3f}; "
          f"validation: move error {val_move_mm:.2f} mm (first step {val_first_mm:.2f} mm), gripper {val_grip:.1%}")
    print(f"{run['seeds']} seeds: {k}/{n} lifted (95% CI {lo:.2f}-{hi:.2f})"
          + ("" if args.required is None else f", required {args.required:.0%}"))
    print("null checks: " + ("PASS" if not problems else "FAIL: " + "; ".join(problems)))
    print(("PASS" if passed else "FAIL") + f"; run folder: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
