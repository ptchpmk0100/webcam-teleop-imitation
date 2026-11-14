"""Train the state policy and export it (ticket 09). Local CPU runs on scripted-expert data; real runs on Colab/Kaggle.

  python -m teleop_il.train_state --expert-episodes 100          # record expert data first, then train
  python -m teleop_il.train_state --data runs/expert_data_...    # train on an existing dataset folder

Writes a new folder under runs/: policy.pt (PyTorch state dict), policy.onnx, losses.json, eval.json. The held-out
score is a pipeline smoke result here (the policy is judged in ticket 12, on human demos).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def record_expert_dataset(folder: Path, episodes: int, first_seed: int = 0) -> Path:
    from teleop_il.dataset import DatasetWriter, record_episode
    from teleop_il.env import Env
    from teleop_il.expert import ExpertPolicy
    env, expert, writer = Env(), ExpertPolicy(), DatasetWriter(folder)
    for seed in range(first_seed, first_seed + episodes):
        writer.add(record_episode(env, expert, seed=seed, source="expert"))
    writer.close()
    return folder


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="dataset folder (manifest.json + episodes)")
    src.add_argument("--expert-episodes", type=int, help="record this many scripted-expert episodes first")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seeds", type=int, default=100, help="how many of the 100 held-out seeds to score on")
    args = ap.parse_args(argv)

    import torch

    from teleop_il.dataset import read_dataset, split
    from teleop_il.evaluation import HELD_OUT_SEEDS, evaluate
    from teleop_il.policy import OnnxPolicy, chunk_targets, episode_observations, export_onnx, torch_actions, train

    stamp = time.strftime("%Y%m%d_%H%M%S")
    data = args.data or record_expert_dataset(REPO / "runs" / f"expert_data_{stamp}", args.expert_episodes)
    ds = read_dataset(data)
    by_file = dict(zip((e["file"] for e in ds.manifest["episodes"]), ds.episodes))
    train_files, val_files = split(ds.manifest["episodes"], val_fraction=0.2, seed=args.seed)
    keep = lambda files: [by_file[f] for f in files if by_file[f].success and not by_file[f].flagged]
    train_eps, val_eps = keep(train_files), keep(val_files)

    out = REPO / "runs" / f"state_policy_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    result = train(train_eps, epochs=args.epochs, seed=args.seed)
    seconds = time.time() - t0
    x_val = np.vstack([episode_observations(e) for e in val_eps])
    y_val = np.vstack([chunk_targets(e) for e in val_eps])
    val_err = float(np.abs(torch_actions(result.model, x_val) - y_val).mean())
    torch.save(result.model.state_dict(), out / "policy.pt")
    export_onnx(result.model, out / "policy.onnx")
    with open(out / "losses.json", "x") as f:
        json.dump({"train_losses": result.losses, "val_mean_abs_error_rad": val_err, "data": str(data),
                   "train_episodes": len(train_eps), "val_episodes": len(val_eps), "epochs": args.epochs,
                   "seconds": seconds}, f)
    seeds = HELD_OUT_SEEDS[:args.eval_seeds]
    score = evaluate(OnnxPolicy(out / "policy.onnx"), seeds=seeds)
    with open(out / "eval.json", "x") as f:
        json.dump({"successes": score.successes, "n": score.n, "ci": [score.ci_low, score.ci_high],
                   "per_seed": list(score.per_seed)}, f)
    print(f"data {data} ({len(train_eps)} train, {len(val_eps)} validation episodes)")
    print(f"trained {args.epochs} epochs in {seconds:.0f} s: loss {result.losses[0]:.4f} -> {result.losses[-1]:.4f}; "
          f"validation mean abs error {val_err:.4f} rad")
    print(f"held-out score (ONNX policy, pipeline smoke): {score.successes}/{score.n} "
          f"(95% CI {score.ci_low:.2f}-{score.ci_high:.2f})")
    print(f"run folder: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
