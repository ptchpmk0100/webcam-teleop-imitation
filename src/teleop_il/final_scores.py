"""Final state-policy scores (ticket 25): every number of the README table from one command.

  make final-scores HUMAN="RUN RUN RUN" COPIES="RUN RUN RUN"
  python -m teleop_il.final_scores --human-only RUN RUN RUN --with-copies RUN RUN RUN

Each RUN is a runs/v2_policy_* folder (one per training seed). On the 100 held-out seeds, for every run: the policy
alone in Python (the headline), the same policy in the C++ rollout, and the policy with the grasp assist (a labelled
secondary number). Next to them: the scripted expert, and the null checks in Python and C++. All training seeds are
printed, none chosen. Writes a NEW folder runs/final_scores_<time>/ (scores.json, C++ rollout files). Exit status 1
if a null check fails or C++ and Python disagree on any seed's outcome.

This is a held-out scoring run: use it only at the ticket 25 checkpoint, with policies and settings already fixed.
The teleop baseline cannot be recomputed (failed attempts were not saved); it is printed as recorded at session
time, with its source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TELEOP_BASELINE = ("human with grasp assist, counted live at record time (not recomputable): 30 of 30 attempts, "
                   "median 8.0 s (ticket 10, first 25 demos); 56 of 58, median 6.9 s (ticket 11, largest session)")


def _score_part(args):
    path, seeds, m, assist = args
    from teleop_il.assist import AssistedPolicy
    from teleop_il.evaluation import evaluate
    from teleop_il.policy_v2 import OnnxPolicyV2, PolicyV2
    policy = PolicyV2(OnnxPolicyV2(path).actions, m=m)
    return list(evaluate(AssistedPolicy(policy) if assist else policy, seeds=seeds).per_seed)


def python_outcomes(onnx_file, seeds, m: float, assist: bool, workers: int) -> list[bool]:
    """Per-seed lifts of the exported policy in the Python harness, alone or with the grasp assist."""
    from multiprocessing import Pool
    parts = [list(seeds[i::workers]) for i in range(workers)]
    with Pool(workers) as pool:
        results = pool.map(_score_part, [(str(onnx_file), p, m, assist) for p in parts])
    outcome = {}
    for p, r in zip(parts, results):
        outcome.update(zip(p, r))
    return [outcome[s] for s in seeds]


def summarise(group: str, rows: list[dict]) -> list[str]:
    """Table lines for one training set: one line per training seed, then the mean."""
    from teleop_il.evaluation import wilson_interval
    lines = []
    for r in rows:
        lo, hi = wilson_interval(r["python"], r["n"])
        lines.append(f"| {group} | {r['seed']} | {r['python']}/{r['n']} ({lo:.2f}-{hi:.2f}) | {r['cpp']}/{r['n']} | "
                     f"{r['agree']}/{r['n']} | {r['assist']}/{r['n']} | {r['run']} |")
    n = len(rows)
    lines.append(f"| {group} | mean of {n} | {sum(r['python'] for r in rows) / n:.1f} | "
                 f"{sum(r['cpp'] for r in rows) / n:.1f} | | {sum(r['assist'] for r in rows) / n:.1f} | |")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--human-only", type=Path, nargs="+", required=True, help="runs trained on the human demos only")
    ap.add_argument("--with-copies", type=Path, nargs="+", required=True, help="runs trained with transformed copies")
    args = ap.parse_args(argv)

    from teleop_il.cpp_parity import build
    from teleop_il.cpp_policy_v2_parity import cpp_policy_v2_rollout
    from teleop_il.evaluation import (HELD_OUT_SEEDS, NULL_HOLD_MAX_SUCCESSES, NULL_RANDOM_MAX_SUCCESSES, evaluate,
                                      null_checks)
    from teleop_il.expert import ExpertPolicy

    seeds = list(HELD_OUT_SEEDS)
    workers = max(1, min(10, (os.cpu_count() or 2) - 1))
    out = REPO / "runs" / f"final_scores_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    exe = build(REPO / "build" / "cpp").parent / "policy_v2_rollout"
    problems, groups = [], {}
    for group, runs in (("100 human demos", args.human_only), ("100 human demos + copies", args.with_copies)):
        rows = []
        for run in runs:
            info = json.loads((run / "run.json").read_text())
            if (group.endswith("copies")) != bool(info.get("extra")):
                problems.append(f"{run.name} is in the wrong group (its run.json says extra={info.get('extra')!r})")
            m, chunk, onnx_file = float(info["m"]), int(info["chunk"]), run / "policy.onnx"
            py = python_outcomes(onnx_file, seeds, m, assist=False, workers=workers)
            assisted = python_outcomes(onnx_file, seeds, m, assist=True, workers=workers)
            cpp = [o.success for o in cpp_policy_v2_rollout(exe, onnx_file, seeds, out / f"cpp_{run.name}", chunk, m)]
            agree = sum(a == b for a, b in zip(py, cpp))
            if agree != len(seeds):
                problems.append(f"{run.name}: C++ and Python agree on {agree}/{len(seeds)} outcomes")
            rows.append({"run": run.name, "seed": info["seed"], "train_episodes": info["train_episodes"],
                         "extra_train_episodes": info.get("extra_train_episodes", 0), "n": len(seeds),
                         "python": sum(py), "cpp": sum(cpp), "agree": agree, "assist": sum(assisted),
                         "python_per_seed": py, "cpp_per_seed": cpp, "assist_per_seed": assisted})
        groups[group] = rows
    expert = evaluate(ExpertPolicy(), seeds=seeds)
    py_null = null_checks()
    problems += py_null
    hold = sum(o.success for o in cpp_policy_v2_rollout(exe, "hold", seeds, out / "cpp_hold"))
    rand = sum(o.success for o in cpp_policy_v2_rollout(exe, "random:0", seeds, out / "cpp_random"))
    if hold > NULL_HOLD_MAX_SUCCESSES or rand > NULL_RANDOM_MAX_SUCCESSES:
        problems.append(f"C++ null checks: do-nothing {hold}/100, random {rand}/100")
    with open(out / "scores.json", "x") as f:
        json.dump({"seeds": "held-out", "groups": groups, "expert": expert.successes,
                   "cpp_null": {"hold": hold, "random": rand}, "python_null_problems": py_null,
                   "teleop_baseline": TELEOP_BASELINE, "problems": problems}, f, indent=1)

    print("Final scores on the 100 held-out seeds (lifts read from MuJoCo state; 95% Wilson interval in brackets)\n")
    print("| training set | training seed | policy alone, Python (headline) | policy alone, C++ | C++ = Python | "
          "policy with assist (secondary) | run |")
    print("|---|---|---|---|---|---|---|")
    for group, rows in groups.items():
        print("\n".join(summarise(group, rows)))
    for group, rows in groups.items():
        print(f"training set '{group}': {rows[0]['train_episodes']} human demos"
              + (f" + {rows[0]['extra_train_episodes']} transformed copies" if rows[0]["extra_train_episodes"] else "")
              + " in training (the other human demos are the validation split)")
    print(f"scripted expert: {expert.successes}/{expert.n} ({expert.ci_low:.2f}-{expert.ci_high:.2f})")
    print(f"teleop baseline: {TELEOP_BASELINE}")
    print("null checks: Python " + ("PASS" if not py_null else "FAIL") + f"; C++ do-nothing {hold}/100, random {rand}/100")
    print(f"run folder: {out}")
    print("final scores: " + ("PASS" if not problems else "FAIL\n  " + "\n  ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
