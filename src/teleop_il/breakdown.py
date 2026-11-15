"""Failure breakdown (ticket 18, slimmed after policy v2 replaced v1): why each rollout of a policy failed, read from
the MuJoCo state of the rollout, one class per episode.

  python -m teleop_il.breakdown RUN_FOLDER [--held-out]     # a runs/v2_policy_* folder; development seeds by default

Classes, first match wins: "lifted" (the harness's success); "raised, dropped or held too short" (the cube's bottom
rose more than 1 cm at some step); "closed at the cube, not raised" (the gripper was commanded closed while its frame
was within 2 cm of the cube's grasp point); "near, never closed" (it came that near, but never closed there); "never
near". Exit status 1 if the lifted count differs from the run's own score on the same seeds, a class row is missing
or a null check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CLASSES = ("lifted", "raised, dropped or held too short", "closed at the cube, not raised", "near, never closed",
           "never near")
NEAR_M = 0.02          # gripper frame to the cube's grasp point
RAISED_M = 0.01        # cube bottom above the table
CLOSED_BELOW = 0.4     # commanded gripper joint target, as in `convert`


def classify(distance, closed, lift_height, success: bool) -> str:
    """`distance`, `closed`, `lift_height`: per step of one rollout (gripper frame to grasp point in metres, gripper
    commanded closed, cube bottom above the table in metres)."""
    distance, closed = np.asarray(distance, float), np.asarray(closed, bool)
    near = distance < NEAR_M
    if success:
        return CLASSES[0]
    if np.max(lift_height, initial=0.0) > RAISED_M:
        return CLASSES[1]
    if (near & closed).any():
        return CLASSES[2]
    if near.any():
        return CLASSES[3]
    return CLASSES[4]


def breakdown(policy, seeds, env=None) -> list[tuple[int, str]]:
    """Run `policy` like `evaluation.evaluate` and classify each episode."""
    from teleop_il.controller import GripperController
    from teleop_il.env import CUBE_HALF, Env
    from teleop_il.evaluation import MAX_STEPS
    from teleop_il.expert import GRASP_Z, grasp_point

    env = Env() if env is None else env
    fk = GripperController()
    rows = []
    for seed in seeds:
        obs = env.reset(seed)
        if hasattr(policy, "reset"):
            policy.reset()
        distance, closed, lift = [], [], []
        for _ in range(MAX_STEPS):
            action = np.asarray(policy(obs), float)
            obs = env.step(action)
            cube = np.asarray(obs["cube_pose"], float)
            goal = grasp_point(cube[:3], z=float(cube[2]) + GRASP_Z - CUBE_HALF)
            distance.append(float(np.linalg.norm(fk.gripper_position(obs["joint_pos"]) - goal)))
            closed.append(bool(action[5] < CLOSED_BELOW))
            lift.append(float(env.lift_height()))
            if env.success():
                break
        rows.append((int(seed), classify(distance, closed, lift, bool(env.success()))))
    return rows


def _part(args):
    path, seeds, m, execute = args
    from teleop_il.policy_v2 import OnnxPolicyV2, PolicyV2
    return breakdown(PolicyV2(OnnxPolicyV2(path).actions, m=m, execute=execute), seeds)


def main(argv=None) -> int:
    from multiprocessing import Pool

    from teleop_il.evaluation import DEV_SEEDS, HELD_OUT_SEEDS, null_checks

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--held-out", action="store_true", help="the held-out seeds (only for a checkpoint's own run)")
    args = ap.parse_args(argv)
    info = json.loads((args.run / "run.json").read_text())
    execute = info["execute"] if info["execute"] == "ensemble" else int(info["execute"])
    seeds = list(HELD_OUT_SEEDS if args.held_out else DEV_SEEDS)
    workers = max(1, min(10, (os.cpu_count() or 2) - 1))
    parts = [seeds[i::workers] for i in range(workers)]
    with Pool(workers) as pool:
        found = dict(r for part in pool.map(_part, [(str(args.run / "policy.onnx"), p, info["m"], execute)
                                                     for p in parts]) for r in part)
    rows = [(s, found[s]) for s in seeds]
    problems = []
    if len(rows) != len(seeds):
        problems.append(f"{len(rows)} rows for {len(seeds)} seeds")
    seed_set = "held-out" if args.held_out else "development"
    if info["seeds"] == seed_set and [c == CLASSES[0] for _, c in rows] != list(info["per_seed"]):
        problems.append("the lifted episodes differ from the run's own score on the same seeds")
    problems += null_checks()
    print(f"failure breakdown of {args.run.name} on the {seed_set} seeds ({len(rows)} episodes):")
    for name in CLASSES:
        members = [s for s, c in rows if c == name]
        shown = "" if name == CLASSES[0] or not members else "  seeds " + ", ".join(map(str, members[:12])) + \
            (" ..." if len(members) > 12 else "")
        print(f"  {name:<36} {len(members):>3}{shown}")
    failures = [(name, sum(c == name for _, c in rows)) for name in CLASSES[1:]]
    largest = max(failures, key=lambda f: f[1])
    if largest[1]:
        print(f"largest failure class: {largest[0]} ({largest[1]})")
    print("PASS" if not problems else "FAIL\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
