"""Dataset conversion for the v2 policy (PRD-v2, ticket 19): clean recorded sessions, and check that the v2 action
labels reproduce each demo (the label gate).

v2 action labels are not stored: they are the gripper-frame position the arm ACHIEVED one control step later (forward
kinematics of the recorded state) plus open/closed from the recorded gripper target. The policy-path controller
(`controller.GripperController`) turns such a target back into joint targets. The gate replays every demo from its
initial state with those labels through the controller and counts lifts from MuJoCo state.

  python -m teleop_il.convert build OUT SESSION [SESSION ...]   # clean record sessions into a NEW folder
  python -m teleop_il.convert gate DATA [--required 0.95] [--keep OUT]

Cleaning (sessions recorded before the two 2026-11-15 recording fixes): an episode the assist grasped at its first
row (stale gripper toggle) is dropped; a stale closed-gripper first row is dropped and the episode starts from the
next recorded state. Arm joint targets are kept as recorded (v2 labels do not use them).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

from teleop_il.dataset import MANIFEST, DatasetWriter, Episode, read_dataset

CLOSED_BELOW = 0.4                 # a recorded gripper target below this (rad) counts as "closed"
PUSH_M = 0.005                     # the cube moved more than this before the first close: the demo pushed it
EXTRA_STEPS = 100                  # a replay may run this many control steps past the demo's own length
_ARRAYS = ("time", "qpos", "qvel", "joint_targets", "cube_pose", "tracking_lost", "assist_active")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gripper_closed(e: Episode) -> np.ndarray:
    """(T,) bool: the recorded gripper command at each step, closed or not."""
    return np.asarray(e.joint_targets)[:, 5] < CLOSED_BELOW


def gripper_path(e: Episode, model=None) -> np.ndarray:
    """(T, 3): where the gripper frame was at each recorded row (forward kinematics of the recorded state)."""
    from teleop_il.env import build_model
    model = build_model() if model is None else model
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    out = np.empty((e.steps, 3))
    for t, q in enumerate(e.qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out[t] = data.site_xpos[site]
    return out


def pushed_before_grasp(e: Episode) -> bool:
    """True if the cube moved more than 5 mm sideways before the gripper was first commanded closed (such contacts
    are chaotic: an open-loop replay cannot be expected to push it the same way)."""
    closed = gripper_closed(e)
    first = int(np.argmax(closed)) if closed.any() else e.steps
    if first == 0:
        return False
    xy = np.asarray(e.cube_pose)[:first, :2]
    return bool(np.linalg.norm(xy - xy[0], axis=1).max() > PUSH_M)


def clean(e: Episode) -> Episode | None:
    """The episode without recording artefacts, or None if it must be dropped (see the module docstring)."""
    if e.assist_active[0]:
        return None
    t = e.joint_targets
    if e.steps > 1 and t[0, 5] < CLOSED_BELOW <= t[1, 5]:
        from teleop_il.env import Env
        lo, hi = Env().action_bounds
        init = {**e.initial_state, "qpos": e.qpos[1].copy(), "qvel": e.qvel[1].copy(), "time": float(e.time[1]),
                "ctrl": np.clip(t[0], lo, hi), "qacc_warmstart": np.zeros_like(e.initial_state["qacc_warmstart"]),
                "hold_steps": 0, "succeeded": False}
        return dataclasses.replace(e, initial_state=init, **{k: getattr(e, k)[1:] for k in _ARRAYS})
    return e


def replay(e: Episode, env=None, controller=None) -> bool:
    """Replay the demo with its v2 labels (achieved gripper position one step ahead, open/closed) through the
    controller, from the demo's initial state. True if the cube is lifted (MuJoCo state)."""
    from teleop_il.controller import GripperController
    from teleop_il.env import Env
    env = Env() if env is None else env
    controller = GripperController() if controller is None else controller
    path, closed = gripper_path(e, env.model), gripper_closed(e)
    last = e.steps - 1
    env.restore(e.initial_state)
    for t in range(e.steps + EXTRA_STEPS):
        env.step(controller.command(env.data.qpos[env._arm_qadr], path[min(t + 1, last)], bool(closed[min(t, last)])))
        if env.success():
            return True
    return False


def _replay_one(e: Episode) -> bool:
    return replay(e)


def build(sources, out) -> Path:
    """Clean the episodes of the record-session folders `sources` (read only) into the NEW folder `out`."""
    sources = [Path(s) for s in sources]
    datasets = [read_dataset(s) for s in sources]
    writer = DatasetWriter(out)
    counts = {"in": 0, "dropped_auto_grasp": 0, "dropped_stale_row0": 0, "kept": 0}
    for ds in datasets:
        for e in ds.episodes:
            counts["in"] += 1
            c = clean(e)
            if c is None:
                counts["dropped_auto_grasp"] += 1
                continue
            counts["dropped_stale_row0"] += c.steps != e.steps
            counts["kept"] += 1
            writer.add(c)
    writer.close({"cleaning": {"sources": [{"folder": str(s), "manifest_sha256": _sha256(s / MANIFEST)}
                                           for s in sources],
                               "method": "drop assist active at row 0; drop stale closed-gripper row 0",
                               "counts": counts}})
    return Path(out)


def gate(data, required: float = 0.95, keep=None, workers: int = 1) -> int:
    """Replay every demo in `data`; exit status 0 only if at least `required` of them lift. With `keep`, the demos
    that replay are written to that NEW folder."""
    data = Path(data)
    ds = read_dataset(data)
    eps = ds.episodes
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            ok = pool.map(_replay_one, eps)
    else:
        ok = [replay(e) for e in eps]
    ok = np.array(ok, bool)
    pushed = np.array([pushed_before_grasp(e) for e in eps], bool)
    n, k = len(eps), int(ok.sum())
    passed = k >= required * n
    print(f"label gate: {k}/{n} demos replay to a lift through the controller (required {required:.0%}): "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"  demos that did not push the cube before the first close: {int(ok[~pushed].sum())}/{int((~pushed).sum())}"
          f"; demos that did: {int(ok[pushed].sum())}/{int(pushed.sum())}")
    failed = [f"{e.seed}{' (pushed)' if p else ''}" for e, good, p in zip(eps, ok, pushed) if not good]
    if failed:
        print("  failed seeds: " + ", ".join(failed))
    if keep is not None:
        from teleop_il.controller import DAMPING, ITERATIONS, ORIENTATION_WEIGHT
        writer = DatasetWriter(keep)
        for e, good in zip(eps, ok):
            if good:
                writer.add(e)
        writer.close({"replay_gate": {"source": {"folder": str(data), "manifest_sha256": _sha256(data / MANIFEST)},
                                      "episodes": n, "replayed": k, "pushed_before_grasp": int(pushed.sum()),
                                      "labels": "achieved gripper position one step ahead + open/closed",
                                      "controller": {"iterations": ITERATIONS, "orientation_weight": ORIENTATION_WEIGHT,
                                                     "damping": DAMPING}}})
        print(f"  kept {k} replayed demos in {keep}")
    return 0 if passed else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Dataset conversion for the v2 policy.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="clean record sessions into a new folder")
    b.add_argument("out", type=Path)
    b.add_argument("sources", type=Path, nargs="+")
    g = sub.add_parser("gate", help="replay every demo through the controller with v2 labels")
    g.add_argument("data", type=Path)
    g.add_argument("--required", type=float, default=0.95)
    g.add_argument("--keep", type=Path)
    args = ap.parse_args(argv)
    if args.command == "build":
        out = build(args.sources, args.out)
        print(f"cleaned into {out}: {read_dataset(out).manifest['cleaning']['counts']}")
        return 0
    return gate(args.data, args.required, args.keep, workers=max(1, (os.cpu_count() or 2) - 1))


if __name__ == "__main__":
    sys.exit(main())
