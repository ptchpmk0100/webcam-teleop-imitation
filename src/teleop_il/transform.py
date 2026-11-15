"""Transformed copies of demos (PRD-v2, ticket 22; MimicGen-style, simplified).

A copy takes one demo's gripper path (the positions the arm achieved, as the v2 labels) and its open/closed commands,
and moves the path to the cube position of a NEW seed. The move turns the path about the arm base by the change in the
cube's bearing and shifts it radially by the change in its distance, so that after the approach the gripper keeps the
demo's offset from the cube, turned with the pan joint as the gripper itself turns. The move is blended in over the
approach (none at the first step, where the arm starts at home; all of it from the step the gripper first comes within
2 cm of the grasp point sideways, or the first close if that is sooner). Heights and timing are kept.

Each copy is replayed through the policy-path controller from the new seed's reset state and recorded like a demo
(rows before each step, until the lift succeeds); it is kept only if the cube lifts (MuJoCo state). Copies use seeds
from 1000 up, below 10000, skipping the demos' own seeds (development and held-out seeds are 10000 and above).

  python -m teleop_il.transform DATA OUT [--copies N]     # writes the NEW folder OUT

The manifest's "transform" entry names the source folder (and its manifest SHA-256), the method, the counts, and for
every copy its source demo file and seed. `train_copies` picks the copies of training-split demos only, so that a
validation demo never reaches training through its copies.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

from teleop_il.dataset import MANIFEST, DatasetWriter, Episode, read_dataset

FIRST_COPY_SEED = 1000
NEAR_M = 0.02                      # the move is complete once the gripper is this close to the grasp point sideways
EXTRA_STEPS = 100                  # as in convert.replay: a copy may run this many steps past the demo's length
METHOD = ("gripper path turned about the base by the change in cube bearing and shifted radially by the change in "
          "distance, blended in (smoothstep) until the gripper is within 2 cm of the grasp point sideways or the "
          "first close; replayed through the policy-path controller from the new seed's reset; kept only if it lifts")


def copy_seeds(n: int, exclude) -> list[int]:
    """`n` distinct seeds from 1000 up, below 10000, none in `exclude`. ValueError if there are not enough."""
    exclude = set(exclude)
    seeds = [s for s in range(FIRST_COPY_SEED, 10_000) if s not in exclude][:n]
    if len(seeds) < n:
        raise ValueError(f"only {len(seeds)} copy seeds below 10000, {n} asked for")
    return seeds


def _turn(xy: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def transform_path(path, cube0, cube1, ramp_end: int) -> np.ndarray:
    """The (T, 3) gripper `path`, recorded with the cube at `cube0`, moved for a cube at `cube1` (see the module
    docstring). From step `ramp_end` on, path - cube1 equals the demo's path - cube0 turned by the bearing change."""
    path = np.asarray(path, float)
    r0, r1 = math.hypot(cube0[0], cube0[1]), math.hypot(cube1[0], cube1[1])
    th0 = math.atan2(cube0[1], cube0[0])
    dth = math.atan2(cube1[1], cube1[0]) - th0
    ramp_end = max(1, int(ramp_end))
    out = path.copy()
    for t, p in enumerate(path):
        s = min(t / ramp_end, 1.0)
        w = s * s * (3 - 2 * s)
        a = th0 + w * dth
        out[t, :2] = _turn(p[:2], w * dth) + w * (r1 - r0) * np.array([math.cos(a), math.sin(a)])
    return out


def ramp_end(path, closed, cube0) -> int:
    """The step by which the move must be complete: the gripper first within 2 cm of the grasp point sideways, or the
    first close if that comes first (the demo's length if it never closes)."""
    from teleop_il.expert import grasp_point
    first_close = int(np.argmax(closed)) if np.any(closed) else len(path)
    goal = grasp_point(cube0)[:2]
    near = np.nonzero(np.linalg.norm(np.asarray(path)[:first_close, :2] - goal, axis=1) < NEAR_M)[0]
    return max(1, int(near[0]) if len(near) else first_close)


def make_copy(e: Episode, seed: int, env=None, controller=None) -> Episode | None:
    """The demo `e` moved to the cube of `seed`, replayed and recorded; None if the cube is not lifted."""
    from teleop_il.controller import GripperController
    from teleop_il.convert import gripper_closed, gripper_path
    from teleop_il.env import Env
    env = Env() if env is None else env
    controller = GripperController() if controller is None else controller
    path, closed = gripper_path(e, env.model), gripper_closed(e)
    cube0 = np.asarray(e.cube_pose[0][:3], float)
    obs = env.reset(seed)
    target = transform_path(path, cube0, obs["cube_pose"][:3], ramp_end(path, closed, cube0))
    initial = env.snapshot()
    last = e.steps - 1
    rows = {k: [] for k in ("time", "qpos", "qvel", "joint_targets", "cube_pose")}
    for t in range(e.steps + EXTRA_STEPS):
        rows["time"].append(float(env.data.time))
        rows["qpos"].append(env.data.qpos.copy())
        rows["qvel"].append(env.data.qvel.copy())
        rows["cube_pose"].append(np.asarray(obs["cube_pose"], float))
        action = controller.command(obs["joint_pos"], target[min(t + 1, last)], bool(closed[min(t, last)]))
        rows["joint_targets"].append(action)
        obs = env.step(action)
        if env.success():
            n = len(rows["time"])
            return Episode(seed=int(seed), source="transformed", success=True, flagged=False, assist_fired=0,
                           initial_state=initial, **{k: np.array(v) for k, v in rows.items()},
                           tracking_lost=np.zeros(n, bool), assist_active=np.zeros(n, bool))
    return None


_WORKER = {}


def _init_worker(data: str) -> None:
    _WORKER["episodes"] = read_dataset(data).episodes


def _copy_job(job):
    index, seed = job
    if "env" not in _WORKER:
        from teleop_il.controller import GripperController
        from teleop_il.env import Env
        _WORKER["env"], _WORKER["controller"] = Env(), GripperController()
    return make_copy(_WORKER["episodes"][index], seed, _WORKER["env"], _WORKER["controller"])


def generate(data, out, copies_per_demo: int = 10, workers: int = 1) -> Path:
    """Try `copies_per_demo` copies of every demo in `data` (read only), each on its own new seed; write the copies
    that lift into the NEW folder `out`."""
    data = Path(data)
    ds = read_dataset(data)
    files = [entry["file"] for entry in ds.manifest["episodes"]]
    seeds = copy_seeds(len(ds.episodes) * copies_per_demo, exclude={e.seed for e in ds.episodes})
    jobs = [(i, seeds[i * copies_per_demo + k]) for i in range(len(ds.episodes)) for k in range(copies_per_demo)]
    _init_worker(str(data))
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=_init_worker, initargs=(str(data),)) as pool:
            results = pool.map(_copy_job, jobs, chunksize=1)
    else:
        results = [_copy_job(j) for j in jobs]
    writer = DatasetWriter(out)
    copies, kept_by_source = {}, {}
    for (i, _), c in zip(jobs, results):
        if c is None:
            continue
        name = writer.add(c).name
        copies[name] = {"source_file": files[i], "source_seed": int(ds.episodes[i].seed)}
        kept_by_source[files[i]] = kept_by_source.get(files[i], 0) + 1
    from teleop_il.controller import DAMPING, ITERATIONS, ORIENTATION_WEIGHT
    writer.close({"transform": {
        "source": {"folder": str(data), "manifest_sha256": hashlib.sha256((data / MANIFEST).read_bytes()).hexdigest()},
        "method": METHOD, "copies_per_demo": copies_per_demo, "near_m": NEAR_M,
        "controller": {"iterations": ITERATIONS, "orientation_weight": ORIENTATION_WEIGHT, "damping": DAMPING},
        "counts": {"demos": len(ds.episodes), "tried": len(jobs), "kept": len(copies),
                   "demos_with_a_copy": len(kept_by_source)},
        "copies": copies}})
    return Path(out)


def train_copies(manifest: dict, train_files) -> list[str]:
    """The copy files (of a copies folder's `manifest`) whose source demo is one of `train_files`."""
    train_files = set(train_files)
    return sorted(f for f, c in manifest["transform"]["copies"].items() if c["source_file"] in train_files)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--copies", type=int, default=10, help="copies tried per demo")
    args = ap.parse_args(argv)
    out = generate(args.data, args.out, args.copies, workers=max(1, (os.cpu_count() or 2) - 1))
    c = read_dataset(out).manifest["transform"]["counts"]
    print(f"transformed copies: {c['kept']}/{c['tried']} lifted and kept ({c['kept'] / max(1, c['tried']):.0%}); "
          f"{c['demos_with_a_copy']}/{c['demos']} demos have at least one copy; written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
