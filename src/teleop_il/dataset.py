"""Dataset (seam C): one .npz file per episode plus a versioned manifest.json, written once into a new folder.

An episode holds one row per 20 Hz control step, recorded BEFORE that step's joint targets are applied: the time, full
MuJoCo positions and velocities, the joint targets (5 arm joints plus the gripper), the cube pose and per-step flags
(tracking lost, grasp assist active). The initial simulator state is stored too, so replay (ticket 08) can rebuild any
observation, state or image. Episode-level fields: seed, source ("expert", "human", "human+assist"), success read from
MuJoCo state, whether tracking was lost for too long, and how often the assist fired.

Stages write to new folders: a dataset folder is created once and never written again (DECISIONS: each stage writes to
a new folder, inputs stay read-only).
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
_ARRAYS = ("time", "qpos", "qvel", "joint_targets", "cube_pose", "tracking_lost", "assist_active")
_SCALAR_STATE = {"time": float, "hold_steps": int, "succeeded": bool}       # snapshot keys that are not arrays
_ENTRY_KEYS = ("file", "seed", "source", "success", "flagged", "assist_fired", "steps", "sha256")


class DatasetError(ValueError):
    """A dataset or episode that does not match the format. The message names the problem."""


@dataclass(frozen=True)
class Episode:
    seed: int
    source: str
    success: bool
    flagged: bool
    assist_fired: int
    initial_state: dict                      # Env.snapshot() taken right after reset
    time: np.ndarray                         # (T,) seconds of simulated time
    qpos: np.ndarray                         # (T, nq)
    qvel: np.ndarray                         # (T, nv)
    joint_targets: np.ndarray                # (T, 6) commanded at this step, gripper last
    cube_pose: np.ndarray                    # (T, 7) position and quaternion
    tracking_lost: np.ndarray                # (T,) bool
    assist_active: np.ndarray                # (T,) bool

    @property
    def steps(self) -> int:
        return len(self.time)


@dataclass(frozen=True)
class Dataset:
    manifest: dict
    episodes: list = field(default_factory=list)


def record_episode(env, policy, seed: int, source: str, max_steps: int = 300) -> Episode:
    """Run `policy` (obs -> 6 joint targets) from `env.reset(seed)` until success or `max_steps` (15 s at 20 Hz)."""
    obs = env.reset(seed)
    if hasattr(policy, "reset"):
        policy.reset()
    initial = env.snapshot()
    rows = {k: [] for k in ("time", "qpos", "qvel", "joint_targets", "cube_pose")}
    for _ in range(max_steps):
        rows["time"].append(float(env.data.time))
        rows["qpos"].append(env.data.qpos.copy())
        rows["qvel"].append(env.data.qvel.copy())
        rows["cube_pose"].append(np.asarray(obs["cube_pose"], float))
        action = np.asarray(policy(obs), dtype=float)
        rows["joint_targets"].append(action)
        obs = env.step(action)
        if env.success():
            break
    n = len(rows["time"])
    return Episode(seed=int(seed), source=source, success=bool(env.success()), flagged=False, assist_fired=0,
                   initial_state=initial, time=np.array(rows["time"]), qpos=np.array(rows["qpos"]),
                   qvel=np.array(rows["qvel"]), joint_targets=np.array(rows["joint_targets"]),
                   cube_pose=np.array(rows["cube_pose"]), tracking_lost=np.zeros(n, bool),
                   assist_active=np.zeros(n, bool))


def _check_episode(e: Episode) -> None:
    for name in _ARRAYS:
        if len(getattr(e, name)) != e.steps:
            raise DatasetError(f"{name} has {len(getattr(e, name))} rows, time has {e.steps}")
    if e.steps == 0:
        raise DatasetError("episode has no steps")
    if e.joint_targets.shape[1:] != (6,) or e.cube_pose.shape[1:] != (7,):
        raise DatasetError(f"joint_targets {e.joint_targets.shape} or cube_pose {e.cube_pose.shape} has the wrong width")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DatasetWriter:
    """Creates `folder` (it must not exist), writes episodes into it, and the manifest on close()."""

    def __init__(self, folder) -> None:
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=False)         # FileExistsError: a stage never writes twice
        self._entries: list[dict] = []

    def add(self, e: Episode) -> Path:
        _check_episode(e)
        path = self._folder / f"episode_{len(self._entries):05d}.npz"
        arrays = {name: getattr(e, name) for name in _ARRAYS}
        meta = {"seed": e.seed, "source": e.source, "success": e.success, "flagged": e.flagged,
                "assist_fired": e.assist_fired}
        state = {f"init_{k}": np.asarray(v) for k, v in e.initial_state.items()}
        with open(path, "xb") as f:
            np.savez_compressed(f, **arrays, **{k: np.asarray(v) for k, v in meta.items()}, **state)
        self._entries.append({"file": path.name, **meta, "steps": e.steps, "sha256": _sha256(path)})
        return path

    def close(self, extra: dict | None = None) -> Path:
        import mujoco
        manifest = {"format_version": FORMAT_VERSION, "created": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "mujoco": mujoco.__version__, "control_hz": 20, **(extra or {}), "episodes": self._entries}
        path = self._folder / MANIFEST
        with open(path, "x") as f:
            json.dump(manifest, f, indent=1)
        return path


def _load_episode(path: Path) -> Episode:
    with np.load(path, allow_pickle=False) as z:
        state = {}
        for key in z.files:
            if key.startswith("init_"):
                name = key[len("init_"):]
                state[name] = _SCALAR_STATE[name](z[key]) if name in _SCALAR_STATE else z[key].copy()
        return Episode(seed=int(z["seed"]), source=str(z["source"]), success=bool(z["success"]),
                       flagged=bool(z["flagged"]), assist_fired=int(z["assist_fired"]), initial_state=state,
                       **{name: z[name].copy() for name in _ARRAYS})


def read_dataset(folder) -> Dataset:
    """Read and validate a dataset folder. Raises DatasetError naming the first problem found."""
    folder = Path(folder)
    path = folder / MANIFEST
    if not path.exists():
        raise DatasetError(f"no {MANIFEST} in {folder}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise DatasetError(f"{MANIFEST} is not valid JSON: {err}") from err
    if manifest.get("format_version") != FORMAT_VERSION:
        raise DatasetError(f"format_version is {manifest.get('format_version')!r}, expected {FORMAT_VERSION}")
    entries = manifest.get("episodes")
    if not isinstance(entries, list):
        raise DatasetError("manifest has no episodes list")
    seen, episodes = set(), []
    for i, entry in enumerate(entries):
        missing = [k for k in _ENTRY_KEYS if k not in entry]
        if missing:
            raise DatasetError(f"episode entry {i} lacks {', '.join(missing)}")
        name = entry["file"]
        if name in seen:
            raise DatasetError(f"{name} is listed twice")
        seen.add(name)
        file = folder / name
        if not file.exists():
            raise DatasetError(f"{name} is listed but missing")
        if _sha256(file) != entry["sha256"]:
            raise DatasetError(f"{name}: sha256 does not match the manifest (file changed after writing)")
        e = _load_episode(file)
        _check_episode(e)
        for key in ("seed", "source", "success", "flagged", "assist_fired", "steps"):
            if getattr(e, key) != entry[key]:
                raise DatasetError(f"{name}: {key} is {getattr(e, key)!r} in the file, {entry[key]!r} in the manifest")
        episodes.append(e)
    return Dataset(manifest=manifest, episodes=episodes)


def split(index, val_fraction: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """Split manifest entries (dicts with "file" and "seed") into train and validation file lists, by episode.

    Episodes are grouped by cube seed first, so retries of the same cube position never straddle the split."""
    groups: dict[int, list[str]] = {}
    for entry in index:
        groups.setdefault(int(entry["seed"]), []).append(entry["file"])
    seeds = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(seeds))
    n_val = round(len(seeds) * val_fraction)
    val_seeds = {seeds[i] for i in order[:n_val]}
    train = sorted(f for s in seeds if s not in val_seeds for f in groups[s])
    val = sorted(f for s in val_seeds for f in groups[s])
    return train, val


def merge_datasets(sources, out) -> Path:
    """Copy every episode of the `sources` (validated first) into the NEW dataset folder `out`, in order. The manifest
    records each source folder and the SHA-256 of its manifest; the sources are only read."""
    sources = [Path(src) for src in sources]
    datasets = [read_dataset(src) for src in sources]          # a broken source fails before anything is written
    writer = DatasetWriter(out)
    for ds in datasets:
        for e in ds.episodes:
            writer.add(e)
    writer.close({"sources": [{"folder": str(src), "manifest_sha256": _sha256(src / MANIFEST)} for src in sources]})
    return Path(out)


def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Dataset tools.")
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge", help="merge dataset folders (e.g. record sessions) into a new one")
    m.add_argument("out", type=Path)
    m.add_argument("sources", type=Path, nargs="+")
    v = sub.add_parser("check", help="validate a dataset folder and summarise it")
    v.add_argument("folder", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.command == "merge":
            merge_datasets(args.sources, args.out)
        ds = read_dataset(args.out if args.command == "merge" else args.folder)
    except DatasetError as err:
        print(f"INVALID: {err}", file=sys.stderr)
        return 1
    eps = ds.episodes
    seeds = [e.seed for e in eps]
    print(f"valid dataset: {len(eps)} episodes, {len(set(seeds))} distinct cube seeds, sources "
          f"{sorted({e.source for e in eps})}, all lifted: {all(e.success for e in eps)}, "
          f"none flagged: {not any(e.flagged for e in eps)}, steps {sum(e.steps for e in eps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
