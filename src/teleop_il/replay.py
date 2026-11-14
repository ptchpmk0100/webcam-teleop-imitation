"""Replay and image render (ticket 08).

Two different uses of a stored episode:
- replay_actions re-simulates from the stored initial state with the stored joint targets, one 20 Hz step each. That is
  exact for expert episodes (recorded with Env.step) and is the consistency check.
- render_dataset draws each step's STORED state, not a re-simulation: human demos are simulated live in 5 ms slices on
  camera timing, so re-running their actions at 20 Hz would not give back the same motion.
Images go into a new folder with their own manifest; the dataset folder is only read.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from teleop_il.dataset import MANIFEST, read_dataset
from teleop_il.env import IMAGE_SIZE, POLICY_CAMERA, Env

IMAGES_MANIFEST = "images_manifest.json"
IMAGES_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ReplayResult:
    success: bool
    cube_pose: np.ndarray          # (T, 7) before each step's action, like Episode.cube_pose
    final_cube_pose: np.ndarray    # (7,) after the last action


def replay_actions(episode, env: Env | None = None) -> ReplayResult:
    env = Env() if env is None else env
    env.restore(episode.initial_state)
    poses = []
    for action in episode.joint_targets:
        poses.append(env.data.qpos[env._cube_qadr:env._cube_qadr + 7].copy())
        env.step(action)
    final = env.data.qpos[env._cube_qadr:env._cube_qadr + 7].copy()
    return ReplayResult(success=bool(env.success()), cube_pose=np.array(poses), final_cube_pose=final)


def _images_name(episode_file: str) -> str:
    return "images_" + Path(episode_file).stem + ".npz"


def render_dataset(source, out, height: int = IMAGE_SIZE[0], width: int = IMAGE_SIZE[1]) -> Path:
    """Render every stored step of every episode in `source` into the NEW folder `out`."""
    source, out = Path(source), Path(out)
    dataset = read_dataset(source)                              # validates the input first
    out.mkdir(parents=True, exist_ok=False)
    env = Env()
    renderer = mujoco.Renderer(env.model, height, width)
    entries = []
    for entry, episode in zip(dataset.manifest["episodes"], dataset.episodes):
        frames = np.empty((episode.steps, height, width, 3), np.uint8)
        for t in range(episode.steps):
            env.data.qpos[:] = episode.qpos[t]
            env.data.qvel[:] = episode.qvel[t]
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=POLICY_CAMERA)
            frames[t] = renderer.render()
        path = out / _images_name(entry["file"])
        with open(path, "xb") as f:
            np.savez_compressed(f, images=frames)
        entries.append({"episode_file": entry["file"], "images_file": path.name, "frames": episode.steps,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    renderer.close()
    manifest = {"format_version": IMAGES_FORMAT_VERSION, "camera": POLICY_CAMERA, "size": [height, width],
                "source": str(source),
                "source_manifest_sha256": hashlib.sha256((source / MANIFEST).read_bytes()).hexdigest(),
                "episodes": entries}
    with open(out / IMAGES_MANIFEST, "x") as f:
        json.dump(manifest, f, indent=1)
    return out


def read_images(folder, episode_file: str) -> np.ndarray:
    """The (T, H, W, 3) uint8 images rendered for one episode file of the source dataset."""
    with np.load(Path(folder) / _images_name(episode_file), allow_pickle=False) as z:
        return z["images"].copy()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Render a dataset's stored states into a new folder of policy images.")
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    render_dataset(args.source, args.out)
    print(f"images written to {args.out}")
    return 0


if __name__ == "__main__":
    import os
    import sys
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.exit(main())
