"""C++ ONNX policy parity (ticket 14).

`python -m teleop_il.cpp_policy_parity [RUN_FOLDER]` (default: the newest runs/state_policy_*) checks, and exits 0
only if all hold:
- the C++ ONNX Runtime output matches PyTorch within 1e-4 on recorded observations (frozen threshold);
- C++ and Python rollouts of the ONNX policy agree on success on at least 98 of the 100 held-out seeds, with every
  final cube height within 5 mm (frozen thresholds);
- the C++ run passes the null checks: do-nothing lifts 0 of 100, random actions at most 2 of 100.
Each run writes a new folder under runs/.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from teleop_il.cpp_parity import REPO, Outcome, compare
from teleop_il.env import CONTROL_DT, CUBE_HALF, HOME_ARM, HOME_GRIPPER, LIFT_HOLD_SECONDS, LIFT_THRESHOLD, Env
from teleop_il.evaluation import MAX_STEPS, NULL_HOLD_MAX_SUCCESSES, NULL_RANDOM_MAX_SUCCESSES

ONNX_TOLERANCE = 1e-4


def _run(cmd) -> None:
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({result.returncode}): {result.stderr.strip()}")


def cpp_infer(exe, onnx_file, observations, run_dir) -> np.ndarray:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savetxt(run_dir / "observations.txt", np.asarray(observations, np.float32), fmt="%.9g")
    _run([exe, "infer", onnx_file, run_dir / "observations.txt", run_dir / "cpp_actions.txt"])
    return np.loadtxt(run_dir / "cpp_actions.txt", ndmin=2).reshape(-1, 10, 6)


def cpp_policy_rollout(exe, policy, seeds, run_dir, max_steps: int = MAX_STEPS) -> list[Outcome]:
    """`policy`: an .onnx path, "hold" or "random:SEED". Returns one Outcome per seed, read from the C++ results."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    env = Env()
    mujoco.mj_saveModel(env.model, str(run_dir / "scene.mjb"), None)
    with open(run_dir / "episodes.txt", "x") as f:
        f.write("home " + " ".join(f"{v:.17g}" for v in (*HOME_ARM, HOME_GRIPPER)) + "\n")
        f.write(f"cube_half {CUBE_HALF:.17g}\nlift_threshold {LIFT_THRESHOLD:.17g}\n")
        f.write(f"hold_steps {round(LIFT_HOLD_SECONDS / CONTROL_DT)}\nsubsteps {env._substeps}\n")
        f.write(f"max_steps {max_steps}\nepisodes {len(seeds)}\n")
        for seed in seeds:
            cube = env.reset(seed)["cube_pose"]
            f.write(f"episode {seed} {cube[0]:.17g} {cube[1]:.17g}\n")
    _run([exe, "rollout", run_dir / "scene.mjb", policy, run_dir / "episodes.txt", run_dir / "cpp_results.txt"])
    out = []
    for line in (run_dir / "cpp_results.txt").read_text().splitlines():
        seed, success, cube_z, lift, _steps = line.split()
        out.append(Outcome(int(seed), success == "1", float(cube_z), float(lift)))
    return out


def python_policy_rollout(policy, seeds, max_steps: int = MAX_STEPS) -> list[Outcome]:
    """The evaluation harness's loop, keeping the final cube height for the comparison."""
    env, out = Env(), []
    for seed in seeds:
        obs = env.reset(seed)
        if hasattr(policy, "reset"):
            policy.reset()
        for _ in range(max_steps):
            obs = env.step(policy(obs))
            if env.success():
                break
        out.append(Outcome(int(seed), bool(env.success()), float(env.cube_position()[2]), float(env.lift_height())))
    return out


def main(argv=None) -> int:
    import json

    import torch

    from teleop_il.cpp_parity import build
    from teleop_il.dataset import read_dataset
    from teleop_il.evaluation import HELD_OUT_SEEDS
    from teleop_il.policy import OnnxPolicy, _model_class, episode_observations, torch_actions

    argv = sys.argv[1:] if argv is None else argv
    runs = sorted((REPO / "runs").glob("state_policy_*"))
    run = Path(argv[0]) if argv else (runs[-1] if runs else None)
    if run is None:
        print("no runs/state_policy_* folder; train one first (python -m teleop_il.train_state ...)")
        return 2
    onnx_file = run / "policy.onnx"
    info = json.loads((run / "losses.json").read_text())
    state = torch.load(run / "policy.pt")
    model = _model_class()(state["x_mean"], state["x_std"], state["y_mean"], state["y_std"],
                           hidden=state["net.0.weight"].shape[0])
    model.load_state_dict(state)
    model.eval()
    x = np.vstack([episode_observations(e) for e in read_dataset(info["data"]).episodes[:10]])

    out = REPO / "runs" / f"cpp_policy_parity_{time.strftime('%Y%m%d_%H%M%S')}"
    exe = build(REPO / "build" / "cpp").parent / "policy_rollout"
    problems = []
    gap = float(np.abs(cpp_infer(exe, onnx_file, x, out / "infer") - torch_actions(model, x)).max())
    if gap > ONNX_TOLERANCE:
        problems.append(f"C++ ONNX Runtime differs from PyTorch by {gap:.3g} (allowed {ONNX_TOLERANCE:g})")
    py = python_policy_rollout(OnnxPolicy(onnx_file), HELD_OUT_SEEDS)
    cpp = cpp_policy_rollout(exe, onnx_file, HELD_OUT_SEEDS, out / "policy")
    problems += compare(py, cpp)
    hold = sum(o.success for o in cpp_policy_rollout(exe, "hold", HELD_OUT_SEEDS, out / "hold"))
    rand = sum(o.success for o in cpp_policy_rollout(exe, "random:0", HELD_OUT_SEEDS, out / "random"))
    if hold > NULL_HOLD_MAX_SUCCESSES:
        problems.append(f"C++ do-nothing policy lifted {hold}/100 (allowed {NULL_HOLD_MAX_SUCCESSES})")
    if rand > NULL_RANDOM_MAX_SUCCESSES:
        problems.append(f"C++ random policy lifted {rand}/100 (allowed {NULL_RANDOM_MAX_SUCCESSES})")
    same = sum(a.success == b.success for a, b in zip(py, cpp))
    print(f"policy {run.name}: {len(x)} recorded observations, C++ ONNX Runtime vs PyTorch max gap {gap:.2e}")
    print(f"held-out rollouts: python {sum(o.success for o in py)}/100, C++ {sum(o.success for o in cpp)}/100, "
          f"outcome agrees on {same}/100, largest cube height gap "
          f"{1000 * max(abs(a.cube_z - b.cube_z) for a, b in zip(py, cpp)):.4f} mm")
    print(f"C++ null checks: do-nothing {hold}/100, random {rand}/100")
    print(f"run folder: {out}")
    print("C++ policy parity: " + ("PASS" if not problems else "FAIL\n  " + "\n  ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
