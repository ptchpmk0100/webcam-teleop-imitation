"""C++ parity for state policy v2 (ticket 24; contract v1 stays in `cpp_policy_parity`).

`python -m teleop_il.cpp_policy_v2_parity [RUN_FOLDER]` (default: the newest runs/v2_policy_*) checks, and exits 0
only if all hold:
- the C++ ONNX Runtime output matches PyTorch within 1e-4 on recorded observations;
- the C++ controller matches `GripperController` within 1e-9 (joint targets in radians, gripper positions in metres)
  on recorded states, with the next recorded gripper position as the target;
- C++ and Python rollouts of the ONNX policy agree on success on 100 of the 100 held-out seeds, every final cube
  height within 5 mm;
- the C++ null checks: do-nothing lifts 0 of 100, random actions at most 2 of 100 (the harness's frozen limits).
Each run writes a new folder under runs/. The rollouts are a parity check, not a score: the policy and its settings
are fixed before it runs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from teleop_il.cpp_parity import REPO, Outcome, compare
from teleop_il.cpp_policy_parity import _run, python_policy_rollout
from teleop_il.env import CONTROL_DT, CUBE_HALF, HOME_ARM, HOME_GRIPPER, LIFT_HOLD_SECONDS, LIFT_THRESHOLD, Env, build_model
from teleop_il.evaluation import MAX_STEPS, NULL_HOLD_MAX_SUCCESSES, NULL_RANDOM_MAX_SUCCESSES

ONNX_TOLERANCE = 1e-4
CONTROLLER_TOLERANCE = 1e-9
REQUIRED_AGREEMENT = 100


def _params(run_dir: Path, chunk: int, m: float | None) -> Path:
    from teleop_il import controller as c
    from teleop_il import expert, policy_v2 as pv
    from teleop_il.ik import GRIPPER_CLOSED, GRIPPER_OPEN
    path = run_dir / "params.txt"
    with open(path, "x") as f:
        f.write(f"iterations {c.ITERATIONS}\norientation_weight {c.ORIENTATION_WEIGHT:.17g}\ndamping {c.DAMPING:.17g}\n")
        f.write(f"gripper_open {GRIPPER_OPEN:.17g}\ngripper_closed {GRIPPER_CLOSED:.17g}\n")
        f.write("home " + " ".join(f"{v:.17g}" for v in HOME_ARM) + "\n")
        f.write(f"grasp_z {expert.GRASP_Z:.17g}\njaw_offset {expert.JAW_OFFSET:.17g}\ncube_half {CUBE_HALF:.17g}\n")
        f.write(f"chunk {chunk}\nm {pv.ENSEMBLE_M if m is None else m:.17g}\nclose_above {pv.CLOSE_ABOVE:.17g}\n"
                f"open_below {pv.OPEN_BELOW:.17g}\n")
    return path


def _controller_model(run_dir: Path) -> Path:
    path = run_dir / "controller.mjb"
    mujoco.mj_saveModel(build_model(with_cube=False), str(path), None)
    return path


def cpp_infer_v2(exe, onnx_file, observations, run_dir) -> np.ndarray:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savetxt(run_dir / "observations.txt", np.asarray(observations, np.float32), fmt="%.9g")
    _run([exe, "infer", onnx_file, run_dir / "observations.txt", run_dir / "cpp_actions.txt"])
    out = np.loadtxt(run_dir / "cpp_actions.txt", ndmin=2)
    return out.reshape(len(out), -1, 4)


def cpp_controller(exe, joints, targets, closed, run_dir, chunk: int = 20):
    """The C++ controller on each (joints, target, closed): returns (joint targets (N, 6), gripper positions (N, 3))."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    cases = np.column_stack([np.asarray(joints, float)[:, :6], np.asarray(targets, float),
                             np.asarray(closed, float)])
    np.savetxt(run_dir / "cases.txt", cases, fmt="%.17g")
    _run([exe, "controller", _controller_model(run_dir), _params(run_dir, chunk, None), run_dir / "cases.txt",
          run_dir / "cpp_out.txt"])
    out = np.loadtxt(run_dir / "cpp_out.txt", ndmin=2)
    return out[:, :6], out[:, 6:9]


def cpp_policy_v2_rollout(exe, policy, seeds, run_dir, chunk: int = 20, m: float | None = None,
                          max_steps: int = MAX_STEPS) -> list[Outcome]:
    """`policy`: a contract-v2 .onnx path, "hold" or "random:SEED". One Outcome per seed, read from the C++ results."""
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
    _run([exe, "rollout", run_dir / "scene.mjb", _controller_model(run_dir), policy, _params(run_dir, chunk, m),
          run_dir / "episodes.txt", run_dir / "cpp_results.txt"])
    out = []
    for line in (run_dir / "cpp_results.txt").read_text().splitlines():
        seed, success, cube_z, lift, _steps = line.split()
        out.append(Outcome(int(seed), success == "1", float(cube_z), float(lift)))
    return out


def recorded_cases(episodes):
    """Controller cases from recorded demos: the measured joints, the next recorded gripper position, open/closed."""
    from teleop_il.convert import gripper_closed
    from teleop_il.policy_v2 import _arm_rows, _path
    joints, targets, closed = [], [], []
    for e in episodes:
        q, path = _arm_rows(e), _path(e)
        joints.append(q[:-1])
        targets.append(path[1:])
        closed.append(gripper_closed(e)[:-1])
    return np.vstack(joints), np.vstack(targets), np.concatenate(closed)


def main(argv=None) -> int:
    import json

    import torch

    from teleop_il.controller import GripperController
    from teleop_il.cpp_parity import build
    from teleop_il.dataset import read_dataset
    from teleop_il.evaluation import HELD_OUT_SEEDS
    from teleop_il.policy_v2 import OnnxPolicyV2, PolicyV2, _model_class, episode_features, torch_actions

    argv = sys.argv[1:] if argv is None else argv
    runs = sorted((REPO / "runs").glob("v2_policy_*"))
    run = Path(argv[0]) if argv else (runs[-1] if runs else None)
    if run is None:
        print("no runs/v2_policy_* folder; train one first (python -m teleop_il.train_v2 ...)")
        return 2
    info = json.loads((run / "run.json").read_text())
    if info["execute"] != "ensemble":
        print(f"{run.name} was run with execute={info['execute']}; the C++ rollout implements 'ensemble' only")
        return 2
    chunk, onnx_file = int(info["chunk"]), run / "policy.onnx"
    state = torch.load(run / "policy.pt")
    model = _model_class()(state["x_mean"], state["x_std"], state["y_mean"], state["y_std"], chunk,
                           state["net.0.weight"].shape[0])
    model.load_state_dict(state)
    model.eval()
    episodes = read_dataset(info["data"]).episodes[:10]
    x = np.vstack([episode_features(e) for e in episodes])

    out = REPO / "runs" / f"cpp_policy_v2_parity_{time.strftime('%Y%m%d_%H%M%S')}"
    exe = build(REPO / "build" / "cpp").parent / "policy_v2_rollout"
    problems = []
    gap = float(np.abs(cpp_infer_v2(exe, onnx_file, x, out / "infer") - torch_actions(model, x)).max())
    if gap > ONNX_TOLERANCE:
        problems.append(f"C++ ONNX Runtime differs from PyTorch by {gap:.3g} (allowed {ONNX_TOLERANCE:g})")
    joints, targets, closed = recorded_cases(episodes)
    cpp_cmd, cpp_ee = cpp_controller(exe, joints, targets, closed, out / "controller", chunk)
    ctl = GripperController()
    py_cmd = np.array([ctl.command(q, t, c) for q, t, c in zip(joints, targets, closed)])
    py_ee = np.array([ctl.gripper_position(q) for q in joints])
    ctl_gap = float(max(np.abs(cpp_cmd - py_cmd).max(), np.abs(cpp_ee - py_ee).max()))
    if ctl_gap > CONTROLLER_TOLERANCE:
        problems.append(f"C++ controller differs from Python by {ctl_gap:.3g} (allowed {CONTROLLER_TOLERANCE:g})")
    m = float(info["m"])
    py = python_policy_rollout(PolicyV2(OnnxPolicyV2(onnx_file).actions, m=m), HELD_OUT_SEEDS)
    cpp = cpp_policy_v2_rollout(exe, onnx_file, HELD_OUT_SEEDS, out / "policy", chunk, m)
    problems += compare(py, cpp, required=REQUIRED_AGREEMENT)
    hold = sum(o.success for o in cpp_policy_v2_rollout(exe, "hold", HELD_OUT_SEEDS, out / "hold", chunk))
    rand = sum(o.success for o in cpp_policy_v2_rollout(exe, "random:0", HELD_OUT_SEEDS, out / "random", chunk))
    if hold > NULL_HOLD_MAX_SUCCESSES:
        problems.append(f"C++ do-nothing policy lifted {hold}/100 (allowed {NULL_HOLD_MAX_SUCCESSES})")
    if rand > NULL_RANDOM_MAX_SUCCESSES:
        problems.append(f"C++ random policy lifted {rand}/100 (allowed {NULL_RANDOM_MAX_SUCCESSES})")
    same = sum(a.success == b.success for a, b in zip(py, cpp))
    print(f"policy {run.name}: {len(x)} recorded observations, C++ ONNX Runtime vs PyTorch max gap {gap:.2e}")
    print(f"controller on {len(joints)} recorded states: C++ vs Python max gap {ctl_gap:.2e}")
    print(f"held-out rollouts: python {sum(o.success for o in py)}/100, C++ {sum(o.success for o in cpp)}/100, "
          f"outcome agrees on {same}/100, largest cube height gap "
          f"{1000 * max(abs(a.cube_z - b.cube_z) for a, b in zip(py, cpp)):.4f} mm")
    print(f"C++ null checks: do-nothing {hold}/100, random {rand}/100")
    print(f"run folder: {out}")
    print("C++ policy v2 parity: " + ("PASS" if not problems else "FAIL\n  " + "\n  ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
