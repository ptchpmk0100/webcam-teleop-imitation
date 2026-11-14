"""C++ physics parity (ticket 13): build the C++ rollout against the Python wheel's MuJoCo library, replay the same
joint-target sequences in both, and compare cube height and success per seed.

`python -m teleop_il.cpp_parity` runs the acceptance check: the scripted expert's sequences on the 100 held-out
seeds plus 10 each of do-nothing, wild and two-short-lift sequences (so failures, clipping and the success hold are compared too). Exit 0 only if every final cube height
agrees within 5 mm and the success outcome agrees on at least 98 of the expert seeds' and all sequences overall.
Each run writes a new folder under runs/ (scene, inputs, both results); nothing is overwritten.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from teleop_il.env import CUBE_HALF, HOME_ARM, HOME_GRIPPER, LIFT_HOLD_SECONDS, LIFT_THRESHOLD, CONTROL_DT, Env

REPO = Path(__file__).resolve().parents[2]
HEIGHT_TOLERANCE_M = 0.005
REQUIRED_AGREEMENT = 98                     # of 100 seeds (frozen threshold, board)
# Stricter, added in ticket 13: the joint-state trajectory, every control step. End states hide physics changes
# (9 instead of 10 physics steps per control step moved the final cube height by only 0.7 mm). Both sides run the same
# library build on the same inputs, so they should agree to rounding.
TRACE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Sequence:
    seed: int
    cube_xy: tuple
    actions: np.ndarray                     # (T, 6) joint targets


@dataclass(frozen=True)
class Outcome:
    seed: int
    success: bool
    cube_z: float
    lift_height: float
    trace: np.ndarray | None = field(default=None, compare=False, repr=False)   # (T, nq) qpos after each step


def build(build_dir) -> Path:
    """Configure and build the C++ rollout; returns the executable's path."""
    build_dir = Path(build_dir)
    lib_dir = Path(mujoco.__file__).parent
    subprocess.run(["cmake", "-S", str(REPO / "cpp"), "-B", str(build_dir), f"-DMUJOCO_DIR={lib_dir}",
                    "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel"], check=True, capture_output=True, text=True)
    return build_dir / "rollout"


def sequences(expert_seeds, hold_seeds=(), wild_seeds=(), relift_seeds=()) -> list[Sequence]:
    """Joint-target sequences per seed, replayed open-loop by both sides:
    expert: the scripted expert's closed-loop targets (all lift);  hold: do nothing (none lift);
    wild: random targets up to 1.5 times the joint range (clipping, a knocked or tumbling cube);
    relift: two partial lifts of 12 to 15 steps above the threshold each, 27 in total, with the cube put down in
    between (the success hold must restart: the flag stays off)."""
    from teleop_il.dataset import record_episode
    from teleop_il.evaluation import HoldPolicy
    from teleop_il.expert import APPROACH_STEPS, CLOSE_STEPS, DESCEND_STEPS, ExpertPolicy
    env, out = Env(), []

    def add(seed, actions):
        cube = env.reset(seed)["cube_pose"]
        out.append(Sequence(seed=int(seed), cube_xy=(float(cube[0]), float(cube[1])), actions=np.asarray(actions)))

    for seed in expert_seeds:
        add(seed, record_episode(env, ExpertPolicy(), seed=seed, source="scripted").joint_targets)
    for seed in hold_seeds:
        add(seed, record_episode(env, HoldPolicy(), seed=seed, source="scripted").joint_targets)
    lo, hi = env.action_bounds
    for seed in wild_seeds:
        rng = np.random.default_rng(seed)
        add(seed, rng.uniform(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo), size=(300, 6)))
    lift0 = APPROACH_STEPS + DESCEND_STEPS + CLOSE_STEPS
    for seed in relift_seeds:
        a = record_episode(env, ExpertPolicy(), seed=seed, source="scripted").joint_targets
        up = a[lift0:lift0 + 22]
        add(seed, np.vstack([a[:lift0], up, [up[-1]] * 6, up[::-1], [a[lift0]] * 10,
                             up, [up[-1]] * 6, up[::-1], [a[lift0]] * 10]))
    return out


def run_python(seqs) -> list[Outcome]:
    env, out = Env(), []
    for s in seqs:
        env.reset(s.seed)
        trace = []
        for a in s.actions:
            env.step(a)
            trace.append(env.data.qpos.copy())
        out.append(Outcome(s.seed, bool(env.success()), float(env.cube_position()[2]), float(env.lift_height()),
                           np.array(trace)))
    return out


def _write_sequences(path: Path, seqs, substeps: int) -> None:
    hold = round(LIFT_HOLD_SECONDS / CONTROL_DT)
    with open(path, "x") as f:
        f.write("home " + " ".join(f"{v:.17g}" for v in (*HOME_ARM, HOME_GRIPPER)) + "\n")
        f.write(f"cube_half {CUBE_HALF:.17g}\nlift_threshold {LIFT_THRESHOLD:.17g}\nhold_steps {hold}\n")
        f.write(f"substeps {substeps}\nepisodes {len(seqs)}\n")
        for s in seqs:
            f.write(f"episode {s.seed} {s.cube_xy[0]:.17g} {s.cube_xy[1]:.17g} {len(s.actions)}\n")
            for a in s.actions:
                f.write(" ".join(f"{v:.17g}" for v in a) + "\n")


def run_cpp(exe, seqs, run_dir, substeps: int | None = None) -> list[Outcome]:
    """Run the C++ rollout in the NEW folder `run_dir`. `substeps` overrides the physics decimation (for the check
    that a different physics is caught); by default it is the Python env's own."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    env = Env()
    mujoco.mj_saveModel(env.model, str(run_dir / "scene.mjb"), None)
    _write_sequences(run_dir / "sequences.txt", seqs, env._substeps if substeps is None else substeps)
    result = subprocess.run([str(exe), str(run_dir / "scene.mjb"), str(run_dir / "sequences.txt"),
                             str(run_dir / "cpp_results.txt"), str(run_dir / "cpp_trace.txt")],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C++ rollout failed ({result.returncode}): {result.stderr.strip()}")
    rows = np.loadtxt(run_dir / "cpp_trace.txt", ndmin=2)
    out, start = [], 0
    for s, line in zip(seqs, (run_dir / "cpp_results.txt").read_text().splitlines()):
        seed, success, cube_z, lift = line.split()
        trace = rows[start:start + len(s.actions), 2:]
        start += len(s.actions)
        out.append(Outcome(int(seed), success == "1", float(cube_z), float(lift), trace))
    return out


def compare(py, cpp, required: int = REQUIRED_AGREEMENT, tolerance: float = HEIGHT_TOLERANCE_M,
            trace_tolerance: float = TRACE_TOLERANCE) -> list[str]:
    """Problems found (empty = parity holds): seeds out of order, heights apart by more than `tolerance`, fewer
    than `required` matching success outcomes, or (when both carry traces) trajectories apart by more than
    `trace_tolerance` at any step."""
    if [o.seed for o in py] != [o.seed for o in cpp]:
        return [f"seed lists differ: python {[o.seed for o in py][:5]}..., C++ {[o.seed for o in cpp][:5]}..."]
    problems = []
    for a, b in zip(py, cpp):
        if abs(a.cube_z - b.cube_z) > tolerance:
            problems.append(f"seed {a.seed}: cube height python {a.cube_z:.4f} m, C++ {b.cube_z:.4f} m")
        if abs(a.lift_height - b.lift_height) > tolerance:
            problems.append(f"seed {a.seed}: lift height python {a.lift_height:.4f} m, C++ {b.lift_height:.4f} m")
        if a.trace is not None and b.trace is not None:
            gap = np.inf if a.trace.shape != b.trace.shape else float(np.abs(a.trace - b.trace).max())
            if gap > trace_tolerance:
                problems.append(f"seed {a.seed}: trajectories differ by up to {gap:.3g} (allowed {trace_tolerance:g})")
    same = sum(a.success == b.success for a, b in zip(py, cpp))
    if same < required:
        bad = [a.seed for a, b in zip(py, cpp) if a.success != b.success]
        problems.append(f"success outcome agrees on {same} of {len(py)} (need {required}); differs on seeds {bad[:10]}")
    return problems


def main() -> int:
    from teleop_il.evaluation import HELD_OUT_SEEDS
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / "runs" / f"cpp_parity_{stamp}"
    exe = build(REPO / "build" / "cpp")
    seqs = sequences(expert_seeds=HELD_OUT_SEEDS, hold_seeds=HELD_OUT_SEEDS[:10], wild_seeds=HELD_OUT_SEEDS[10:20],
                     relift_seeds=HELD_OUT_SEEDS[20:30])
    py = run_python(seqs)
    cpp = run_cpp(exe, seqs, run_dir)
    n_expert = len(HELD_OUT_SEEDS)
    problems = compare(py[:n_expert], cpp[:n_expert], required=REQUIRED_AGREEMENT)
    problems += compare(py[n_expert:], cpp[n_expert:], required=len(py) - n_expert)
    gap = max(abs(a.cube_z - b.cube_z) for a, b in zip(py, cpp))
    trace_gap = max(float(np.abs(a.trace - b.trace).max()) for a, b in zip(py, cpp))
    with open(run_dir / "python_results.json", "x") as f:
        json.dump([{k: v for k, v in asdict(o).items() if k != "trace"} for o in py], f)    # traces: C++ file
    print(f"C++ vs Python, {len(seqs)} sequences ({n_expert} expert, then 10 each do-nothing, wild, two short lifts): "
          f"largest cube height gap {gap * 1000:.4f} mm, largest trajectory gap {trace_gap:.3g}; success agrees on "
          f"{sum(a.success == b.success for a, b in zip(py, cpp))} of {len(py)}; "
          f"expert lifts python {sum(o.success for o in py[:n_expert])}, C++ {sum(o.success for o in cpp[:n_expert])}")
    print(f"run folder: {run_dir}")
    print("parity: " + ("PASS" if not problems else "FAIL\n  " + "\n  ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
