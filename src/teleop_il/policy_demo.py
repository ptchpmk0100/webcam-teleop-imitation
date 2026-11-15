"""Watch a trained v2 policy lift the cube, live, in the demo window (no hand, no webcam, no assist).

  make policy-demo RUN=runs/v2_policy_...                 # 10 development positions, real time
  make policy-demo RUN=runs/v2_policy_... ARGS="--seeds 20010 20011 --speed 2"

The arm is driven by the policy alone, at 20 Hz in real time; each cube position ends at the lift (plus a short pause)
or after 15 s. Keys in the window: N next position, SPACE pause, Q quit. Lifts are read from MuJoCo state. The
default positions are development seeds; this window is for looking, the scores come from `train_v2`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from teleop_il.env import CONTROL_DT, EPISODE_SECONDS

AFTER_LIFT_STEPS = 20              # keep showing the lifted cube for 1 s before the next position


def rollouts(policy, env, seeds, on_step=None, max_steps: int = round(EPISODE_SECONDS / CONTROL_DT)) -> list[bool]:
    """Run `policy` alone from each seed's reset; one lifted/not per seed (MuJoCo state). `on_step(seed, step, lifts)`
    is called after every control step; returning "next" ends the position, "quit" ends the run."""
    results: list[bool] = []
    for seed in seeds:
        obs = env.reset(seed)
        if hasattr(policy, "reset"):
            policy.reset()
        lifted_at = None
        for t in range(max_steps):
            obs = env.step(policy(obs))
            if env.success() and lifted_at is None:
                lifted_at = t
            answer = on_step(seed, t, results + [env.success()]) if on_step is not None else None
            if answer in ("next", "quit") or (lifted_at is not None and t - lifted_at >= AFTER_LIFT_STEPS):
                break
        results.append(bool(env.success()))
        if on_step is not None and answer == "quit":
            break
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="a runs/v2_policy_... folder (policy.onnx)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20_000, 20_010)), help="cube positions")
    ap.add_argument("--speed", type=float, default=1.0, help="1 is real time")
    args = ap.parse_args(argv)
    if not (args.run / "policy.onnx").exists():
        print(f"no policy.onnx in {args.run}", file=sys.stderr)
        return 2
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2

    from teleop_il.display import Display
    from teleop_il.env import Env
    from teleop_il.policy_v2 import OnnxPolicyV2, PolicyV2
    env = Env()
    policy = PolicyV2(OnnxPolicyV2(args.run / "policy.onnx").actions)
    display = Display(env)
    title = "policy alone (N next, SPACE pause, Q quit)"
    step_s = CONTROL_DT / args.speed
    state = {"next_frame": time.perf_counter()}

    def show(seed, t, lifts):
        lines = ["POLICY ALONE", "no hand, no webcam, no assist", f"policy: {args.run.name}",
                 f"position {len(lifts)} of {len(args.seeds)} (seed {seed})", f"time {(t + 1) * CONTROL_DT:4.1f} s",
                 f"cube height {100 * env.lift_height():4.1f} cm", f"lifted {sum(lifts)} of {len(lifts)}",
                 "N next  SPACE pause  Q quit"]
        cv2.imshow(title, display.compose(None, lines, banner="LIFTED" if env.success() else ""))
        while True:
            wait_ms = max(1, round(1000 * (state["next_frame"] - time.perf_counter())))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord(" "):                                   # pause until any key
                cv2.waitKey(0)
                state["next_frame"] = time.perf_counter()
            elif key in (ord("q"), ord("Q"), 27):
                return "quit"
            elif key in (ord("n"), ord("N")):
                return "next"
            if time.perf_counter() >= state["next_frame"]:
                break
        state["next_frame"] = max(state["next_frame"] + step_s, time.perf_counter() - step_s)
        return None

    try:
        results = rollouts(policy, env, args.seeds, on_step=show)
    finally:
        cv2.destroyAllWindows()
        display.close()
    print(f"policy alone ({args.run.name}): lifted {sum(results)} of {len(results)} positions shown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
