# webcam-teleop-imitation

A webcam tracks one hand and drives a simulated SO-101 arm in MuJoCo. The demonstrations recorded that way train an
imitation policy that then picks up a cube by itself, in Python and in a C++ rollout. Success is always read from the
simulator's state (cube bottom more than 5 cm above the table for 1 s), never from the program's own status output.

## Result

State policy v2, **policy alone** (no hand, no webcam, no grasp assist), on 100 held-out cube positions the policy
never saw. Three training seeds, all reported, none chosen.

| training set | seed 0 | seed 1 | seed 2 | mean | same policy in C++ |
|---|---|---|---|---|---|
| 100 human demos | 96/100 | 90/100 | 84/100 | 90.0 | equal on every position |
| 100 human demos + 798 transformed copies | 100/100 | 100/100 | 100/100 | 100.0 | equal on every position |

Next to it: the scripted expert lifts 100/100; a do-nothing policy and a random policy lift 0/100 (null checks, Python
and C++). All of these numbers, with 95 percent intervals, are printed by one command:

```
make final-scores HUMAN="RUN RUN RUN" COPIES="RUN RUN RUN"      # RUN = a runs/v2_policy_* folder per training seed
```

What the numbers do and do not say:
- Of the 100 human demos, 80 are in training and 20 are the validation split. The copies in training are the 626
  made from those 80 demos (798 exist in total).
- The demos were recorded with a grasp assist: the human steers the approach and presses the gripper key near the
  cube, and the assist lines up, descends and closes. **The grasp segment of each demo is the assist's motion**, not
  the human's. The trained policy runs without the assist.
- "Policy with assist" is a secondary number only (same command): 96, 87, 89 for the human-only policies and
  100, 100, 100 with copies.
- Teleop baseline (human with grasp assist, counted live while recording, not recomputable because failed attempts
  were not saved): 30 of 30 attempts lifted, median 8.0 s (first 25 demos); 56 of 58, median 6.9 s (largest session).
- The task is one cube on a table in simulation, with the cube pose given to the policy. Nothing here was run on a
  real arm. An image policy (camera input in place of the cube pose) is written but not trained or scored yet.
- The first policy design scored 0/100 on the same demos (absolute joint targets, short chunks). The re-design and
  its gates are in [docs/research/PRD-v2.md](docs/research/PRD-v2.md).

## Video

Placeholder: the demonstration video (a screen recording of `make policy-demo`, the policy alone in the live window)
is not added yet.

## How it works

1. **Teleop** (`make demo`): MediaPipe hand landmarks from the webcam move the gripper target; inverse kinematics
   turns it into joint targets at 20 Hz. A clutch key engages the hand; a grasp assist finishes the grasp.
2. **Demos** (`make demo ARGS=--record`): every attempt that lifts the cube without losing the hand for more than
   0.3 s is saved with the full simulator state. Demo cubes use seeds below 10000; the held-out positions are seeds
   10000-10099 and are scored only at named checkpoints; development seeds 20000-20099 are used for everything else.
3. **Labels** (`python -m teleop_il.convert build|gate`): the action is where the gripper actually went one step
   later, relative to where it is, plus open/closed. A gate replays every demo from its labels through the policy's
   own controller and counts lifts.
4. **Transformed copies** (`python -m teleop_il.transform DATA OUT`): each demo's gripper path is moved to a new cube
   position, replayed in the simulator, and kept only if the cube lifts (798 of 1000 tries).
5. **Policy** (`python -m teleop_il.train_v2 --data DIR [--extra COPIES]`): a small MLP on 6 cube-relative numbers
   predicts the next 20 gripper moves; the predictions of recent steps are blended (as in ACT) and a controller
   turns the target into joint targets. Trains on CPU in a few minutes and exports ONNX.
6. **C++** (`make policy-v2-parity RUN=...`): the same ONNX policy and controller in a C++ rollout with ONNX Runtime,
   checked against Python on every held-out position.

## Commands

Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), CMake and a C++ compiler. `make setup` (run by the other
targets) installs the environment and fetches the pinned SO-101 model (mujoco_menagerie), the MediaPipe hand model
and ONNX Runtime.

- `make test`: all tests.
- `make demo`: webcam teleop practice in one window. Light your hand, face the webcam. SPACE engages the clutch, then
  your hand moves the gripper left-right and up-down; W/S move it forward and back; G closes and opens the gripper.
  When the marker under the gripper turns green you are within 5 cm of the cube: press G and the grasp assist lines
  up, descends and closes, then hands back and you lift. R restarts, N skips, Q quits.
- `make demo ARGS=--record`: record demos into a new `runs/demos_<time>/`. `make trial`: 10 counted attempts.
- `make demo ARGS="--log runs/tryN.npz"` then `make diagnose LOG=runs/tryN.npz`: a session log and its summary.
- `python -m teleop_il.convert build OUT SESSION...` and `... gate DATA`: clean recorded sessions; the label gate.
- `python -m teleop_il.transform DATA OUT [--copies N]`: transformed copies.
- `python -m teleop_il.train_v2 --data DIR [--extra COPIES] [--seed S]`: train, export and score on the development
  seeds (`--held-out` only at a checkpoint); the null checks run in the same command.
- `python -m teleop_il.breakdown RUN`: where a policy's failures happen.
- `make policy-demo RUN=runs/v2_policy_...`: watch the policy alone, live.
- `make policy-v2-parity RUN=...`, `make parity`: C++ against Python (policy rollouts; physics).
- `make final-scores HUMAN="..." COPIES="..."`: the table above.
- `python -m teleop_il.evaluation`: the harness null checks.

Python commands run through `env -u PYTHONPATH uv run ...`. Recorded demos, trained policies and videos live in
`runs/` and are not in the repository.

Design, decisions and the evidence for each step: [docs/research/](docs/research/) (start with `HANDOFF.md`).
