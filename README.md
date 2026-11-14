# webcam-teleop-imitation

Webcam hand-teleoperation of a simulated SO-101 arm in MuJoCo, used to record demonstrations and train an imitation policy. Success is measured against simulator ground truth (cube height read from MuJoCo state), never against the program's own status output.

Status: early build. Design and decisions: [docs/research/DECISIONS.md](docs/research/DECISIONS.md), [docs/research/PRD.md](docs/research/PRD.md).

## Commands
- `make test` runs the tests (sets up the environment and fetches the pinned SO-101 model and hand model first).
- `make demo` opens the webcam and one window (view from behind the arm, top and side views, the camera, a status panel) for practice. Light your hand with a lamp, face the webcam. Keys in the window: SPACE engages the clutch, then your hand moves the gripper left-right and up-down; W/S move it forward and back; G closes and opens the gripper. When the marker under the gripper turns green you are within 5 cm of the cube: press G and the grasp assist lines up, descends and closes (marker yellow), then hands back and you lift. R restarts the attempt, N skips it, Q quits. Each attempt has 30 s; a lift (cube bottom above 5 cm for 1 s, read from MuJoCo) ends it.
- `make trial` runs the counted trial: 10 attempts on fixed cubes, results saved under `runs/`. Bar: 5 of 10 lifted.
- Record a session with `make demo ARGS="--log runs/try4.npz"` and read it with `make diagnose LOG=runs/try4.npz`. `--no-assist` switches the assist off, `--auto-exposure` uses the camera's own exposure (slower frame rate in dim light).
- `make parity` builds the C++ rollout (CMake, against the MuJoCo library inside the pinned Python wheel) and checks it against the Python physics on 130 sequences; exit 0 only if they agree (they are bit-identical).
- `make policy-parity` runs the newest trained state policy in C++ with ONNX Runtime (pinned release, fetched by `make setup`) and checks it against PyTorch (within 1e-4) and against the Python rollouts on the 100 held-out seeds (at least 98 agree), plus the C++ null checks; `RUN=runs/state_policy_...` picks a run.
- `uv run python -m teleop_il.train_state --expert-episodes 100` records scripted-expert episodes into a new folder, trains the state policy (MLP, 10-step action chunks) on CPU, exports PyTorch and ONNX files and scores the ONNX policy on the 100 held-out seeds (`--data FOLDER` trains on an existing dataset).
- `python -m teleop_il.evaluation` (through `uv run`) runs the harness null checks: a do-nothing and a random policy must score about 0 percent.
