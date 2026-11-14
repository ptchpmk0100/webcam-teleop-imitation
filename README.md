# webcam-teleop-imitation

Webcam hand-teleoperation of a simulated SO-101 arm in MuJoCo, used to record demonstrations and train an imitation policy. Success is measured against simulator ground truth (cube height read from MuJoCo state), never against the program's own status output.

Status: early build. Design and decisions: [docs/research/DECISIONS.md](docs/research/DECISIONS.md), [docs/research/PRD.md](docs/research/PRD.md).

## Commands
- `make test` runs the tests (sets up the environment and fetches the pinned SO-101 model and hand model first).
- `make demo` opens the webcam and the simulation window. Light your hand with a lamp. Press SPACE in the simulation window to engage the clutch, then move your hand. Add `--auto-exposure` to use the camera's own exposure (slower frame rate in dim light).
- `python -m teleop_il.evaluation` (through `uv run`) runs the harness null checks: a do-nothing and a random policy must score about 0 percent.
