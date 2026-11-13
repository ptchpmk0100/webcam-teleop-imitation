# Decisions so far (2026-11-13/14)

Settled with Khoi in a grill session before this repo existed. A new session should treat these as decided and only grill what is listed under "Still open".

## Project
- **Chosen idea: A.** The webcam tracks Khoi's hand (MediaPipe, pretrained) and drives a simulated arm in MuJoCo (not a Franka Panda). Recorded demonstrations form a dataset. A behaviour-cloning policy (PyTorch, trained on Colab or Kaggle) learns the task and is exported to ONNX.
- **C++ module is required**, not optional, in weeks 3–4: an ONNX Runtime inference node in C++. C++ appears in 25 of 50 JDs and is listed on the CV without any project behind it.
- **No ROS 2** (Q15). This keeps the project clearly separate from the Panda digital twin (ROS 2 + Gazebo). A thin ROS 2 wrapper (about 2 days) was offered as an option; it would raise the weighted ATS score by 19 points. Not chosen yet.
- Parked ideas B–E are in [draft2_report.md](draft2_report.md). E (EKF sensor fusion for a mobile robot) is the recommended second project.

## Constraints
- **Target:** internship, working-student (Werkstudent) and thesis roles in Germany. German and English JDs count as one market. The bar is to impress beyond what recruiters expect from an intern; no research breakthrough needed (Q7, Q12).
- **Hardware:** laptop webcam (/dev/video0, readable by the user) and a GTX 1650 with 4 GB. Nothing to buy. Print a ChArUco board.
- **Compute:** free Colab or Kaggle first; paid GPU time capped at €20 in total (Q16).
- **Time:** something showable after about 1 week, then continue (Q5).
  - Week-1 milestone: the live webcam drives the simulated arm using pretrained models only. Recorded as a video of 60 s or less (Q13).
  - Training and benchmark numbers come in week 2 and later.
- **Showability (Q7):**
  - one measurable number
  - a demo video of 60 s or less
  - one command each to run the tests and the demo
  - at least one deployment keyword (ONNX)

## Rules for checking results (Q17; lesson from the Panda false pass of 2026-11-06)
- Success is never the program's own "success" output.
- **Week 1:** MuJoCo ground truth (object and end-effector poses read from the simulator) plus the printed ChArUco board as the real-world reference.
- **Later:** recorded-clip regression tests in CI. A ticket may only run unattended (AFK) if such an independent check exists and fails loudly.
- **Task success:** the cube is more than 5 cm above the table for at least 1 s, read from MuJoCo state.

## How to work
- Git: on 2026-11-14 Khoi gave standing approval to create, commit and push **this repo only**. Never touch other GitHub repos. Never delete without asking.
- Shell: no `rm` and no overwrite-in-place, even in the scratchpad. Each stage writes to a new folder; inputs stay read-only.
- Planning files go only where Khoi names them. This `docs/research/` folder was named on 2026-11-14.

## Still open (grill these in the new session)
1. Which MuJoCo arm: a menagerie model such as UR5e, xArm or SO-100, or something else?
2. Task: pick-and-place a cube, or something richer?
3. Hand → arm mapping: wrist position to end-effector position, pinch to gripper? Which IK solver?
4. Dataset format: LeRobot format, or plain HDF5/npz?
5. Policy type: plain behaviour cloning (MLP or CNN), or ACT / diffusion-lite?
6. Feasibility spike first: webcam → MediaPipe → MuJoCo frame rate and latency on this laptop.
7. Repo name, licence, and when to make the repo public (currently private until the week-1 video exists).

## G1 decisions (2026-11-14, autopilot grill; supersedes "Still open" above)
All seven open items are settled. Facts from a sub-agent (menagerie, mink, MediaPipe, LeRobot) are claims with sources until the spike checks them.

- **Task:** pick-and-lift a cube from a randomised start. Place-at-target and moving around are parked, out of scope for now.
- **Arm and IK:** SO-101 (`robotstudio_so101` in mujoco_menagerie; 5 arm joints plus gripper) with `mink`. IK target is end-effector position plus a soft "point down" preference. Fallbacks: UR5e plus Robotiq 2F-85; damped-least-squares Jacobian solver on `mujoco`.
- **Mapping:** clutch (relative) mapping, One-Euro filter on MediaPipe world landmarks, pinch to gripper with hysteresis, fixed downward orientation. ChArUco is a scale sanity check only.
- **Spike first (throwaway):** webcam to MediaPipe to MuJoCo at 25 FPS or better and 100 ms latency or less; `mink` at 50 Hz or better; SO-101 reaches the whole cube spawn area; offscreen render time at 96x96 or 128x128.
- **Rate:** 20 Hz recording and policy, 15 s (300 step) timeout.
- **Tracking loss:** over 0.3 s means the arm holds and the episode is discarded. Right hand only, one hand in view.
- **Observation:** state first (joints, gripper, cube pose), then image plus joints. The headline number names which one it uses. Full MuJoCo state is recorded so images can be replay-rendered.
- **Action:** absolute joint targets, action chunk of about 10 steps.
- **Dataset:** one `.npz` per episode plus a versioned `manifest.json`. Each stage writes to a new folder. LeRobot export is a stretch goal.
- **Policy:** MLP on state (week 2), then small CNN plus MLP on image and joints (weeks 3-4). ACT is a stretch goal; diffusion is not planned. Watch for mode averaging.
- **Demos:** about 100 successful demos, checkpoints at 25 and 50. Split by episode, 80/20. Spawn rectangle proven reachable by a scripted expert.
- **Validation:** success from MuJoCo state only; 100 fixed held-out seeds; binomial confidence interval; null checks (do-nothing and random policies about 0%); scripted-expert ceiling; open-loop replay must fail; ONNX parity between C++ and Python rollouts.
- **Headline number:** policy success rate over 100 held-out rollouts versus the teleop baseline.
- **C++:** an executable linking ONNX Runtime and MuJoCo's C library owns the rollout loop, same scene XML and seeds as Python, MuJoCo version pinned. Fallback: socket policy server.
- **Training:** Khoi runs Colab or Kaggle by hand and brings the checkpoint back. Paid GPU time is EUR 0. No tokens or keys are given to the agent.
- **Repo:** name `webcam-teleop-imitation`, MIT licence, private until the week-1 video exists. Making it public needs a fresh ask.
- **Out of scope:** place-at-target, other arms, Panda, ROS 2 (thin wrapper stays an option), multiple operators or hands, full camera calibration, paid compute, diffusion.
- **Deferred to PRD and board:** test seams, PRD and ticket location, one command each for tests and demo.
