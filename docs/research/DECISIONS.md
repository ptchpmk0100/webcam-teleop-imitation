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
