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

## Teleop redesign grill (2026-11-14, after three logged sessions; supersedes the G1 mapping where they differ)
Evidence: sessions `runs/try1-3.npz` (local, not committed). Tracking is fine (90 percent detection, 1/1000-image jitter, 0-50 ms software lag). The control design is not: depth from hand size follows palm tilt, the pinch gap sits in the 3-5 cm band 35 percent of the time (29 gripper switches in 59 s), pinching moves and hides the positioning hand, and nobody can align the asymmetric SO-101 jaws by eye from one 3D view.
- **Bar (Q1):** after about 10 minutes of practice, at least 5 of 10 attempts lift the cube, at most 30 s each, judged from MuJoCo state.
- **Shared autonomy (Q2, Q13):** the human controls approach and timing. When the gripper is commanded closed within 3 cm (horizontal) of the cube, a grasp assist moves to the expert's grasp pose (jaw offset, grasp height) over about 0.5 s, closes, and hands control back; the human lifts. Episodes record that the assist fired; the baseline is reported as "human with grasp assist".
- **Input (Q3, Q14, Q15):** two throwaway prototypes, one session to build and one for Khoi to test, scratchpad only: (b) whole-arm mirroring with MediaPipe's pretrained pose model (Khoi's idea), and (c) right hand positions, left hand commands (fist closes the gripper). Each: 5 minutes practice, then 10 timed attempts on fixed seeds. Winner = more lifts; if neither reaches 5 of 10, fall back to hand for left-right and up-down, keys for forward-back and the gripper, plus the assist.
- **Two hands (Q11):** allowed (changes G1 "right hand only"), told apart by MediaPipe's handedness label.
- **Camera (Q4, Q12):** prefer the webcam looking down at the desk (laptop lid tilted forward) so left-right and forward-back are flat image motion. Whether this webcam can see the desk is open (Khoi checks). Front-facing stays as a mode.
- **Gripper (Q5):** never on the positioning hand: other-hand fist, or a key that works from any window.
- **Display (Q6):** add a top-down and a side view plus a shadow marker under the gripper.
- **No detector training (Q7):** pretrained MediaPipe models only; the learning in this project is the imitation policy.
- **Demo time limit (Q8):** 30 s for recorded human demos; policies are still evaluated at 15 s.
- **Pipeline (Q9):** tickets 07, 08, 09 and 13 continue on scripted-expert data, always labelled as expert data, while teleop is redesigned.
- **Retargeting repos (Q10):** `dexsuite/dex-retargeting` and `AnyDexRetarget` (both MIT) retarget human finger keypoints to multi-finger robot hands by optimisation; neither trains detection or positions an arm, and the SO-101 has a one-motion gripper. Recorded as references for a possible later dexterous-hand variant (out of scope now).
- **Week-1 video (Q16):** recorded after the redesign wins, not with the current mapping.
- **Outcome (G5, 2026-11-14):** both prototypes lifted 0; the fallback (hand for left-right and up-down, W/S for forward-back, G for the gripper, grasp assist) passed the bar on its first counted trial, 6 of 10 (verified from logged cube height), and Khoi accepted it. The assist zone is 5 cm, not 3 (the prototypes' nearest close was 3.2 cm), and it fires whenever the gripper is closed inside the zone, once per close.
- **After 25 demos (G5, 2026-11-15):** ticket 10 accepted; ticket 11 (to 100 demos) agreed. Khoi's challenge: a robot that follows the hand and lets a scripted grasp finish is not a real-world design; if that were the goal, re-plan. Answer: it is not the goal. The product is an autonomous policy (state, later image) that does the whole pick-and-lift alone; teleop is only the data-collection tool and the assist a crutch for it. Proposed: try a no-assist trial now that W/S and the window work; if Khoi lifts at least 5 of 10 without it, record the remaining demos without the assist (source "human"). Scoring: Khoi chose (b) "policy with the assist"; flagged to him that this puts the scripted grasp inside the deployed controller, the very thing he objects to, so (a) "policy alone" is recommended as the headline with (b) as a secondary number. Waiting for his confirmation.
- **Confirmed (Khoi, 2026-11-15):** the headline score is (a) the policy alone, with no assist and no human; (b) policy with the assist is reported only as a labelled secondary number. The assist stays for collecting demos (the end goal is without it anyway); the no-assist trial is skipped.

## Policy re-plan (2026-11-15, after 0/100 on 100 clean demos; supersedes the policy decisions above where they differ)
Evidence: ticket 12. Khoi does not accept 0/100. Grill rounds 1 and 2, all answered "yes". Details: PRD-v2.md.
- **Target (Q1):** policy alone at least 50 of 100 held-out seeds, stretch 80, with confidence interval.
- **Action and observation (Q2):** gripper moves plus an open/close output, observation relative to the cube's grasp point, and a small controller identical in Python and C++. Replaces "absolute joint targets".
- **Labels:** the motion the arm achieved, not the recorded teleop target. Gate: replay at least 95 of 100 before training.
- **Execution:** whole chunks with blending (ACT-style). "ACT is a stretch goal" now applies only to the transformer network; the MLP stays the default.
- **Design gate:** clone the scripted expert at 90 of 100 or better before training on human demos.
- **Transformed demos (Q3):** Khoi's 100 demos moved to new cube positions, simulator-verified, plus recovery data; disclosed as "100 human demos plus N transformed copies".
- **Corrections (Q4):** up to three rounds of about 10 minutes, only if the transformed-demo stage misses 50 of 100.
- **Disclosure (Q5):** the README states that the grasp segment of each demo is the assist's motion. No recording without the assist.
- **Tickets (Q6):** 12 closed as superseded; new tickets 18 to 25; 16 and 17 re-blocked on 25.
- **Seeds (Q7):** development seeds 20000-20099 for experiments; held-out seeds only at named checkpoints. About 20 variants had already been scored on the held-out seeds before this rule.
- **Headline (Q8):** both scores reported; the augmented one is the headline, the human-only one beside it.
- **Stop rule (Q9):** under 50 of 100 after three correction rounds: stop and ask. Diffusion stays out of scope and needs a fresh ask.
- **Model use:** planning was done on Fable 5.2; execution (ticket 18 onward) on Opus 5.5.

## Execution answers (Khoi, 2026-11-15, "yes for all")
- **Label gate (ticket 19):** judged on the demos that did not push the cube before the first close (at least 95
  percent must replay; at least one such demo must exist). Demos that pushed are counted and listed, not judged. Reason:
  a push is a chaotic contact that no replay repeats (10-16 of 23 replay, whatever the format). Now 77/77, PASS.
- **Training set (ticket 21):** all 100 cleaned demos, not only the 93 that replay.
- **Held-out checkpoint (ticket 21):** run now, before ticket 18's breakdown; training seeds 0, 1 and 2 are all scored
  and all reported (none chosen). Then ticket 24, and ticket 18 reduced to the failure breakdown (v1 is retired).
