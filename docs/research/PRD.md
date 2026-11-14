# PRD: webcam hand-teleop to imitation policy in MuJoCo

Status: draft 1, 2026-11-14. Synthesised from the G1 grill (see DECISIONS.md) and the spike (see SPIKE_RESULTS.md). Closed or deleted when shipped, not kept as documentation.

## Problem statement

Khoi wants an internship-grade robotics portfolio project that shows the full loop from human demonstration to deployed policy, with numbers that cannot be faked. Today there is no project: no way to demonstrate a simulated arm controlled by his hand, no recorded demonstrations, no trained policy, and no deployment artifact. A previous project produced a false pass (a program reported its own success while the task had failed), so any result in this project must be judged by an independent check, never by the program's own status output.

## Solution

A webcam tracks Khoi's right hand. The hand drives a simulated SO-101 arm in MuJoCo through a clutch mapping and inverse kinematics. Khoi records about 100 successful pick-and-lift demonstrations. A behaviour-cloning policy is trained on them (state first, then image plus joints), exported to ONNX, and run from a C++ executable that owns the rollout loop. Success is the cube more than 5 cm above the table for at least 1 s, read from MuJoCo state, measured over 100 fixed held-out cube positions, and compared with the teleop baseline.

Showable result at the end of week 1: the live webcam drives the arm using pretrained models only, in a video of 60 s or less. Training and benchmark numbers follow from week 2. One command runs the tests and one command runs the demo.

## User stories

1. As Khoi, I want my right hand in front of the webcam to move the simulated arm, so that I can demonstrate a task by hand.
2. As Khoi, I want a clutch so that the arm only follows my hand while I hold an engage gesture or key, so that I can reposition my hand without moving the arm.
3. As Khoi, I want the hand position change since engagement scaled onto the end-effector, so that small hand motions reach the whole workspace.
4. As Khoi, I want a One-Euro filter on the hand landmarks, so that webcam depth noise does not shake the arm.
5. As Khoi, I want a thumb-index pinch to close and open the gripper with hysteresis, so that the gripper does not flicker at the threshold.
6. As Khoi, I want the gripper to point roughly down while the hand controls position only, so that I do not need to control orientation.
7. As Khoi, I want the arm to hold its last target when hand tracking is lost for more than 0.3 s, so that a lost hand never throws the arm.
8. As Khoi, I want an episode with tracking loss flagged and left out of the dataset, so that the policy never learns from frozen-arm segments.
9. As Khoi, I want a lamp-lit manual-exposure camera mode that records at 25 FPS or better, so that the loop rate is not limited by exposure.
10. As Khoi, I want to see the simulation window while I teleoperate, so that I have visual feedback.
11. As Khoi, I want each recorded episode to start from a seeded random cube position inside a reachable rectangle, so that demos cover the workspace and can be replayed exactly.
12. As Khoi, I want an episode to end automatically on success or after 15 s, so that recording is hands-free between attempts.
13. As Khoi, I want only successful episodes kept in the dataset, with my success rate and time per attempt logged, so that the teleop baseline is measured honestly.
14. As Khoi, I want one episode per file with the full MuJoCo state, joint targets, gripper, cube pose, seed and success flag, plus a versioned manifest, so that any observation (state or image) can be rebuilt by replay.
15. As Khoi, I want images rendered by replay into a new folder, so that no stage overwrites another.
16. As Khoi, I want train and validation split by episode, so that frames of one episode never leak across the split.
17. As Khoi, I want a scripted expert policy that picks and lifts the cube, so that I know the achievable ceiling and the spawn rectangle is proven reachable.
18. As Khoi, I want an evaluation harness that runs any policy over 100 fixed held-out seeds and reports a success rate with a binomial confidence interval, so that the headline number is trustworthy.
19. As Khoi, I want the do-nothing and random-action policies to score about 0 percent, so that a broken harness fails loudly.
20. As Khoi, I want an open-loop replay of one recorded demo on a new cube position to fail, so that I know the policy has to react to the cube.
21. As Khoi, I want a state-based MLP policy with an action chunk of about 10 steps, so that week 2 has a first real number.
22. As Khoi, I want an image plus joints CNN policy with the same chunking, so that weeks 3 and 4 can claim vision-based imitation if it works.
23. As Khoi, I want the headline number to name whether it is the state or the image policy, so that the two are never mixed.
24. As Khoi, I want the mode-averaging failure of plain regression to be looked for in the evaluation, so that it is reported if it happens.
25. As Khoi, I want the trained policy exported to ONNX, so that deployment is demonstrated.
26. As Khoi, I want a C++ executable that links ONNX Runtime and MuJoCo's C library and owns the rollout loop with the same scene and seeds, so that the C++ work is real deployment and not a wrapper.
27. As Khoi, I want C++ outputs to match PyTorch on recorded observations and C++ rollouts to match Python rollouts on the same 100 seeds, so that the deployment is proven equivalent.
28. As Khoi, I want one command to run the tests and one command to run the demo, so that a reviewer can check the project in minutes.
29. As Khoi, I want a demo video of 60 s or less with the headline number in the README, so that the project is showable.
30. As Khoi, I want the ChArUco board used once as a scale check of hand displacement against arm motion, so that there is a real-world reference without a full calibration pipeline.
31. As Khoi, I want to run Colab or Kaggle training by hand from a script and a data bundle I am given, so that no tokens or paid compute are involved.
32. As a reviewer, I want every number in the README traceable to a command, so that I can reproduce it.

## Implementation decisions

Four deep modules. Khoi owns each interface; the inside is filled in by agents.

1. **Env and Eval.** A simulated tabletop scene with the SO-101, a cube and a floor acting as the table (a table is not required by the spike; add one only if the grasp needs it). Interface: reset with a seed returns an observation; step with joint targets (5 arm joints plus gripper) advances physics at a fixed control rate of 20 Hz and returns an observation; success reads cube height from MuJoCo state; render returns a small camera image; snapshot and restore of full state. Evaluate takes any callable policy and a list of seeds and returns a success rate with a binomial confidence interval. Physics timestep and the decimation to 20 Hz are internal. Episode timeout is 15 s.
2. **Teleop.** Interface: a stream of hand landmarks (world and image) with timestamps in; a stream of joint targets plus gripper command plus a tracking-loss flag out. Internals: clutch, scaling, One-Euro filter, pinch with hysteresis, fixed downward orientation preference, inverse kinematics with `mink` against the SO-101 (soft orientation weight near 0.02, from the spike), hold on tracking loss. A thin live adapter feeds webcam frames through MediaPipe's pretrained hand model to this interface; the live adapter is not a test seam.
3. **Dataset.** Interface: write an episode (state, joint targets, gripper, cube pose, seed, success, tracking flags), read episodes, split by episode, validate against a versioned manifest. One file per episode. Stages write to new folders.
4. **Policy and deployment.** Training takes a dataset and produces a PyTorch checkpoint and an ONNX file; it runs on Colab or Kaggle by Khoi's hand. A C++ executable loads the ONNX file and the same scene description, runs the rollout loop over a seed list and writes a result file. The same MuJoCo release is pinned for Python and C++.

Technical decisions carried from the grill and spike:
- Arm: SO-101 from mujoco_menagerie, 5 arm joints plus gripper, position actuators. Verified by loading the model in the spike.
- Reachable spawn area (UPDATED by ticket 06: forward 0.17 to 0.30 m is the proven rectangle; the text below is the spike's original claim): the spike found a rectangle (forward 0.10 to 0.30 m, sideways -0.15 to +0.15 m, from the arm base) reachable with the gripper within 10 degrees of vertical at cube-centre height, ignoring collisions and grasp physics. The scripted expert must prove the final spawn rectangle with a real grasp before demos are recorded. The default cube position in menagerie's box scene is out of reach and must not be used.
- Camera: lamp-lit hand, manual exposure, 640 by 480 MJPG, 30 FPS requested. Auto exposure delivers 17 FPS in this room and is not acceptable for recording. Scripts set exposure mode explicitly because V4L2 settings persist on the device.
- Rates: policy and recording at 20 Hz; teleop loop at camera rate, decimated to 20 Hz for recording.
- Action space: absolute joint targets with a chunk of about 10 future steps.
- Observation: state (joints, gripper, cube pose) first; image plus joints later. Full MuJoCo state is always recorded.
- Dependencies installed in a project venv managed by uv; Python 3.12.
- Tests: pytest. One command for tests, one for the demo.

## Testing decisions

A good test checks external behaviour at a seam, never internals. Seams agreed at G2 (Khoi, by standing instruction to follow the recommendation):

- **Seam A, Env plus Eval (highest seam).** Everything is judged here. Tests: null policy and random policy score about 0 percent on the held-out seeds; scripted expert scores high; open-loop replay of a recorded demo on a new seed fails; the success check agrees with a direct read of cube height; evaluation is deterministic for a fixed seed list.
- **Seam B, Teleop.** Input is recorded landmark clips stored as JSON (no video in the repo). Tests: golden joint-target trajectories with tolerances; tracking loss longer than 0.3 s holds the arm and sets the flag; clutch engage and release; pinch hysteresis; IK reaches the whole proven spawn rectangle within the tilt bound.
- **Seam C, Dataset.** Tests: write and read round trip; manifest schema validation fails loudly on a malformed file; train and validation share no episode.
- **Seam D, ONNX and C++ parity.** Tests: C++ output equals PyTorch output on recorded observations within a tolerance; C++ and Python rollouts on the same 100 seeds agree in success rate within a stated tolerance and the C++ run also passes the null checks.
- Training code is tested locally only on a tiny dataset: loss decreases and export works. Real training runs are manual.
- A ticket may run unattended only if an independent check from the seams above exists and fails loudly. The null-policy check and the parity check are the main ones.
- No prior art in the repo; this is the first code.

## Out of scope

- Place-at-target, moving around, other tasks (parked for later).
- Other arms (UR5e fallback only if SO-101 fails), the Panda, ROS 2 (a thin wrapper stays a later option), multiple operators or hands, full camera calibration, paid GPU time, diffusion policies, LeRobot export (stretch only), ACT (stretch only).
- Real hardware.

## Further notes

- Week 1 milestone (pretrained models only): live webcam drives the arm; video of 60 s or less. The repo stays private until that video exists; making it public is a separate ask.
- Unverified until built: MediaPipe world-landmark depth quality, whether the SO-101 gripper can lift the cube, camera-to-arm physical latency, and whether the image policy works with about 100 demos.
- Training runs are manual: Khoi runs Colab or Kaggle and brings the checkpoint back. No tokens or keys are shared.
