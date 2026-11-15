# 19: Gripper-move controller and the label gate
What to build: a controller that turns a gripper move into joint targets (damped least squares on the gripper site, soft point-down), and a conversion command that rebuilds the training set from the recorded sessions (the cleaning now done by a scratchpad script becomes part of it) with actions taken from the motion the arm achieved, plus the open/close command. Both candidate formats (move against the current gripper position; target relative to the cube) are replayed open loop; the one that lifts more demos is kept.
Blocked by: None (can start immediately)
Kind: AFK
- [x] Controller tests through forward kinematics: commanded move reached within tolerance, joint limits respected.
- [x] Gate command: replay of the converted 100 human demos lifts at least 95; exits non-zero below that. (Changed by
  Khoi 2026-11-15: at least 95 percent of the demos without a pre-grasp push; 77/77.)
- [x] The scripted expert's demos convert and replay at least 95 of 100 too.
- [x] Output is a new dataset folder with a manifest naming source, method and counts; inputs unchanged.

## Results 2026-11-15 (Opus 5.5)
- Controller `GripperController`: damped least squares on the gripper site Jacobian from the measured joints, joints
  at a limit frozen and the step re-solved, soft orientation weight 0.005. 17 tests through forward kinematics pass.
- Label format: achieved gripper position one control step ahead plus open/closed from the recorded target (no
  stored labels, nothing relabelled). The two candidate formats (move against the current gripper position, target
  relative to the cube) give the same absolute target in replay, so the gate cannot choose between them; the policy
  ticket (20) chooses.
- Screen (scratch probes, replay of `runs/demos_100`): recorded joint targets open loop 85/100; task-space tracking
  with lookahead 1-3 steps 85-91; tracking gain above 1 unstable (20/100 at gain 2: the servos are force-limited, any
  error above about 3 mrad saturates); exact one-step inverse dynamics 84/100. Orientation weight 0.02 / 0.1 / 0.005 /
  0.002 / 0.001: 88 / 69 / 93 / 86 / 87.
- **Why demos fail to replay:** 23 of 100 demos pushed the cube more than 5 mm before the first close (the assist or
  the hand knocked it). Pushes are chaotic contacts: 10-16 of these 23 replay, whatever the format. The 77 demos without
  a push replay 77/77.
- Repo commands, verified:
  - `python -m teleop_il.convert build runs/v2_human_20261115 <7 sessions>`: 112 in, 12 dropped (auto-grasp), 34 stale
    rows dropped, 100 kept (exit 0).
  - `python -m teleop_il.convert gate runs/v2_human_20261115 --keep runs/v2_human_replayed_20261115`: **93/100, FAIL
    against the frozen 95** (exit 1); no-push 77/77, pushed 16/23; the 93 that replay are in the kept folder.
  - `python -m teleop_il.convert gate runs/expert_data_20261114_235944`: 100/100, PASS (exit 0).
- Open, Khoi (short G3): the 95/100 gate is not reachable with demos that pushed the cube. Proposed: the gate counts
  demos without a push (required 95 percent; now 77/77), and training uses the 93 demos that replay.

## Closed 2026-11-15
Khoi accepted the gate change ("yes for all"); training uses all 100 cleaned demos (ticket 21 decision).
- Gate now judges the demos without a push before the first close, and fails if there are none (two new tests, red
  first, then green; the "fails loudly" test now uses a failing demo with no push).
- Verified: `python -m teleop_il.convert gate runs/v2_human_20261115`: 77/77 without a push, PASS (exit 0); all demos
  93/100; pushed 16/23, not judged.
