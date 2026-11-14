# 03: Teleop core on synthetic landmark clips
What to build: hand landmarks in, joint targets and gripper command and a tracking-loss flag out. Clutch, scaling, One-Euro filter, pinch with hysteresis, soft downward orientation (weight near 0.02), IK with mink, hold on tracking loss.
Blocked by: 01
Kind: AFK
- [ ] Test command exits 0: synthetic wrist motion by d moves the end-effector by the scaled amount (read through forward kinematics, within 5 mm inside the rectangle).
- [ ] Tracking loss over 0.3 s holds the arm and sets the flag; clutch release freezes the arm; pinch does not flicker at the threshold.
- [ ] Every point of the 0.10-0.30 by +-0.15 rectangle is reached within the tilt bound.

Status: CLOSED 2026-11-14. Verified: `make test` exit 0 (68 passed); 9 mutations of teleop and IK code each turn the suite red.
Findings to carry forward: (1) with the soft orientation weight 0.02 the far corners (x 0.30, y +-0.15) tilt 10.7 degrees, so the test bound is 12 degrees. (2) At x 0.20 the SO-101 can only hold the gripper vertical up to about 9-10 cm height (wrist joint limit); above that it tilts (13 deg at 12 cm, 42 deg at 20 cm). Home hover pose is therefore the vertical pose at z 0.09. (3) Gripper joint: -0.175 closed, 1.2 about 10 cm open (measured). (4) Clutch is a boolean in the input frame; the live adapter (04) decides which key.
