# 03: Teleop core on synthetic landmark clips
What to build: hand landmarks in, joint targets and gripper command and a tracking-loss flag out. Clutch, scaling, One-Euro filter, pinch with hysteresis, soft downward orientation (weight near 0.02), IK with mink, hold on tracking loss.
Blocked by: 01
Kind: AFK
- [ ] Test command exits 0: synthetic wrist motion by d moves the end-effector by the scaled amount (read through forward kinematics, within 5 mm inside the rectangle).
- [ ] Tracking loss over 0.3 s holds the arm and sets the flag; clutch release freezes the arm; pinch does not flicker at the threshold.
- [ ] Every point of the 0.10-0.30 by +-0.15 rectangle is reached within the tilt bound.
