# 19: Gripper-move controller and the label gate
What to build: a controller that turns a gripper move into joint targets (damped least squares on the gripper site, soft point-down), and a conversion command that rebuilds the training set from the recorded sessions (the cleaning now done by a scratchpad script becomes part of it) with actions taken from the motion the arm achieved, plus the open/close command. Both candidate formats (move against the current gripper position; target relative to the cube) are replayed open loop; the one that lifts more demos is kept.
Blocked by: None (can start immediately)
Kind: AFK
- [ ] Controller tests through forward kinematics: commanded move reached within tolerance, joint limits respected.
- [ ] Gate command: replay of the converted 100 human demos lifts at least 95; exits non-zero below that.
- [ ] The scripted expert's demos convert and replay at least 95 of 100 too.
- [ ] Output is a new dataset folder with a manifest naming source, method and counts; inputs unchanged.
