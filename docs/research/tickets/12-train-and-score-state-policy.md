# 12: Train and score the state policy on human demos
What to build: train the state MLP on the human dataset (local GPU, or a Colab script and bundle if needed), export ONNX, score over the 100 held-out seeds, and look for mode averaging.
Blocked by: 02, 09, 11
Kind: HITL
- [ ] Success rate with confidence interval printed, next to the teleop baseline and the expert ceiling.
- [ ] Null checks pass in the same run.

Lever screen on the 38 clean demos of `runs/demos_50`, 2026-11-15 (scratchpad `levers.py`, in memory, 50 held-out
seeds each; claim level: experiment, single training seed):
- Recorded labels: 13 variants (base, relative cube-minus-gripper input, 5x weight within 6 cm of the grasp point,
  20-step chunk, delta targets, previous gripper command as input, input noise, combinations, 512 hidden and 400
  epochs) all 0/50; closest approach median 6.5-9.5 cm. The policy moves (median 11 cm of travel) but not to the
  cube, and fails on its own training cubes too (0/15) while its teacher-forced error is 0.006 rad: compounding
  error on labels that do not match the motion (ticket 11, second bug).
- Relabelled by per-step inverse dynamics (scratchpad `relabel.py` into `runs/demos_50_relabelled`; one-step
  residual median 0.0002 rad, 99th percentile 0.086): base 2/50, rel+delta+w5 3/50 (closest median 6.1 cm, 9 of 50
  within 2 cm), prev 0/50. First lifts from human data.
- Reading: the label fix helps, but 38 demos of human approaches are far from enough; the expert reached 59/100
  with 80 exact demos. Next data point: 100 clean demos recorded with both fixes.
