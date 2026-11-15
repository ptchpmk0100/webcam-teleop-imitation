# 20: The new policy design clones the scripted expert
What to build: state policy v2 (cube-relative observation, gripper-move chunk plus open/close output, chunk execution with blending, gripper hysteresis), its training command and ONNX export (contract v2), trained on 80 converted scripted-expert demos.
Blocked by: 19
Kind: AFK
- [ ] Gate command: at least 90 of 100 on development seeds; exits non-zero below that. Null checks in the same run.
- [ ] ONNX output matches PyTorch within 1e-4 on recorded observations.
- [ ] If the MLP misses the gate after 3 cycles, park and report (a different network needs Khoi's yes).

## Done 2026-11-15 (Opus 5.5)
- `policy_v2` (observation: gripper minus cube grasp point, gripper height, cube bottom height, gripper joint; chunk of
  gripper moves plus a closed logit; L1 + BCE; ACT-style blending, hysteresis 0.6/0.4; `GripperController`) and the
  `train_v2` command. 8 tests (labels checked against the simulator, overfit, ONNX within 1e-4, 4 of 5 lifts on the
  training positions of a 5-demo clone, determinism, development seeds disjoint).
- Screen on 80 expert demos, development seeds, one training seed each: chunk 10 82/100 (re-plan every step without
  blending 86, re-plan every 5 steps 65, m 0.2 85, 500 epochs 69, 512 hidden 72); **chunk 20: 100/100 on training
  seeds 0, 1 and 2**; chunk 30 100/100. Default chunk is now 20.
- Gate, verified: `python -m teleop_il.train_v2 --data runs/expert_data_20261114_235944 --required 0.9` -> 100/100
  development seeds (95% CI 0.96-1.00), null checks PASS, exit 0 (`runs/v2_policy_20261115_133840`). The v1 design
  scored 59/100 on the same demos.
