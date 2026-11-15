# 20: The new policy design clones the scripted expert
What to build: state policy v2 (cube-relative observation, gripper-move chunk plus open/close output, chunk execution with blending, gripper hysteresis), its training command and ONNX export (contract v2), trained on 80 converted scripted-expert demos.
Blocked by: 19
Kind: AFK
- [ ] Gate command: at least 90 of 100 on development seeds; exits non-zero below that. Null checks in the same run.
- [ ] ONNX output matches PyTorch within 1e-4 on recorded observations.
- [ ] If the MLP misses the gate after 3 cycles, park and report (a different network needs Khoi's yes).
