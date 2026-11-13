# 14: C++ ONNX policy parity
What to build: the C++ executable loads an ONNX policy with ONNX Runtime and runs the rollout loop.
Blocked by: 09, 13
Kind: AFK
- [ ] Parity script exits 0: ONNX output matches PyTorch within 1e-4 on recorded observations, and C++ rollouts agree with Python rollouts on at least 98 of 100 seeds.
- [ ] The C++ run also passes the null checks.
