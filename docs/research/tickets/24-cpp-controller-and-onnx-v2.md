# 24: C++ controller and ONNX contract v2 (replaces 14)
What to build: the C++ rollout runs policy v2: builds the cube-relative observation, runs the network, blends chunks, applies the gripper rule and the same controller as Python.
Blocked by: 20
Kind: AFK
- [ ] Parity command: C++ and Python outcomes agree on 100 of 100 held-out seeds; exits non-zero otherwise.
- [ ] C++ controller output matches Python on recorded states within a stated tolerance.
- [ ] C++ null checks 0 of 100.
