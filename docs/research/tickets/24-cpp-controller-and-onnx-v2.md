# 24: C++ controller and ONNX contract v2 (replaces 14)
What to build: the C++ rollout runs policy v2: builds the cube-relative observation, runs the network, blends chunks, applies the gripper rule and the same controller as Python.
Blocked by: 20
Kind: AFK
- [ ] Parity command: C++ and Python outcomes agree on 100 of 100 held-out seeds; exits non-zero otherwise.
- [ ] C++ controller output matches Python on recorded states within a stated tolerance.
- [ ] C++ null checks 0 of 100.

## Done 2026-11-15
New executable `policy_v2_rollout` (the v1 `policy_rollout` is unchanged): the cube-relative observation, ONNX
contract v2, chunk blending (numpy's summation order for the weights), hysteresis and the gripper-move controller in
C++, with the constants passed from Python. Check: `make policy-v2-parity RUN=<runs/v2_policy_* folder>`
(`python -m teleop_il.cpp_policy_v2_parity`). 4 tests (`tests/test_cpp_policy_v2_parity.py`); mutation checks: a wrong
orientation weight on the C++ side fails the controller test, a wrong blending exponent fails the rollout test.
Verified on `runs/v2_policy_20261115_175750` (exit 0): ONNX Runtime vs PyTorch 9.5e-6 on 2714 observations; controller
6.6e-15 on 2704 recorded states (tolerance 1e-9); held-out rollouts Python 96, C++ 96, outcome agrees 100/100, cube
height gap 0.0000 mm; C++ null checks do-nothing 0/100, random 0/100. Run folder `runs/cpp_policy_v2_parity_20261115_180435`.
- [x] Parity command: C++ and Python outcomes agree on 100 of 100 held-out seeds; exits non-zero otherwise.
- [x] C++ controller output matches Python on recorded states within a stated tolerance.
- [x] C++ null checks 0 of 100.
