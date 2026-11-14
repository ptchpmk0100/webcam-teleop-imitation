# 14: C++ ONNX policy parity
What to build: the C++ executable loads an ONNX policy with ONNX Runtime and runs the rollout loop.
Blocked by: 09, 13
Kind: AFK
- [ ] Parity script exits 0: ONNX output matches PyTorch within 1e-4 on recorded observations, and C++ rollouts agree with Python rollouts on at least 98 of 100 seeds.
- [ ] The C++ run also passes the null checks.

Done 2026-11-15. Acceptance `make policy-parity` (python -m teleop_il.cpp_policy_parity) exits 0 on the ticket 09 policy (runs/state_policy_20261114_235944): C++ ONNX Runtime vs PyTorch on 1376 recorded observations, max gap 6.0e-7 (limit 1e-4); held-out rollouts Python 59/100, C++ 59/100, outcome agrees on 100/100, largest cube height gap 0.0000 mm; C++ null checks do-nothing 0/100, random 0/100. Tests `pytest tests/test_cpp_policy_parity.py` (6 passed; the fixture policy is trained on 60 expert episodes because with 20 it lifted none of the test seeds and the comparison must include lifts).
Build: ONNX Runtime 1.32.0 C/C++ release (the Python package's version), fetched and SHA-256-pinned by scripts/fetch_assets.sh into assets/ (not committed); policy_rollout links it and the wheel's libmujoco; the simulation core moved into cpp/sim.h so both executables share reset, step and success. The C++ loop mirrors the evaluation harness (observe as float32, run, execute the chunk's first action, stop on success or after 300 steps). Mutations: 4 of 4 caught (last action of the chunk, swapped cube coordinates, no stop on success, 4-digit output).
Status: DONE.
