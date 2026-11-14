# 13: C++ rollout, physics parity
What to build: a C++ executable that links MuJoCo's C library at the pinned release, loads the same scene, runs a scripted action sequence over seeds and writes results.
Blocked by: 01, 02
Kind: AFK
- [ ] Parity script exits 0: cube height within 5 mm of the Python run after the sequence, and same success outcome on at least 98 of 100 seeds.

Done 2026-11-14. Acceptance `make parity` (python -m teleop_il.cpp_parity) exits 0: 130 sequences (scripted expert on the 100 held-out seeds, plus 10 each of do-nothing, wild random targets up to 1.5 times the joint range, and two short lifts that must not count): largest cube height gap 0.0000 mm, largest trajectory gap 0 (bit-identical), success agrees 130 of 130, expert lifts 100 in both. The C++ side links the libmujoco.so.4.0.1 inside the pinned Python wheel and loads the compiled scene as a MuJoCo binary model (no XML round trip). Each run writes a new folder under runs/.
Stricter than the frozen threshold, added here (frozen 5 mm and 98 of 100 kept as the acceptance): the end state alone missed a real physics change (9 instead of 10 physics steps per control step moved the final cube height by only 0.7 mm, all outcomes equal), so the joint-state trajectory is compared at every control step with tolerance 1e-6, and lift height is compared too. C++ mutations: 4 of 5 caught after adding the wild and two-short-lift sequences (before: 1 of 5); the survivor (no clipping) is equivalent, because the scene's actuators are ctrllimited and MuJoCo clamps the control itself (checked).
Status: DONE.
