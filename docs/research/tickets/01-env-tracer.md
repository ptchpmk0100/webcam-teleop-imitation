# 01: Env tracer
What to build: a runnable project (uv, Python 3.12, pytest, one command that runs the tests) with the SO-101 scene, a cube, reset by seed to a cube position, step on joint targets at 20 Hz, a success check read from cube height in MuJoCo state, and snapshot and restore of full state.
Blocked by: None (can start immediately)
Kind: AFK
- [ ] The test command exits 0 and fails (nonzero) if the success check is wrong: tests place the cube above 5 cm for 1 s and below, and compare the check with a direct height read.
- [ ] Same seed gives the same cube position; snapshot then restore gives identical state.
- [ ] Verifier written first and seen red for the right reason before the implementation.

Status: CLOSED 2026-11-14. Verified: `make test` exit 0 (16 passed); 4 mutations of the success check each turn it red (exit 1). Cube success is measured on the cube bottom (lift_height), cube is 4 cm, yaw fixed.
