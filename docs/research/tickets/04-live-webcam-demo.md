# 04: Live webcam demo (week-1 core)
What to build: one command that opens the webcam in the lamp-lit manual-exposure mode, runs MediaPipe, feeds the teleop core and shows the simulation window, printing FPS and hand-detection rate.
Blocked by: 03
Kind: HITL
- [ ] Khoi moves the arm with his hand and judges control quality in a session.
- [ ] The command prints at least 25 FPS in the recording mode, with a lamp, and says so.
- [ ] The camera mode sets exposure explicitly and restores auto exposure on exit.

Build notes 2026-11-14 (waiting for Khoi, G5): the adapter is built and tested at its boundaries (6 tests: MediaPipe-to-frame conversion, real-time 20 Hz pacing, stats, camera release on crash). Real run, headless, 8 s, manual exposure 330, nobody in view: 30.0 FPS (PASS for 25), MediaPipe 19.4 ms mean, 0% hand detection (no hand in view, so tracking quality is NOT measured). Camera is returned in auto exposure on exit (inferred from frame rate, not read back).
For Khoi to judge: run `make demo`, lamp on your hand, SPACE toggles the clutch (a toggle, not hold-to-engage as in G1: decision for you), move and pinch. Not verified by me: the simulation window, the preview window, control feel, right-hand-only (not enforced: any single hand is used), depth mapping quality. Direction mapping to check: hand to your right moves the arm to -y, up lifts, closer to the camera moves forward.
