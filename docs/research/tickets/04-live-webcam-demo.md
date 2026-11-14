# 04: Live webcam demo (week-1 core)
What to build: one command that opens the webcam in the lamp-lit manual-exposure mode, runs MediaPipe, feeds the teleop core and shows the simulation window, printing FPS and hand-detection rate.
Blocked by: 03
Kind: HITL
- [ ] Khoi moves the arm with his hand and judges control quality in a session.
- [ ] The command prints at least 25 FPS in the recording mode, with a lamp, and says so.
- [ ] The camera mode sets exposure explicitly and restores auto exposure on exit.

Build notes 2026-11-14 (waiting for Khoi, G5): the adapter is built and tested at its boundaries (6 tests: MediaPipe-to-frame conversion, real-time 20 Hz pacing, stats, camera release on crash). Real run, headless, 8 s, manual exposure 330, nobody in view: 30.0 FPS (PASS for 25), MediaPipe 19.4 ms mean, 0% hand detection (no hand in view, so tracking quality is NOT measured). Camera is returned in auto exposure on exit (inferred from frame rate, not read back).
For Khoi to judge: run `make demo`, lamp on your hand, SPACE toggles the clutch (a toggle, not hold-to-engage as in G1: decision for you), move and pinch. Not verified by me: the simulation window, the preview window, control feel, right-hand-only (not enforced: any single hand is used), depth mapping quality. Direction mapping to check: hand to your right moves the arm to -y, up lifts, closer to the camera moves forward.

G5 verdict, Khoi, 2026-11-14 (two rounds): "does most of the functions". Round 1 (before fixes): movement did not follow the hand well, gripper close was loose, the pinch was hidden from the camera when the fingers closed, arm felt loose. Fixes shipped in e23f261 (newest-frame camera reader, faster filter, tilt-invariant depth cue, G gripper key, view from behind the arm, `--log` and `make diagnose`). Round 2 (after fixes), 30 FPS, hand detected 81-90 percent, many short HAND LOST flags: the arm still shakes (tracking noise or lag, to deal with later), it is very hard to navigate by hand and almost impossible to pick the cube up. No session log was recorded, so noise and lag are still not measured. `make demo` also ends with exit 139 (segfault) after the summary is printed, in both windowed runs; cause unknown.
Status: ACCEPTED WITH KNOWN LIMITS as the week-1 adapter (the pipeline works end to end at 30 FPS). Teleop quality is an open risk, carried to a later teleop-tuning ticket: record a log, then try lower and speed-dependent gain, a rate limit on the target, depth on a key, and a grasp assist. Not accepted as demo-recording quality (that is ticket 10).
