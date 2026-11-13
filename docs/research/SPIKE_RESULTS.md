# Feasibility spike results (2026-11-14)

Throwaway spike, run on Khoi's laptop (GTX 1650, 640x480 webcam). Scripts and the venv live in the Claude scratchpad, not in the repo. Every number below comes from a command run in that session. Nothing here has been run twice, so treat timings as single samples.

Versions pulled by pip on Python 3.12: mujoco 4.0.1, mink (installed, no version attribute), mediapipe 1.2.3, numpy 2.5.4. mediapipe also pulled opencv-contrib-python 5.0.0.93.

## Verdict against the G1 thresholds

| Check | Threshold | Result | Status |
|---|---|---|---|
| End-to-end loop rate | 25 FPS or better | 30.0 FPS with manual exposure (image dark, mean brightness 25 of 255). 16.6 FPS with auto exposure (brightness about 103). | PASS only with manual exposure |
| Hand-to-arm latency | 100 ms or less | Software part only: 19-20 ms mean, 32-34 ms p95 (manual exposure). Camera exposure and USB delay not measured. | Partly measured |
| `mink` IK rate | 50 Hz or better | 0.15 ms per iteration in the reach sweep, 0.4 ms in the live loop. | PASS |
| SO-101 reach | whole spawn area | See below. Default box at x=0.5 is out of reach. | PASS with a smaller area |
| Offscreen render | measure | EGL: 1.65 ms per frame at 96x96, 1.49 ms at 128x128, image not blank. osmesa is not available. | measured |

## Webcam frame rate (the one real problem)

- Capture alone, no processing, gives 16.7 FPS in auto exposure at every format and size tried (YUYV, MJPG, 320x240, 640x480, 1280x720). The cause is exposure time, not format.
- Manual exposure gives 30.3 FPS but a dark image: brightness 22 of 255 at the longest exposure that still allows 30 FPS (330 units of 100 microseconds).
- So the camera trades frame rate for brightness in this room's light. More light is the likely fix, but it is untested.
- V4L2 exposure settings persist on the camera device after a script exits. Any script must set auto-exposure explicitly (`CAP_PROP_AUTO_EXPOSURE` = 3 for auto) or it inherits the last setting.
- The G1 fallback "lower resolution and the lite model" would not help, because the limit is exposure.

## Pipeline timing (webcam, MediaPipe HandLandmarker float16 on CPU, one mink iteration, 4 mujoco steps per frame)

Manual exposure 330, 8 s, 241 frames: read 13.9 ms, MediaPipe 18.9 ms, IK 0.4 ms, sim 0.1 ms. Loop rate is camera-bound at 30 FPS. Compute alone is about 20 ms per frame, so the loop could reach about 50 FPS with a faster camera mode.

Auto exposure, 8 s, 133 frames, no hand in view: 16.6 FPS. MediaPipe took 28.7 ms per frame when no hand was found, against 18.9 ms with a hand.

Hand detection in the dark runs was 42 to 62 percent of frames. Whether a hand was held steadily in view was not controlled, so this is not a detection-quality measurement.

## SO-101 facts (checked by loading the model)

- 6 position actuators: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (kp about 998). nq 13 with the box's free joint. Physics timestep 5 ms.
- `scene.xml` has no table and no keyframe. `scene_box.xml` has the box and a `pickup` keyframe, which moves the box to (0.219, 0.024, 0.02).
- Reach, with `mink` and the gripperframe site, position tolerance 3 mm, cube-centre height z=0.03, floor as the table:
  - Position-only IK reaches 80 of 88 grid points (x 0.10-0.45, y -0.25 to 0.25), but tilt was not measured for that run.
  - With the orientation weight 0.02 and the keyframe orientation as target, 60 of 88 points are reached with tilt 30 degrees or less. The rectangle x 0.10-0.30, y -0.15 to +0.15 is reached with tilt at most 10 degrees.
  - Orientation weights 0.2 and 1.0 shrink the area to about x 0.10-0.25 (32 and 14 of 88 points).
  - The default box position x=0.5 is outside the reachable area.
- Not modelled in this sweep: collision with the floor or the cube, gripper actuator limits, grasp physics.

## Claims from the sub-agent, now checked

- SO-101 has a gripper and a `scene_box.xml`: confirmed.
- The box is probably out of reach as spawned: confirmed.
- `pip install mink` and `mediapipe` work on Python 3.12: confirmed.
- mink real-time at 50 Hz: confirmed on this CPU.

## Not tested yet

- MediaPipe world-landmark depth (z) quality and the One-Euro filter.
- Pinch to gripper mapping, and whether the SO-101 gripper can lift the cube.
- Physical camera-to-arm latency.
- ChArUco scale check.
- C++ build of MuJoCo plus ONNX Runtime.
