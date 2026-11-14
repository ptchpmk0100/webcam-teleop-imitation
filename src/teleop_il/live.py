"""Live adapter: webcam -> MediaPipe hand landmarks -> teleop core -> simulated arm, paced in real time.

Run with `make demo`. SPACE in the simulation window toggles the clutch (arm follows the hand only while engaged).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from teleop_il.env import CONTROL_DT
from teleop_il.teleop import HandFrame

MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "models" / "hand_landmarker.task"
RECORDING_FPS = 25.0                       # the camera loop must reach this in recording mode


@dataclass(frozen=True)
class SessionStats:
    frames: int
    detected_frames: int
    seconds: float
    fps: float
    stage_ms: dict            # stage name -> (mean ms, p95 ms)
    sim_seconds: float
    reached_success: bool


def user_view(cam) -> None:
    """Look from behind the arm towards the workspace, like the user sits: the user's right is the screen's right."""
    cam.type = 0                                       # mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 0.0, -25.0, 0.95
    cam.lookat[:] = (0.2, 0.0, 0.07)


def landmarks_to_frame(t: float, result, engage: bool, close_gripper: bool = False) -> HandFrame:
    """Turn a MediaPipe HandLandmarker result into the teleop core's input frame."""
    if not result.hand_landmarks:
        return HandFrame(t=t, image_landmarks=None, world_landmarks=None, engage=engage, close_gripper=close_gripper)
    image = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
    world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
    return HandFrame(t=t, image_landmarks=image, world_landmarks=world, engage=engage, close_gripper=close_gripper)


def run_session(camera, tracker, teleop, env, *, secs: float, clock, engaged, viewer=None, preview=None,
                status=None, close_gripper=lambda: False, log=None) -> SessionStats:
    """Run the live loop for `secs` seconds of `clock` time; the simulation advances in real time at 20 Hz."""
    stages = {"read": [], "tracking": [], "teleop": [], "sim": []}
    frames = detected = 0
    sim_t = 0.0
    reached = False
    t0 = last_status = clock()
    try:
        while clock() - t0 < secs and (viewer is None or viewer.is_running()):
            a = clock()
            ok, image = camera.read()
            b = clock()
            if not ok:
                continue
            result = tracker.detect(image, int((b - t0) * 1000))
            c = clock()
            key_closed = close_gripper()
            frame = landmarks_to_frame(b - t0, result, engaged(), key_closed)
            out = teleop.update(frame)
            d = clock()
            while sim_t + CONTROL_DT <= clock() - t0:
                if viewer is not None:
                    with viewer.lock():
                        env.step(out.joint_targets)
                else:
                    env.step(out.joint_targets)
                sim_t += CONTROL_DT
            if viewer is not None:
                viewer.sync()
            e = clock()
            reached = reached or env.success()
            if log is not None:
                log.add(t=b - t0, detected=bool(result.hand_landmarks), engaged=out.engaged, info=out.info,
                        ee=env.ee_position(), gripper=float(out.joint_targets[5]), key_closed=key_closed,
                        image=frame.image_landmarks, world=frame.world_landmarks)
            frames += 1
            detected += bool(result.hand_landmarks)
            for name, dur in zip(stages, (b - a, c - b, d - c, e - d)):
                stages[name].append(dur * 1e3)
            if preview is not None and preview(image, result, out, frames / max(clock() - t0, 1e-6)) is False:
                break
            if status is not None and clock() - last_status >= 1.0:
                last_status = clock()
                status(frames, detected, frames / max(clock() - t0, 1e-6), out)
    finally:
        camera.close()
    seconds = clock() - t0
    summary = {k: (float(np.mean(v)), float(np.percentile(v, 95))) for k, v in stages.items() if v}
    return SessionStats(frames=frames, detected_frames=detected, seconds=seconds, fps=frames / max(seconds, 1e-6),
                        stage_ms=summary, sim_seconds=sim_t, reached_success=reached)


class Camera:
    """V4L2 webcam. Exposure is set explicitly because V4L2 settings persist on the device after we exit."""

    def __init__(self, device: int = 0, exposure: int | None = 330, width: int = 640, height: int = 480,
                 fps: int = 30) -> None:
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open /dev/video{device}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # Do NOT set CAP_PROP_BUFFERSIZE=1: measured on this webcam it halves the rate (15.2 FPS instead of 30.2).
        if exposure:                                   # manual exposure: fixed frame rate, needs a lamp
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        for _ in range(5):
            self._cap.read()

    def read(self):
        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.set(self._cv2.CAP_PROP_AUTO_EXPOSURE, 3)    # give the camera back in auto exposure
            self._cap.release()
            self._cap = None


class LatestFrameCamera:
    """Reads the camera in a thread and hands over only the newest frame.

    When tracking takes about as long as a camera frame (MediaPipe ~25 ms against 33 ms), frames queue up inside the
    driver and reach the loop 100 ms or more late. Skipping the stale ones keeps the delay near one frame.
    """

    def __init__(self, camera) -> None:
        import threading
        self._camera, self._cond = camera, threading.Condition()
        self._frame, self._ok, self._seq, self._taken, self._stop = None, False, 0, 0, False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop:
            try:
                ok, frame = self._camera.read()
            except Exception:                              # the camera died; let read() report it and stop
                ok, frame = False, None
                self._stop = True
            with self._cond:
                self._ok, self._frame, self._seq = ok, frame, self._seq + 1
                self._cond.notify_all()
            if not ok and not self._stop:
                time.sleep(0.005)                          # a dropped frame: try again without spinning

    def read(self):
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._taken or not self._thread.is_alive())
            self._taken = self._seq
            return self._ok, self._frame

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=3.0)
        self._camera.close()


class HandTracker:
    """MediaPipe HandLandmarker (pretrained), one hand, CPU."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self._mp, self._cv2 = mp, __import__("cv2")
        options = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path), delegate=mpp.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=1)
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr, timestamp_ms: int):
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()


def _draw_preview(cv2, key_closed):
    def preview(image, result, out, fps):
        shown = cv2.flip(image, 1)                    # mirrored, like a selfie view
        if result.hand_landmarks:
            h, w = shown.shape[:2]
            for p in result.hand_landmarks[0]:
                cv2.circle(shown, (int((1 - p.x) * w), int(p.y * h)), 3, (0, 255, 0), -1)
        text = f"{fps:4.1f} FPS  {'ENGAGED' if out.engaged else 'clutch released'}"
        if out.tracking_lost:
            text += "  HAND LOST"
        cv2.putText(shown, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        gap = out.info.get("pinch_gap_m", float("nan"))
        grip = "GRIPPER CLOSED" if out.joint_targets[5] < 0 else "gripper open"
        cv2.putText(shown, f"pinch gap {100 * gap:4.1f} cm (closes < 3, opens > 5)  {grip}{' (key)' if key_closed() else ''}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("hand tracking (q to quit)", shown)
        return cv2.waitKey(1) & 0xFF != ord("q")
    return preview


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drive the simulated SO-101 with your hand through the webcam.")
    ap.add_argument("--secs", type=float, default=600.0, help="stop after this many seconds")
    ap.add_argument("--exposure", type=int, default=330, help="manual exposure in 100 us units (lamp needed)")
    ap.add_argument("--auto-exposure", action="store_true", help="use the camera's auto exposure (about 17 FPS here)")
    ap.add_argument("--headless", action="store_true", help="no windows; clutch always engaged; prints timing")
    ap.add_argument("--no-preview", action="store_true", help="no camera preview window")
    ap.add_argument("--seed", type=int, default=0, help="cube position seed")
    ap.add_argument("--log", type=Path, help="save a per-frame session log to this NEW .npz file (then run teleop_il.diagnose on it)")
    args = ap.parse_args(argv)

    from teleop_il.env import Env
    from teleop_il.teleop import Teleop
    if not MODEL_PATH.exists():
        print(f"missing {MODEL_PATH}; run `make setup`", file=sys.stderr)
        return 2
    env = Env()
    env.reset(args.seed)
    teleop = Teleop()
    tracker = HandTracker()
    camera = LatestFrameCamera(Camera(exposure=None if args.auto_exposure else args.exposure))
    if args.log is not None and args.log.exists():
        print(f"{args.log} already exists; logs go to new files", file=sys.stderr)
        return 2
    state = {"engaged": args.headless, "key_closed": False}
    viewer = preview = None
    if not args.headless:
        import mujoco.viewer

        def on_key(keycode):
            if keycode == 32:                          # SPACE toggles the clutch
                state["engaged"] = not state["engaged"]
                print("clutch", "ENGAGED" if state["engaged"] else "released")
            elif keycode == ord("G"):                  # G toggles the gripper (the pinch can be hidden from the camera)
                state["key_closed"] = not state["key_closed"]
                print("gripper key", "CLOSED" if state["key_closed"] else "released")
        viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key)
        user_view(viewer.cam)
        if not args.no_preview:
            import cv2
            preview = _draw_preview(cv2, lambda: state["key_closed"])
        print("In the simulation window: SPACE toggles the clutch, G toggles the gripper (a pinch closes it too).")
        print("Hold your hand in view, press SPACE, then move.")

    def status(frames, detected, fps, out):
        print(f"  {fps:5.1f} FPS | hand detected {100 * detected / frames:3.0f}% | "
              f"{'engaged' if out.engaged else 'released'}{' | HAND LOST' if out.tracking_lost else ''}"
              f"{' | cube lifted!' if env.success() else ''}")

    from teleop_il.diagnose import SessionLog
    log = SessionLog() if args.log is not None else None
    try:
        stats = run_session(camera, tracker, teleop, env, secs=args.secs, clock=time.perf_counter,
                            engaged=lambda: state["engaged"], viewer=viewer, preview=preview, status=status,
                            close_gripper=lambda: state["key_closed"], log=log)
    finally:
        tracker.close()
        if viewer is not None:
            viewer.close()
    if log is not None and len(log):
        log.save(args.log)
        print(f"session log saved to {args.log}; summarise with: make diagnose LOG={args.log}")
    print(f"\n{stats.frames} frames in {stats.seconds:.1f} s = {stats.fps:.1f} FPS; "
          f"hand detected in {100 * stats.detected_frames / max(stats.frames, 1):.0f}% of frames")
    for name, (mean, p95) in stats.stage_ms.items():
        print(f"  {name:9s} mean {mean:6.2f} ms   p95 {p95:6.2f} ms")
    verdict = "PASS" if stats.fps >= RECORDING_FPS else "FAIL"
    print(f"recording-mode check (>= {RECORDING_FPS:.0f} FPS): {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Everything is closed and saved by now. MuJoCo's viewer thread segfaults in glfw.terminate() during normal
    # interpreter shutdown (faulthandler: viewer.py line 531), so leave without running the shutdown handlers.
    os._exit(code)
