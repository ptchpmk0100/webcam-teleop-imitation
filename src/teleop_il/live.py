"""Live adapter: webcam -> MediaPipe hand landmarks -> teleop core -> simulated arm, paced in real time.

Run with `make demo` (practice) or `make trial` (10 counted attempts). Keys in the window: SPACE clutch (the arm
follows the hand only while engaged), G gripper, W/S forward and back, R restart the attempt, N skip it, Q quit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# The demo window renders offscreen and shows the pictures with OpenCV; this must be chosen before MuJoCo is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

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


def landmarks_to_frame(t: float, result, engage: bool, close_gripper: bool = False,
                       forward_m: float = 0.0) -> HandFrame:
    """Turn a MediaPipe HandLandmarker result into the teleop core's input frame."""
    if not result.hand_landmarks:
        return HandFrame(t=t, image_landmarks=None, world_landmarks=None, engage=engage, close_gripper=close_gripper,
                         forward_m=forward_m)
    image = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
    world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
    return HandFrame(t=t, image_landmarks=image, world_landmarks=world, engage=engage, close_gripper=close_gripper,
                     forward_m=forward_m)


class Attempts:
    """Timed attempts on fixed seeds (grill Q1: at most 30 s each; success read from the simulation, never reported).

    `counted=True` is a trial: the session ends after the last seed. Otherwise it is practice and the seeds repeat.
    The keys call restart() (same seed again, nothing recorded) and skip() (recorded as a failure)."""

    def __init__(self, seeds, limit_s: float = 30.0, counted: bool = False) -> None:
        self.seeds, self.limit_s, self.counted = tuple(seeds), limit_s, counted
        self.results: list[dict] = []
        self._index, self._start, self._pending = 0, None, None

    @property
    def seed(self) -> int:
        return self.seeds[self._index % len(self.seeds)]

    @property
    def number(self) -> int:
        return self._index + 1

    @property
    def done(self) -> bool:
        return self.counted and self._index >= len(self.seeds)

    def elapsed(self, t: float) -> float:
        return 0.0 if self._start is None else t - self._start

    def restart(self) -> None:
        self._pending = "restart"

    def skip(self) -> None:
        self._pending = "skipped"

    def check(self, t: float, lifted: bool, assist_fired: int = 0) -> bool:
        """Called once per frame; True means: load `seed` into the simulation and start the next attempt now."""
        if self._start is None:
            self._start = t
        pending, self._pending = self._pending, None
        if pending == "restart":
            self._start = t
            return True
        outcome = "lifted" if lifted else pending if pending else "time up" if t - self._start > self.limit_s else None
        if outcome is None:
            return False
        self.results.append({"seed": self.seed, "outcome": outcome, "success": outcome == "lifted",
                             "seconds": round(t - self._start, 2), "assist_fired": int(assist_fired),
                             "counted": self.counted})
        self._index += 1
        self._start = t
        return not self.done


def run_session(camera, tracker, teleop, env, *, secs: float, clock, engaged, viewer=None, preview=None,
                status=None, close_gripper=lambda: False, forward=lambda: 0.0, attempts=None, last_key=lambda: "",
                log=None) -> SessionStats:
    """Run the live loop for `secs` seconds of `clock` time; the simulation advances in real time at 20 Hz.

    With `attempts`, each attempt ends on a lift (judged by `env.success()`) or its time limit, and the next seed is
    loaded into the simulation; a counted trial ends the session after its last seed."""
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
            frame = landmarks_to_frame(b - t0, result, engaged(), key_closed, forward())
            out = teleop.update(frame)
            d = clock()
            while sim_t + env.model.opt.timestep <= clock() - t0:     # physics in 5 ms slices: smooth motion, not 50 ms jumps
                if viewer is not None:
                    with viewer.lock():
                        env.substep(out.joint_targets)
                else:
                    env.substep(out.joint_targets)
                sim_t += env.model.opt.timestep
            if viewer is not None:
                viewer.sync()
            e = clock()
            lifted = env.success()
            reached = reached or lifted
            if log is not None:
                log.add(t=b - t0, detected=bool(result.hand_landmarks), engaged=out.engaged, info=out.info,
                        ee=env.ee_position(), gripper=float(out.joint_targets[5]), key_closed=key_closed,
                        image=frame.image_landmarks, world=frame.world_landmarks, cube=env.cube_position(),
                        forward_m=frame.forward_m, attempt=attempts.number if attempts is not None else 0,
                        key=last_key())
            frames += 1
            detected += bool(result.hand_landmarks)
            for name, dur in zip(stages, (b - a, c - b, d - c, e - d)):
                stages[name].append(dur * 1e3)
            if preview is not None and preview(image, result, out, frames / max(clock() - t0, 1e-6)) is False:
                break
            if status is not None and clock() - last_status >= 1.0:
                last_status = clock()
                status(frames, detected, frames / max(clock() - t0, 1e-6), out)
            if attempts is not None:
                if attempts.check(clock() - t0, lifted, out.info.get("assist_fired", 0)):
                    env.reset(attempts.seed)
                    teleop.reset()
                    sim_t = clock() - t0                           # the new attempt's physics starts now
                elif attempts.done:
                    break
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

    def __init__(self, model_path: Path = MODEL_PATH, confidence: float = 0.3) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self._mp, self._cv2 = mp, __import__("cv2")
        options = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path), delegate=mpp.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            # defaults are 0.5; half of the dropouts in Khoi's log started while the fingers were pinched
            min_hand_detection_confidence=confidence, min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence)
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr, timestamp_ms: int):
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()


TRIAL_SEEDS = tuple(range(500, 510))       # the counted trial (same seeds as the prototype trials)
PRACTICE_SEEDS = tuple(range(600, 620))    # practice uses other cubes, so the trial is not rehearsed
FORWARD_STEP_M = 0.005                     # one W/S press; holding the key repeats it (about 15 cm/s)


def plain_letter_keys(environ=os.environ) -> None:
    """Make letter keys reach the OpenCV (Qt) window as plain keys. With an input method between keyboard and window
    (here ibus with Unikey, Telex typing: W and S are composing keys), W, S and G never arrived in Khoi's try4 session
    (0 presses in 6 attempts) while SPACE did. Must run before the first window opens."""
    environ["QT_IM_MODULE"] = "none"
    environ.pop("XMODIFIERS", None)


_KEY_NAMES = {32: "SPACE", 27: "ESC"}


def apply_key(k: int, state: dict, attempts) -> bool:
    """One key from cv2.waitKey (already masked to 8 bits; -1 & 0xFF = 255 means none). Returns False to quit.
    Letters work in either case; every key received is recorded in state["last_key"] so the screen can show it."""
    if k == 255:
        return True
    ch = chr(k).lower() if 32 < k < 127 else ""
    name = _KEY_NAMES.get(k, ch or str(k))
    used = True
    if ch == "q" or k == 27:
        return False
    if k == 32:
        state["engaged"] = not state["engaged"]
    elif ch == "g":
        state["key_closed"] = not state["key_closed"]
    elif ch == "w":
        state["forward"] += FORWARD_STEP_M
    elif ch == "s":
        state["forward"] -= FORWARD_STEP_M
    elif ch == "r":
        attempts.restart()
        state["key_closed"] = False
    elif ch == "n":
        attempts.skip()
    else:
        used = False
    state["last_key"] = name if used else f"{name} (unused)"
    state["key_event"] = name                       # read once by the session log, so each press is logged once
    return True


def start_of_attempt(state: dict, attempts) -> list[str]:
    """Report results not reported yet; if there are any, a new attempt has begun, so open the gripper again (in try5
    the closed gripper carried over from attempt 3 into attempt 4)."""
    new = attempts.results[state.get("printed", 0):]
    state["printed"] = len(attempts.results)
    if new:
        state["key_closed"] = False
    return [f"attempt {attempts.results.index(r) + 1}: {r['outcome']} in {r['seconds']:.1f} s "
            f"(assist fired {r['assist_fired']}x)" for r in new]


def _window(cv2, display, state, attempts, env):
    """The demo window's per-frame callback: draw everything, handle the keys. Returns False to quit."""
    seen_results = {"banner": "", "until": 0.0}

    def preview(image, result, out, fps):
        now = time.perf_counter()
        for line in start_of_attempt(state, attempts):
            last = attempts.results[-1]
            seen_results["banner"], seen_results["until"] = ("LIFTED!" if last["success"] else last["outcome"]), now + 1.5
            print(line, flush=True)
        info, ee = out.info, env.ee_position()
        assist = info.get("assist", "off")
        lifted = sum(r["success"] for r in attempts.results)
        lines = [
            f"{fps:4.1f} FPS   hand {'seen' if result.hand_landmarks else 'NOT SEEN'}",
            "ENGAGED (SPACE releases)" if out.engaged else "clutch released: press SPACE",
            f"gripper {'CLOSED' if state['key_closed'] else 'open'} (G)   forward {100 * state['forward']:+.1f} cm (W/S)",
            f"assist {'ACTIVE' if assist == 'active' else 'ready: press G' if assist == 'ready' else '-'}"
            f"   fired {info.get('assist_fired', 0)}x",
            f"{100 * info.get('grasp_distance_m', float('nan')):4.1f} cm from the grasp point (assist below 5)",
            f"gripper {100 * ee[2]:4.1f} cm above the table",
            f"attempt {attempts.number}{'/' + str(len(attempts.seeds)) if attempts.counted else ' (practice)'}"
            f"   {max(attempts.limit_s - attempts.elapsed(now - state['t0']), 0):4.1f} s left",
            f"lifted {lifted} of {len(attempts.results)}",
            f"last key: {state.get('last_key') or 'none yet'}   R restart  N skip  Q quit",
        ]
        banner = seen_results["banner"] if now < seen_results["until"] else ""
        shown = display.compose(image, lines, assist=assist, banner=banner)
        if result.hand_landmarks:                       # landmarks on the mirrored camera tile (top right)
            h, w = display._small_size
            x0 = shown.shape[1] - w
            for p in result.hand_landmarks[0]:
                cv2.circle(shown, (x0 + int((1 - p.x) * w), int(p.y * h)), 2, (0, 255, 0), -1)
        cv2.imshow("webcam teleop (Q quits)", shown)
        return apply_key(cv2.waitKey(1) & 0xFF, state, attempts)
    return preview


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drive the simulated SO-101 with your hand through the webcam. "
                                 "Hand left-right and up-down move the gripper, W/S move it forward and back, "
                                 "G closes it; near the cube the grasp assist finishes the grasp, you lift.")
    ap.add_argument("--secs", type=float, default=1800.0, help="stop after this many seconds")
    ap.add_argument("--exposure", type=int, default=330, help="manual exposure in 100 us units (lamp needed)")
    ap.add_argument("--auto-exposure", action="store_true", help="use the camera's auto exposure (about 17 FPS here)")
    ap.add_argument("--headless", action="store_true", help="no window; clutch always engaged; prints timing")
    ap.add_argument("--trial", action="store_true",
                    help=f"counted trial: {len(TRIAL_SEEDS)} attempts on seeds {TRIAL_SEEDS[0]}-{TRIAL_SEEDS[-1]}, "
                         "results saved under runs/")
    ap.add_argument("--attempt-secs", type=float, default=30.0, help="time limit per attempt (grill Q1/Q8: 30 s)")
    ap.add_argument("--no-assist", action="store_true", help="switch the grasp assist off")
    ap.add_argument("--log", type=Path, help="save a per-frame session log to this NEW .npz file (then run teleop_il.diagnose on it)")
    args = ap.parse_args(argv)

    from teleop_il.assist import AssistedTeleop
    from teleop_il.env import Env
    from teleop_il.teleop import Teleop
    if not MODEL_PATH.exists():
        print(f"missing {MODEL_PATH}; run `make setup`", file=sys.stderr)
        return 2
    if args.log is not None and args.log.exists():
        print(f"{args.log} already exists; logs go to new files", file=sys.stderr)
        return 2
    attempts = Attempts(TRIAL_SEEDS if args.trial else PRACTICE_SEEDS, limit_s=args.attempt_secs, counted=args.trial)
    env = Env()
    env.reset(attempts.seed)
    teleop = Teleop() if args.no_assist else AssistedTeleop(Teleop(), ee=env.ee_position, cube=env.cube_position)
    tracker = HandTracker()
    camera = LatestFrameCamera(Camera(exposure=None if args.auto_exposure else args.exposure))
    state = {"engaged": args.headless, "key_closed": False, "forward": 0.0, "t0": time.perf_counter(), "last_key": ""}
    preview = display = None
    if not args.headless:
        plain_letter_keys()
        import cv2

        from teleop_il.display import Display
        display = Display(env)
        preview = _window(cv2, display, state, attempts, env)
        print("Keys (in the window): SPACE clutch, G gripper, W/S forward/back, R restart, N skip, Q quit.")
        print("Hold your hand in view, press SPACE, then move. Near the cube, press G: the robot finishes the grasp.")

    def status(frames, detected, fps, out):
        if args.headless:
            print(f"  {fps:5.1f} FPS | hand detected {100 * detected / frames:3.0f}% | "
                  f"{'engaged' if out.engaged else 'released'}{' | HAND LOST' if out.tracking_lost else ''}")

    from teleop_il.diagnose import SessionLog
    log = SessionLog() if args.log is not None else None
    try:
        state["t0"] = time.perf_counter()
        stats = run_session(camera, tracker, teleop, env, secs=args.secs, clock=time.perf_counter,
                            engaged=lambda: state["engaged"], preview=preview, status=status,
                            close_gripper=lambda: state["key_closed"], forward=lambda: state["forward"],
                            last_key=lambda: state.pop("key_event", ""),
                            attempts=attempts, log=log)
    finally:
        tracker.close()
        if display is not None:
            import cv2
            cv2.destroyAllWindows()
    if log is not None and len(log):
        log.save(args.log)
        print(f"session log saved to {args.log}; summarise with: make diagnose LOG={args.log}")
    print(f"\n{stats.frames} frames in {stats.seconds:.1f} s = {stats.fps:.1f} FPS; "
          f"hand detected in {100 * stats.detected_frames / max(stats.frames, 1):.0f}% of frames")
    for name, (mean, p95) in stats.stage_ms.items():
        print(f"  {name:9s} mean {mean:6.2f} ms   p95 {p95:6.2f} ms")
    for line in start_of_attempt(state, attempts):                 # the last result ends the loop before the window
        print(line)                                                # could report it
    if attempts.results:
        k = sum(r["success"] for r in attempts.results)
        print(f"{'TRIAL' if args.trial else 'practice'}: lifted {k} of {len(attempts.results)} attempts"
              f"{' (bar: 5 of 10)' if args.trial else ''}; assist "
              f"{'off' if args.no_assist else 'on'}")
    if args.trial and attempts.results:
        import json
        out_dir = Path(__file__).resolve().parents[2] / "runs"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"trial_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "x") as f:                                  # "x": never overwrites an earlier trial
            json.dump({"assist": not args.no_assist, "attempt_secs": args.attempt_secs,
                       "results": attempts.results}, f, indent=1)
        print(f"trial results saved to {path}")
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
