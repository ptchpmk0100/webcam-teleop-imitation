"""Live adapter at its boundaries: the camera and MediaPipe are faked, the loop and teleop are real."""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import hand
from teleop_il.env import Env
from teleop_il.live import landmarks_to_frame, run_session
from teleop_il.teleop import Teleop


def lm(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def mediapipe_result(with_hand=True):
    if not with_hand:
        return SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[])
    img = [lm(i / 100, 1 - i / 100, i / 1000) for i in range(21)]
    world = [lm(i / 1000, -i / 1000, 2 * i / 1000) for i in range(21)]
    return SimpleNamespace(hand_landmarks=[img], hand_world_landmarks=[world])


class FakeClock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now


class FakeCamera:
    """Delivers a frame every 1/30 s of fake time, like a 30 FPS camera. Records whether it was closed."""
    def __init__(self, clock, fail_after=None):
        self.clock, self.closed, self.reads, self.fail_after = clock, False, 0, fail_after
    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise RuntimeError("camera unplugged")
        self.clock.now += 1 / 30
        return True, np.zeros((480, 640, 3), np.uint8)
    def close(self): self.closed = True


class FakeTracker:
    """Returns a hand in every frame whose index is not in `missing`."""
    def __init__(self, missing=()):
        self.n, self.missing = 0, set(missing)
    def detect(self, image, timestamp_ms):
        self.n += 1
        return mediapipe_result(with_hand=self.n not in self.missing)


def test_a_mediapipe_result_becomes_a_hand_frame_with_both_landmark_sets():
    frame = landmarks_to_frame(1.5, mediapipe_result(), engage=True)
    assert frame.t == 1.5 and frame.engage is True
    assert frame.image_landmarks.shape == (21, 3) and frame.world_landmarks.shape == (21, 3)
    assert frame.image_landmarks[5] == pytest.approx([0.05, 0.95, 0.005])
    assert frame.world_landmarks[5] == pytest.approx([0.005, -0.005, 0.010])


def test_a_result_without_a_hand_becomes_a_frame_without_landmarks():
    frame = landmarks_to_frame(2.0, mediapipe_result(with_hand=False), engage=False)
    assert frame.image_landmarks is None and frame.world_landmarks is None


def test_the_simulation_runs_in_real_time_at_20_hz_whatever_the_camera_rate():
    clock = FakeClock()
    env = Env(); env.reset(0)
    stats = run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=3.0, clock=clock, engaged=lambda: True)
    assert stats.sim_seconds == pytest.approx(3.0, abs=0.05)         # 60 steps of 0.05 s in 3 s of camera time
    assert env.data.time == pytest.approx(3.0, abs=0.05)


def test_stats_report_camera_rate_and_hand_detection():
    clock = FakeClock()
    env = Env(); env.reset(0)
    stats = run_session(FakeCamera(clock), FakeTracker(missing=range(1, 91, 2)), Teleop(), env,
                        secs=3.0, clock=clock, engaged=lambda: False)
    assert stats.fps == pytest.approx(30.0, abs=1.0)
    assert stats.detected_frames / stats.frames == pytest.approx(0.5, abs=0.02)
    assert set(stats.stage_ms) >= {"read", "tracking", "teleop", "sim"}


def test_the_camera_is_released_even_when_the_loop_crashes():
    clock = FakeClock()
    cam = FakeCamera(clock, fail_after=10)
    env = Env(); env.reset(0)
    with pytest.raises(RuntimeError):
        run_session(cam, FakeTracker(), Teleop(), env, secs=3.0, clock=clock, engaged=lambda: True)
    assert cam.closed is True


def test_the_camera_is_released_after_a_normal_run():
    clock = FakeClock()
    cam = FakeCamera(clock)
    env = Env(); env.reset(0)
    run_session(cam, FakeTracker(), Teleop(), env, secs=1.0, clock=clock, engaged=lambda: True)
    assert cam.closed is True


# ---- G5 feedback: the simulation window should look from behind the arm, like the user sits ----

def test_the_simulation_view_puts_the_arms_minus_y_side_on_the_right_of_the_screen():
    import mujoco
    from teleop_il.live import user_view
    env = Env(); env.reset(0)
    cam = mujoco.MjvCamera()
    user_view(cam)
    scene = mujoco.MjvScene(env.model, maxgeom=100)
    mujoco.mjv_updateScene(env.model, env.data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    gl = scene.camera[0]
    forward, up = np.array(gl.forward), np.array(gl.up)
    right = np.cross(forward, up)
    assert right @ np.array([0.0, -1.0, 0.0]) > 0.9           # user's right = arm -y = screen right
    assert forward @ np.array([1.0, 0.0, 0.0]) > 0.7          # looking from behind the arm towards the cube area


def test_a_session_log_records_one_row_per_frame_with_the_real_gripper_position():
    from teleop_il.diagnose import SessionLog
    clock = FakeClock()
    env = Env(); env.reset(0)
    log = SessionLog()
    stats = run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=1.0, clock=clock,
                        engaged=lambda: True, log=log)
    assert len(log) == stats.frames
    assert log.last_ee == pytest.approx([0.2, 0.0, 0.09], abs=0.01)


def test_the_gripper_key_state_reaches_the_teleop_frame():
    clock = FakeClock()
    env = Env(); env.reset(0)
    outputs = []
    class Spy(Teleop):
        def update(self, frame):
            out = super().update(frame); outputs.append(out); return out
    run_session(FakeCamera(clock), FakeTracker(), Spy(), env, secs=0.5, clock=clock, engaged=lambda: True,
                close_gripper=lambda: True)
    assert outputs[-1].joint_targets[5] == pytest.approx(-0.175)


# ---- latency: when processing is slower than the camera, frames must not queue up and age ----

class GatedCamera:
    """Delivers frame n only after the test calls release(); frames are small arrays holding their own number."""
    def __init__(self):
        import threading
        self._gate = threading.Semaphore(0)
        self.n = 0
        self.closed = False
    def release(self, k=1):
        for _ in range(k): self._gate.release()
    def read(self):
        if not self._gate.acquire(timeout=2.0):
            return False, None
        self.n += 1
        return True, np.full((2, 2, 3), self.n, np.uint8)
    def close(self): self.closed = True


def wait_until(cond, timeout=2.0):
    import time
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.002)
    return cond()


def test_the_reader_hands_over_the_newest_frame_and_skips_the_stale_ones():
    from teleop_il.live import LatestFrameCamera
    inner = GatedCamera()
    cam = LatestFrameCamera(inner)
    inner.release(5)                                       # five frames arrive while the consumer is busy
    assert wait_until(lambda: inner.n == 5)
    ok, frame = cam.read()
    assert ok and frame[0, 0, 0] == 5                      # not frame 1
    cam.close()


def test_the_reader_waits_for_a_new_frame_instead_of_repeating_the_last_one():
    import threading
    from teleop_il.live import LatestFrameCamera
    inner = GatedCamera()
    cam = LatestFrameCamera(inner)
    inner.release(1)
    first = cam.read()[1][0, 0, 0]
    seen = []
    t = threading.Thread(target=lambda: seen.append(cam.read()[1][0, 0, 0]))
    t.start()
    t.join(timeout=0.2)
    assert t.is_alive()                                    # no new frame yet, so read() is still waiting
    inner.release(1)
    t.join(timeout=2.0)
    assert (first, seen) == (1, [2])
    cam.close()


def test_closing_the_reader_stops_its_thread_and_closes_the_camera():
    from teleop_il.live import LatestFrameCamera
    inner = GatedCamera()
    cam = LatestFrameCamera(inner)
    cam.close()
    assert inner.closed is True and not cam.thread_alive


def test_the_session_log_gets_the_landmarks_of_every_detected_frame():
    from teleop_il.diagnose import SessionLog
    clock = FakeClock()
    env = Env(); env.reset(0)
    log = SessionLog()
    run_session(FakeCamera(clock), FakeTracker(missing={2}), Teleop(), env, secs=0.5, clock=clock,
                engaged=lambda: True, log=log)
    assert not np.isnan(log.last_image).any()                # the last frame had a hand
    assert np.isnan(log.image_at(1)).all() and not np.isnan(log.image_at(0)).any()   # frame 2 (index 1) had none


def test_the_simulation_window_is_refreshed_with_a_smooth_amount_of_simulated_time_each_frame():
    """Stepping in 50 ms bursts against a 33 ms camera frame made the arm judder: some refreshes moved it, some did not."""
    clock = FakeClock()
    env = Env(); env.reset(0)
    times = []
    class Viewer:
        def is_running(self): return True
        def lock(self):
            import contextlib
            return contextlib.nullcontext()
        def sync(self): times.append(env.data.time)
    run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=2.0, clock=clock, engaged=lambda: True,
                viewer=Viewer())
    steps = np.diff(times[5:])
    assert steps.min() >= 0.025 and steps.max() <= 0.045      # about 33 ms of simulated time per refresh, never 0 or 50 ms


# ---- redesign (Q14 fallback): W/S keys for forward-back, timed attempts on fixed seeds (Q1 bar: 5 of 10 in 30 s) ----

def test_the_forward_key_offset_reaches_the_teleop_frame():
    clock = FakeClock()
    env = Env(); env.reset(0)
    frames = []
    class Spy(Teleop):
        def update(self, frame):
            frames.append(frame); return super().update(frame)
    run_session(FakeCamera(clock), FakeTracker(), Spy(), env, secs=0.3, clock=clock, engaged=lambda: True,
                forward=lambda: 0.03)
    assert frames[-1].forward_m == pytest.approx(0.03)


def cube_at(env):
    return env.cube_position()[:2].round(4).tolist()


def test_an_attempt_that_runs_out_of_time_is_recorded_as_failed_and_the_next_seed_is_loaded():
    from teleop_il.live import Attempts
    clock = FakeClock()
    env = Env(); env.reset(3)
    attempts = Attempts(seeds=(3, 4), limit_s=1.0)
    run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=1.5, clock=clock, engaged=lambda: True,
                attempts=attempts)
    assert [(r["seed"], r["outcome"]) for r in attempts.results] == [(3, "time up")]
    assert cube_at(env) == cube_at(_fresh(4))
    assert env.data.time < 0.6                                  # the simulation restarted with the new attempt


def _fresh(seed):
    e = Env(); e.reset(seed); return e


def test_a_lift_judged_by_the_simulation_ends_the_attempt_as_a_success():
    from teleop_il.live import Attempts
    class LiftedAfterHalfASecond(Env):
        def success(self): return self.data.time > 0.5
    clock = FakeClock()
    env = LiftedAfterHalfASecond(); env.reset(3)
    attempts = Attempts(seeds=(3, 4), limit_s=30.0)
    run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=0.8, clock=clock, engaged=lambda: True,
                attempts=attempts)
    assert attempts.results[0]["outcome"] == "lifted" and attempts.results[0]["success"]
    assert attempts.results[0]["seconds"] == pytest.approx(0.5, abs=0.1)


def test_a_counted_trial_stops_the_session_after_the_last_seed():
    from teleop_il.live import Attempts
    clock = FakeClock()
    env = Env(); env.reset(3)
    attempts = Attempts(seeds=(3, 4), limit_s=0.5, counted=True)
    stats = run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=10.0, clock=clock,
                        engaged=lambda: True, attempts=attempts)
    assert len(attempts.results) == 2 and attempts.done
    assert stats.seconds < 1.5


def test_restarting_an_attempt_reloads_the_same_seed_and_records_nothing():
    from teleop_il.live import Attempts
    clock = FakeClock()
    env = Env(); env.reset(3)
    attempts = Attempts(seeds=(3, 4), limit_s=30.0)
    def preview(image, result, out, fps):
        if 0.5 < clock.now < 0.55:
            attempts.restart()
    run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=0.8, clock=clock, engaged=lambda: True,
                attempts=attempts, preview=preview)
    assert attempts.results == [] and attempts.seed == 3
    assert env.data.time < 0.4


def test_skipping_an_attempt_records_it_as_failed():
    from teleop_il.live import Attempts
    clock = FakeClock()
    env = Env(); env.reset(3)
    attempts = Attempts(seeds=(3, 4), limit_s=30.0)
    def preview(image, result, out, fps):
        if 0.5 < clock.now < 0.55:
            attempts.skip()
    run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=0.8, clock=clock, engaged=lambda: True,
                attempts=attempts, preview=preview)
    assert [(r["seed"], r["outcome"], r["success"]) for r in attempts.results] == [(3, "skipped", False)]


# ---- keys: Khoi's try4 log had 0 W/S/G presses in 6 attempts while SPACE worked. Cause found on his desktop: an
# ---- input method (ibus Unikey, Telex) sits between the keyboard and Qt windows, and Telex uses letters for composing.

def key_state():
    from teleop_il.live import Attempts
    return {"engaged": False, "key_closed": False, "forward": 0.0, "last_key": ""}, Attempts(seeds=(1, 2))


@pytest.mark.parametrize("letter", ["g", "G"])
def test_the_gripper_key_works_in_lower_and_upper_case(letter):
    from teleop_il.live import apply_key
    state, attempts = key_state()
    apply_key(ord(letter), state, attempts)
    assert state["key_closed"]


def test_w_and_s_build_up_the_forward_offset_half_a_centimetre_per_press():
    from teleop_il.live import apply_key
    state, attempts = key_state()
    for k in "wwwW":
        apply_key(ord(k), state, attempts)
    apply_key(ord("s"), state, attempts)
    assert state["forward"] == pytest.approx(0.015)


def test_every_key_the_window_receives_is_shown_even_unknown_ones():
    from teleop_il.live import apply_key
    state, attempts = key_state()
    apply_key(ord("x"), state, attempts)
    assert state["last_key"] == "x (unused)"


def test_q_and_escape_quit():
    from teleop_il.live import apply_key
    state, attempts = key_state()
    assert apply_key(ord("q"), state, attempts) is False and apply_key(27, state, attempts) is False
    assert apply_key(ord(" "), state, attempts) is True and state["engaged"]


def test_the_demo_window_bypasses_the_desktop_input_method():
    from teleop_il.live import plain_letter_keys
    environ = {"QT_IM_MODULE": "ibus", "XMODIFIERS": "@im=ibus", "HOME": "/x"}
    plain_letter_keys(environ)
    assert environ["QT_IM_MODULE"] == "none" and "XMODIFIERS" not in environ and environ["HOME"] == "/x"


def test_restarting_an_attempt_opens_the_gripper_again():
    from teleop_il.live import apply_key
    state, attempts = key_state()
    apply_key(ord("g"), state, attempts)
    apply_key(ord("r"), state, attempts)
    assert not state["key_closed"]


def test_a_new_attempt_starts_with_the_gripper_open_and_the_last_result_is_reported():
    from teleop_il.live import Attempts, start_of_attempt
    state = {"key_closed": True, "printed": 0}
    attempts = Attempts(seeds=(1, 2))
    attempts.results.append({"seed": 1, "outcome": "lifted", "success": True, "seconds": 9.6, "assist_fired": 1})
    lines = start_of_attempt(state, attempts)
    assert not state["key_closed"]
    assert lines == ["attempt 1: lifted in 9.6 s (assist fired 1x)"]
    assert start_of_attempt(state, attempts) == []                       # each result reported once


def test_the_session_log_records_which_key_the_window_received():
    from teleop_il.diagnose import SessionLog
    clock = FakeClock()
    env = Env(); env.reset(0)
    log = SessionLog()
    keys = iter(["", "w", "", "s"] + [""] * 100)
    run_session(FakeCamera(clock), FakeTracker(), Teleop(), env, secs=0.3, clock=clock, engaged=lambda: True,
                log=log, last_key=lambda: next(keys))
    assert [r["key"] for r in log._rows[:4]] == ["", "w", "", "s"]
