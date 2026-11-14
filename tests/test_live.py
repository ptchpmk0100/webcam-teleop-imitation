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
