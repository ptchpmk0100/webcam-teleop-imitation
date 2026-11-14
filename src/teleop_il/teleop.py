"""Teleop core (seam B): hand landmarks in, joint targets and flags out."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from teleop_il.ik import GRIPPER_CLOSED, GRIPPER_OPEN, ArmController

_WRIST, _MIDDLE_MCP, _THUMB_TIP, _INDEX_TIP = 0, 9, 4, 8
# Palm segments (landmark pairs) whose image length over camera-plane world length gives inverse distance to the camera.
_PALM_SEGMENTS = ((0, 9), (5, 17), (0, 5), (0, 17))


@dataclass(frozen=True)
class HandFrame:
    t: float                                   # seconds
    image_landmarks: np.ndarray | None         # (21, 3) MediaPipe image coordinates (x, y in 0..1, z relative); None = no hand
    world_landmarks: np.ndarray | None         # (21, 3) metres, origin at the hand centre; None = no hand
    engage: bool                               # clutch held
    close_gripper: bool = False                # gripper key held or toggled on (the pinch can be hidden from the camera)


@dataclass(frozen=True)
class TeleopOutput:
    joint_targets: np.ndarray                  # (6): 5 arm joints plus gripper
    engaged: bool
    tracking_lost: bool                        # hand missing for longer than the limit right now
    flagged: bool                              # hand was missing for longer than the limit at some point since reset
    info: dict = field(default_factory=dict)   # diagnostics for the overlay and the session log


@dataclass(frozen=True)
class TeleopConfig:
    gain_y: float = 0.8                        # metres per image width, hand towards user's right (u down) = arm -y
    gain_z: float = 0.8                        # metres per image height, hand up = arm +z
    gain_x: float = 0.5                        # metres per 100 percent growth of the hand in the image = arm +x
    image_aspect: float = 4 / 3                # image width over height (640 x 480)
    low: tuple = (0.08, -0.17, 0.015)          # workspace box the target is clamped to
    high: tuple = (0.32, 0.17, 0.25)
    ik_iterations: int = 8
    pinch_close_below_m: float = 0.03          # thumb-index gap that closes the gripper
    pinch_open_above_m: float = 0.05           # gap that opens it again (hysteresis in between)
    lost_after_s: float = 0.3                  # hand missing longer than this flags the episode
    min_cutoff_hz: float = 1.0                 # One-Euro filter on the hand position (u, v)
    beta: float = 20.0                         # raised from 5 after Khoi's G5 test: 90 percent of a quick move in 3 frames
    derivative_cutoff_hz: float = 1.0
    size_min_cutoff_hz: float = 0.8            # separate, heavier filter on the depth cue, which is the noisiest
    size_beta: float = 2.0


class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012): smooths hard when the hand is still, follows quickly when it moves."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._min_cutoff, self._beta, self._d_cutoff = min_cutoff, beta, d_cutoff
        self._x = None
        self._dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + (1.0 / (2.0 * np.pi * cutoff)) / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self._x is None:
            self._x, self._dx = x.copy(), np.zeros_like(x)
            return x.copy()
        dx = (x - self._x) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        a = self._alpha(self._min_cutoff + self._beta * np.abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


class Teleop:
    def __init__(self, config: TeleopConfig = TeleopConfig()) -> None:
        self._cfg = config
        self._arm = ArmController()
        self.reset()

    def reset(self) -> None:
        self._arm.reset()
        self._target = self._arm.home_position
        self._q = self._arm.home_joints
        self._gripper_closed = False
        self._q[5] = GRIPPER_OPEN
        self._engaged = False
        self._ref = None
        self._ee0 = None
        self._flagged = False
        self._last_seen_t = None
        self._last_t = None
        self._missing = False
        self._gap = float("nan")
        self._new_filter()

    def _new_filter(self) -> None:
        c = self._cfg
        self._filter = OneEuroFilter(c.min_cutoff_hz, c.beta, c.derivative_cutoff_hz)
        self._size_filter = OneEuroFilter(c.size_min_cutoff_hz, c.size_beta, c.derivative_cutoff_hz)

    def _features(self, frame: HandFrame) -> np.ndarray:
        """(u, v, inverse distance). The depth cue is palm image length over palm length across the image plane, so
        tilting the hand does not change it: the world landmarks say how much of each segment points at the camera."""
        img, world = frame.image_landmarks, frame.world_landmarks
        seen = perp = 0.0
        for a, b in _PALM_SEGMENTS:
            di = img[b, :2] - img[a, :2]
            seen += float(np.hypot(di[0] * self._cfg.image_aspect, di[1]))
            perp += float(np.hypot(*(world[b, :2] - world[a, :2])))
        return np.array([img[_WRIST, 0], img[_WRIST, 1], seen / max(perp, 1e-6)])

    def _smooth(self, feats: np.ndarray, dt: float) -> np.ndarray:
        return np.array([*self._filter(feats[:2], dt), *self._size_filter(feats[2:], dt)])

    def _update_gripper(self, frame: HandFrame) -> None:
        world = frame.world_landmarks
        gap = self._gap = float(np.linalg.norm(world[_THUMB_TIP] - world[_INDEX_TIP]))
        if gap < self._cfg.pinch_close_below_m:
            self._gripper_closed = True
        elif gap > self._cfg.pinch_open_above_m:
            self._gripper_closed = False
        self._q[5] = GRIPPER_CLOSED if (self._gripper_closed or frame.close_gripper) else GRIPPER_OPEN

    def update(self, frame: HandFrame) -> TeleopOutput:
        c = self._cfg
        if frame.image_landmarks is None:                       # no hand in this frame: hold everything
            self._missing = True
            if not frame.engage:
                self._engaged = False
            lost = bool(self._engaged and self._last_seen_t is not None
                        and frame.t - self._last_seen_t > c.lost_after_s)
            self._flagged = self._flagged or lost
            return TeleopOutput(self._q.copy(), self._engaged, lost, self._flagged)

        dt = max(frame.t - self._last_t, 1e-3) if self._last_t is not None else 1.0 / 30.0
        self._last_t, self._last_seen_t = frame.t, frame.t
        if self._missing:                                       # hand is back: start fresh so the arm cannot jump
            self._new_filter()
        raw = self._features(frame)
        feats = self._smooth(raw, dt)
        self._update_gripper(frame)
        if not frame.engage:
            self._engaged = False
        elif not self._engaged or self._missing:
            self._ref, self._ee0, self._engaged = feats, self._target.copy(), True
        self._missing = False
        if self._engaged:
            du, dv, ds = feats[0] - self._ref[0], feats[1] - self._ref[1], (feats[2] - self._ref[2]) / self._ref[2]
            wanted = self._ee0 + np.array([c.gain_x * ds, c.gain_y * du, -c.gain_z * dv])
            self._target = np.clip(wanted, c.low, c.high)
            gripper = self._q[5]
            self._q = self._arm.track(self._target, c.ik_iterations)
            self._q[5] = gripper
        info = {"raw": raw, "filtered": feats, "target": self._target.copy(), "pinch_gap_m": self._gap}
        return TeleopOutput(self._q.copy(), self._engaged, False, self._flagged, info)
