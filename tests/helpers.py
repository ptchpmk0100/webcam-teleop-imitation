"""Synthetic hand clips: landmark arrays built from a few numbers, so expected motion is known by construction."""
import mujoco
import numpy as np

from teleop_il.env import build_model
from teleop_il.teleop import HandFrame

FRAME_DT = 1 / 30
THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9


ASPECT = 4 / 3                      # image width over height (640 x 480)
FX = 1.0                            # focal length in image widths of the synthetic pinhole camera
# Palm in metres, camera-aligned axes like MediaPipe world landmarks (x right, y down, z away), origin at the wrist.
PALM = {0: (0.0, 0.0, 0.0), 9: (0.0, -0.09, 0.0), 5: (-0.03, -0.085, 0.0), 17: (0.03, -0.07, 0.0)}


def hand(t, u=0.5, v=0.5, size=0.1, pinch_m=0.08, engage=True, close_gripper=False):
    """A flat hand whose wrist is at image (u, v); the wrist-to-middle-knuckle length in the image is `size` (heights)."""
    img = np.zeros((21, 3)); world = np.zeros((21, 3))
    k = size / 0.09                                   # image heights per metre of palm
    for i, (x, y, z) in PALM.items():
        img[i] = (u + x * k / ASPECT, v + y * k, 0.0)
        world[i] = (x, y, z)
    world[THUMB_TIP] = (0.0, 0.0, 0.0); world[INDEX_TIP] = (pinch_m, 0.0, 0.0)
    return HandFrame(t=t, image_landmarks=img, world_landmarks=world, engage=engage, close_gripper=close_gripper)


def pinhole_hand(t, u=0.5, v=0.5, distance=0.5, pitch_deg=0.0, yaw_deg=0.0, pinch_m=0.08, engage=True):
    """A palm at `distance` metres from a pinhole camera, tilted about the wrist; landmarks come from projecting it."""
    a, b = np.radians(pitch_deg), np.radians(yaw_deg)
    pitch = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    yaw = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    origin = np.array([(u - 0.5) * distance / FX, (v - 0.5) * distance / (FX * ASPECT), distance])
    img = np.zeros((21, 3)); world = np.zeros((21, 3))
    for i, p in PALM.items():
        rel = yaw @ pitch @ np.array(p)
        cam = origin + rel
        img[i] = (0.5 + FX * cam[0] / cam[2], 0.5 + FX * ASPECT * cam[1] / cam[2], 0.0)
        world[i] = rel
    world[THUMB_TIP] = (0.0, 0.0, 0.0); world[INDEX_TIP] = (pinch_m, 0.0, 0.0)
    return HandFrame(t=t, image_landmarks=img, world_landmarks=world, engage=engage)


def no_hand(t, engage=True):
    return HandFrame(t=t, image_landmarks=None, world_landmarks=None, engage=engage)


class Clock:
    """Hands out frames at 30 Hz."""
    def __init__(self): self.n = 0
    def t(self): self.n += 1; return self.n * FRAME_DT


_model = build_model(with_cube=False)
_data = mujoco.MjData(_model)
_site = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")


def fk(joint_targets):
    """Gripper-frame position and tilt from vertical (degrees) for joint values, read from MuJoCo directly."""
    _data.qpos[:6] = joint_targets
    mujoco.mj_kinematics(_model, _data)
    approach = _data.site_xmat[_site].reshape(3, 3)[:, 0]
    return _data.site_xpos[_site].copy(), float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1))))
