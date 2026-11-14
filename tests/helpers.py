"""Synthetic hand clips: landmark arrays built from a few numbers, so expected motion is known by construction."""
import mujoco
import numpy as np

from teleop_il.env import build_model
from teleop_il.teleop import HandFrame

FRAME_DT = 1 / 30
THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9


def hand(t, u=0.5, v=0.5, size=0.1, pinch_m=0.08, engage=True):
    """A hand whose wrist is at image (u, v), wrist-to-middle-knuckle length `size`, thumb-index gap `pinch_m` metres."""
    img = np.zeros((21, 3)); world = np.zeros((21, 3))
    img[WRIST] = (u, v, 0.0); img[MIDDLE_MCP] = (u, v - size, 0.0)
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
