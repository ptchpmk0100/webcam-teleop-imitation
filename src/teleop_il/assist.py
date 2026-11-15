"""Grasp assist (shared autonomy): the human brings the gripper near the cube and closes it; the robot finishes the grasp.

Redesign grill Q2/Q13 and the Q14 fallback. In the prototypes nobody could line up the SO-101's asymmetric jaws by eye,
and a 3 cm zone that had to be hit at the instant of closing never fired (closest close: 3.2 cm). So the zone is 5 cm
and the assist fires whenever the gripper is commanded closed inside it, once per close.

Sequence: move across to above the grasp point (at least 7 cm high), descend straight down
(diagonal descents knocked the cube with the fixed jaw in 7 of 10 prototype checks), close, then hand control back.
The human lifts. Episodes record that the assist fired: the baseline is "human with grasp assist".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from teleop_il.expert import grasp_point
from teleop_il.ik import GRIPPER_CLOSED, GRIPPER_OPEN, ArmController
from teleop_il.teleop import HandFrame, Teleop, TeleopOutput


@dataclass(frozen=True)
class AssistConfig:
    radius_m: float = 0.05            # horizontal distance from the grasp point that counts as "near"
    max_start_z: float = 0.12         # and the gripper must be lower than this
    safe_z: float = 0.07              # travel height above the cube (its top is at 0.04)
    align_s: float = 0.5
    descend_s: float = 0.5
    close_s: float = 0.75
    ik_iterations: int = 8


class AssistedTeleop:
    """Same interface as Teleop (update(frame) -> TeleopOutput), with the grasp assist on top.

    `ee` and `cube` are callables returning the gripper frame's and the cube centre's real positions (simulator state).
    Extra info keys: "assist" ("off", "ready" = inside the zone, "active"), "assist_fired" (count since reset),
    "grasp_distance_m" (horizontal, gripper frame to grasp point), "holding" (after a hand-back the gripper is kept
    closed until the gripper key changes).
    """

    def __init__(self, teleop: Teleop, ee: Callable[[], np.ndarray], cube: Callable[[], np.ndarray],
                 config: AssistConfig = AssistConfig()) -> None:
        self._teleop, self._ee, self._cube, self._cfg = teleop, ee, cube, config
        self._arm = ArmController()
        self.reset()

    def reset(self) -> None:
        self._teleop.reset()
        self._active = False
        self._armed = True
        self._fired = 0
        self._last = None
        self._holding = False            # after a hand-back: gripper kept closed until the key changes
        self._key_at_handback = False

    @property
    def fired(self) -> int:
        return self._fired

    def _start(self, t: float, ee: np.ndarray, joints: np.ndarray, cube: np.ndarray) -> None:
        c = self._cfg
        goal = grasp_point(cube)
        above = np.array([goal[0], goal[1], max(float(ee[2]), c.safe_z)])
        # No separate "rise first" leg: from 2-3 cm high beside the cube, the diagonal move up and across lifted 6 of
        # 6 in a probe, the same as with a rise, so it was dropped as untested insurance.
        self._legs = [(ee.copy(), above, c.align_s), (above, goal, c.descend_s)]
        self._t0 = t
        self._arm.reset(joints)
        self._active, self._armed = True, False
        self._fired += 1

    def _command(self, t: float) -> np.ndarray:
        s = t - self._t0
        target, gripper = self._legs[-1][1], GRIPPER_CLOSED
        for start, end, duration in self._legs:
            if s < duration:
                target, gripper = start + (s / duration) * (end - start), GRIPPER_OPEN
                break
            s -= duration
        q = self._arm.track(target, self._cfg.ik_iterations)
        q[5] = gripper
        if s >= self._cfg.close_s and gripper == GRIPPER_CLOSED:
            self._active = False                             # done: the next frame goes back to the human
        return q

    def update(self, frame: HandFrame) -> TeleopOutput:
        ee, cube = np.asarray(self._ee(), float), np.asarray(self._cube(), float)
        distance = float(np.hypot(*(ee[:2] - grasp_point(cube)[:2])))
        near = distance < self._cfg.radius_m and ee[2] < self._cfg.max_start_z
        if self._active:
            q = self._command(frame.t)
            if not self._active:
                self._teleop.resume_from(q)
                # try5 attempt 1: G is a toggle and was open at hand-back twice, so the gripper opened at once and
                # dropped a cube the assist had just grasped. Keep it closed until the key changes (one press).
                self._holding, self._key_at_handback = True, frame.close_gripper
            out = self._last
            return TeleopOutput(q, frame.engage, False, out.flagged if out else False,
                                {**(out.info if out else {}), "assist": "active", "assist_fired": self._fired,
                                 "grasp_distance_m": distance, "holding": False})
        out = self._teleop.update(frame)
        if self._holding and frame.close_gripper != self._key_at_handback:
            self._holding = False
        if self._holding:
            q = out.joint_targets.copy()
            q[5] = GRIPPER_CLOSED
            out = TeleopOutput(q, out.engaged, out.tracking_lost, out.flagged, out.info)
        closed = out.joint_targets[5] == GRIPPER_CLOSED
        if not closed:
            self._armed = True
        elif self._armed and near:
            self._start(frame.t, ee, out.joint_targets, cube)
            self._last = out
            return self.update(frame)
        self._last = out
        info = {**out.info, "assist": "ready" if near else "off", "assist_fired": self._fired,
                "grasp_distance_m": distance, "holding": self._holding}
        return TeleopOutput(out.joint_targets, out.engaged, out.tracking_lost, out.flagged, info)


class _PolicyAsTeleop:
    """A harness policy (obs -> 6 joint targets) behind the Teleop interface the assist wraps."""

    def __init__(self, policy) -> None:
        self._policy, self.obs = policy, None

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def update(self, frame: HandFrame) -> TeleopOutput:
        return TeleopOutput(np.asarray(self._policy(self.obs), float), True, False, False, {})

    def resume_from(self, joints) -> None:
        self.reset()                     # the policy's plans from before the assist are stale: start it afresh


class AssistedPolicy:
    """Policy with assist (PRD-v2 story 16, a labelled secondary number, never the headline): `policy` drives the arm;
    when it commands the gripper closed inside the assist zone, the grasp assist takes over as it does for the human
    (move above the grasp point, descend, close), then the policy continues with the gripper held closed. Reads only
    the observation: the gripper position is forward kinematics of the joints, the cube pose is the observed one."""

    def __init__(self, policy, config: AssistConfig = AssistConfig()) -> None:
        from teleop_il.controller import GripperController
        fk = GripperController()
        self._inner = _PolicyAsTeleop(policy)
        self._assisted = AssistedTeleop(self._inner, ee=lambda: fk.gripper_position(self._inner.obs["joint_pos"]),
                                        cube=lambda: self._inner.obs["cube_pose"][:3], config=config)

    @property
    def fired(self) -> int:
        """How often the assist fired since the last reset."""
        return self._assisted.fired

    def reset(self) -> None:
        self._assisted.reset()

    def __call__(self, obs) -> np.ndarray:
        self._inner.obs = obs
        return self._assisted.update(HandFrame(float(obs["time"]), None, None, True)).joint_targets
