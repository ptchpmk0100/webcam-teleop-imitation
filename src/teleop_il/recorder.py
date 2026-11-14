"""Record mode (ticket 10): turn live attempts into dataset episodes.

The live loop advances physics in 5 ms slices on camera timing. A row is taken at every 20 Hz control-step boundary
(every 10 physics steps), before that step's physics: simulated time, full positions and velocities, the joint
targets in effect at that moment (the latest teleop output), the cube pose and the tracking-lost and assist-active
flags. The targets can change inside a 50 ms step when a new camera frame arrives, so human episodes are replayed
from their stored states, not re-simulated (ticket 08).

Kept: attempts that lifted the cube (judged by the simulation) without the hand lost for too long. Failed, timed-out,
restarted and unfinished attempts are not saved. One new dataset folder per session.
"""
from __future__ import annotations

import numpy as np

from teleop_il.dataset import DatasetWriter, Episode


class EpisodeRecorder:
    def __init__(self, folder, source: str) -> None:
        self._writer = DatasetWriter(folder)
        self._source = source
        self.counts = {"kept": 0, "failed": 0, "tracking lost": 0}
        self._rows = None

    def start(self, env, seed: int) -> None:
        """Begin a new attempt right after env.reset(seed); drops whatever was being recorded."""
        self._seed, self._initial, self._rows = int(seed), env.snapshot(), []

    def row(self, env, out) -> None:
        """Call before every physics step; records only at 20 Hz control-step boundaries."""
        if self._rows is None or not env.at_control_boundary:
            return
        self._rows.append((float(env.data.time), env.data.qpos.copy(), env.data.qvel.copy(),
                           np.asarray(out.joint_targets, float).copy(), env.cube_pose(),
                           bool(out.tracking_lost), out.info.get("assist") == "active"))

    def finish(self, result: dict, flagged: bool) -> bool:
        """End the attempt with its result (from Attempts); saves it if it lifted without tracking loss."""
        rows, self._rows = self._rows, None
        if not rows:
            return False
        lost = flagged or any(r[5] for r in rows)
        if not result["success"]:
            self.counts["failed"] += 1
            return False
        if lost:
            self.counts["tracking lost"] += 1
            return False
        t, qpos, qvel, targets, cube, tracking, assist = (np.array(c) for c in zip(*rows))
        self._writer.add(Episode(seed=self._seed, source=self._source, success=True, flagged=False,
                                 assist_fired=int(result.get("assist_fired", 0)), initial_state=self._initial,
                                 time=t, qpos=qpos, qvel=qvel, joint_targets=targets, cube_pose=cube,
                                 tracking_lost=tracking.astype(bool), assist_active=assist.astype(bool)))
        self.counts["kept"] += 1
        return True

    def close(self):
        """Write the dataset manifest (unfinished attempts are dropped)."""
        self._rows = None
        return self._writer.close()
