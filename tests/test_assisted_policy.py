"""Policy with assist (PRD-v2 story 16, the labelled secondary number): the policy drives; when it commands the
gripper closed near the cube, the teleop grasp assist finishes the grasp and hands back with the gripper held closed."""
import numpy as np

from teleop_il.assist import AssistedPolicy
from teleop_il.controller import GripperController
from teleop_il.evaluation import evaluate
from teleop_il.expert import grasp_point


class Sloppy:
    """Hovers 3 cm beside the grasp point and 6 cm up, commands closed there (too far off to grasp), then goes up."""

    def __init__(self, close: bool = True):
        self._c, self._close = GripperController(), close

    def reset(self):
        self._t, self.resets = 0, getattr(self, "resets", 0) + 1

    def __call__(self, obs):
        self._t += 1
        goal = grasp_point(obs["cube_pose"][:3])
        if obs["cube_pose"][2] > 0.03 or self._lifting:
            self._lifting = True
            return self._c.command(obs["joint_pos"], [goal[0], goal[1], 0.15], True)
        target = np.array([goal[0] + 0.03, goal[1], 0.06])
        return self._c.command(obs["joint_pos"], target, self._close and self._t > 40)

    _lifting = False


def test_the_assist_finishes_a_grasp_the_policy_started_badly():
    alone = evaluate(Sloppy(), seeds=[3])
    assisted = AssistedPolicy(Sloppy())
    result = evaluate(assisted, seeds=[3])
    assert alone.successes == 0 and result.successes == 1
    assert assisted.fired == 1


def test_a_policy_that_never_closes_gets_no_assist():
    assisted = AssistedPolicy(Sloppy(close=False))
    assert evaluate(assisted, seeds=[3]).successes == 0
    assert assisted.fired == 0
