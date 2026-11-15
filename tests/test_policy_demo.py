"""The policy demo's loop: a policy alone from each seed's reset, lifts read from MuJoCo state, one callback per step."""
from teleop_il.env import Env
from teleop_il.expert import ExpertPolicy
from teleop_il.ik import GRIPPER_OPEN
from teleop_il.policy_demo import AFTER_LIFT_STEPS, rollouts


def test_it_reports_a_lift_per_seed_and_shows_every_step():
    steps = []
    results = rollouts(ExpertPolicy(), Env(), [3, 4], on_step=lambda seed, t, lifts: steps.append((seed, t, lifts[-1])))
    assert results == [True, True]
    assert {s for s, _, _ in steps} == {3, 4}
    first_lift = next(t for s, t, lifted in steps if s == 3 and lifted)
    assert max(t for s, t, _ in steps if s == 3) == first_lift + AFTER_LIFT_STEPS      # a short pause, then the next


def test_a_policy_that_never_closes_is_not_counted_and_quit_stops_the_run():
    env = Env()
    home = env.reset(3)["joint_pos"].copy()
    home[5] = GRIPPER_OPEN
    assert rollouts(lambda obs: home, env, [3], max_steps=40) == [False]
    assert rollouts(lambda obs: home, env, [3, 4, 5], on_step=lambda *a: "quit") == [False]
    assert rollouts(lambda obs: home, env, [3, 4], on_step=lambda *a: "next") == [False, False]
