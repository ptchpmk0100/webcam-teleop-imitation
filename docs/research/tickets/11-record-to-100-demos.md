# 11: Record to 100 demos
What to build: continue recording to about 100 valid episodes with a checkpoint at 50.
Blocked by: 10
Kind: HITL
- [ ] 100 valid episodes pass dataset validation; the split by episode is created.

Progress 2026-11-15: 25 of 100 (`runs/demos_25`, ticket 10). Next session starts at seed 628. Checkpoint at 50: merge, train the state policy, report the policy-alone score.

Checkpoint 50, 2026-11-15 (session try11, `runs/demos_20261115_095943`: 26 of 27 lifted, 25 kept, median 7.0 s;
19.6 FPS, below the 25 FPS recording check, because a screen recording ran at the same time).
- Merged: `runs/demos_50` = the four ticket-10 sessions plus try11; valid, 50 episodes, 50 distinct seeds
  (verified, `python -m teleop_il.dataset merge ...`, exit 0).
- State policy on it (`train_state --data runs/demos_50`, 40 train / 10 validation): 0/100 held-out, CI 0.00-0.04
  (verified, exit 0; `runs/state_policy_20261115_100425`). Failure probe on 20 held-out seeds: closest approach to
  the grasp point median 5.5 cm (3 of 20 within 2 cm), gripper close commanded in 6 of 20, never within 3 cm of the
  grasp point; the cube rose more than 1 cm once. The 25-demo policy: median 4.1 cm, 8 of 20 closes, same picture.
- Recording bug found and fixed: at a new attempt the G toggle was opened by the window callback one camera frame
  after the new cube was loaded, so that frame still sent "closed". Effect: 34 of 50 episodes have a closed-gripper
  label on row 0 (row 1 is open in all 50), and in 12 the new cube lay within 5 cm (sideways) of the start pose, so
  the assist fired at row 0 and grasped with no human approach (the 3 s lifts). Fix: `run_session` calls
  `on_new_attempt` right after the reset, before the next frame; test
  `test_a_new_attempt_opens_the_gripper_before_its_first_recorded_step` (near and far cube), mutant without the call
  fails both cases. Suite 221 passed.
- Does the bug explain the 0/100? No: trained on all 50, on the 38 without auto-grasp, and on those 38 without row
  0, each scores 0/100 on the 100 held-out seeds (in-memory experiment, scratchpad `clean_experiment.py`). The
  failure is the approach: the policy does not get the gripper within the grasp zone precisely enough.
- Counting decision (Claude's recommendation): the 12 auto-grasp episodes stay on disk but are excluded from the
  headline dataset, so the clean count is 38; 62 more to reach 100.

Second recording bug found and fixed, 2026-11-15: a camera frame changed the joint target in the middle of a 20 Hz
control step while the recorded row kept the target from the step's start, so human demos did not replay (open-loop
replay of 34 demos: 28 lifted, joints drifted up to 0.35 rad; one-step error median 0.013 rad, up to 0.1; the
scripted expert's one-step error is 0.00000). Fix: `run_session` holds one teleop output for a whole control step,
as `Env.step` does. Test `test_replaying_the_recorded_joint_targets_reproduces_the_recorded_motion_exactly` (a hand
that moves every frame; replay must match every row bit for bit) was red at row 1 before the fix. Suite 222 passed.
Sessions started before this fix (try11, and try12 if it was already running) have the mismatch; they can be
relabelled (ticket 12 notes).

Done 2026-11-15. Sessions try12 (`runs/demos_20261115_104242`, 9 kept of 11) and try13 (`..._104619`, 53 kept of 58,
baseline 56 of 58 lifted, median 6.9 s); both started before the second fix reached disk, both free of the stale
toggle (0 auto-grasps, 0 stale rows). Both ran at 18-21 FPS (recording check FAIL) while lever experiments shared
the CPU. Cleaning stage per session (scratchpad `clean_stage.py`: drop auto-grasp episodes, drop stale row 0,
relabel by per-step inverse dynamics; residual median below 0.0005 rad, p99 0.07-0.09) into `runs/clean_*`, merged
into `runs/demos_100`: valid, 100 episodes, 100 distinct seeds, 16075 steps (verified, dataset merge, exit 0).
Split 80/20 by seed is made by `train_state`.
