# PRD v2: a policy that learns the pick-and-lift from Khoi's 100 demos

Status: approved by Khoi 2026-11-15 (grill rounds 1 and 2, see DECISIONS.md "Policy re-plan"). Replaces the policy
part of PRD.md (user stories 21 to 24 and the state/image training tickets). Everything else in PRD.md stands. Closed
or deleted when shipped, not kept as documentation.

## Problem statement

The teleop, dataset, evaluation and C++ parts work: Khoi lifts the cube in 97 percent of attempts and 100 clean
demos exist. The policy trained on them lifts the cube on 0 of 100 held-out positions. Khoi does not accept that: a
policy that never lifts has learned nothing, and the project is then not imitation learning.

Measured causes (all on this machine, 2026-11-15, details in ticket 12):
- The policy memorises: error 0.006 rad on training demos, 0.107 rad on validation demos (about 3 cm at the gripper;
  the grasp needs about 1 cm).
- The labels do not reproduce the motion: open-loop replay of the relabelled actions lifts 38 of 50 demos.
- The representation matters more than the data volume: 38 to 100 demos changed nothing; relative input plus delta
  targets took "gripper within 2 cm of the cube" from 3 to 24 of 100 and gave the first lifts (5 of 100).
- The design is weak even on perfect data: 59 of 100 when cloning 80 scripted-expert demos.
- Not yet separated: compounding error with no recovery data, human pauses and arbitrary key timing, and why 22
  closes near the cube give only 5 lifts.

## Solution

Keep the 100 demos. Change what the policy sees, what it outputs, how its output is executed and what it is trained
on, and never train on anything that has not passed a gate first.

1. Diagnose the current failures with measurements (no training).
2. Policy in gripper coordinates relative to the cube: it outputs small gripper moves and an open/close decision; a
   small controller turns gripper moves into joint targets, identically in Python and C++.
3. Labels are the motion the arm achieved, not the recorded teleop target.
4. Whole action chunks are executed, with overlapping predictions blended (ACT-style temporal ensembling).
5. The 100 demos are multiplied by moving each demo's cube-relative path to new cube positions and keeping only
   copies that lift in the simulator (MimicGen-style), plus recovery data that starts off the path and steers back.
6. If that misses the target, Khoi gives up to three rounds of corrections while the policy drives (HG-DAgger-style).

Target: the policy alone lifts on at least 50 of 100 held-out seeds (stretch 80), reported with its confidence
interval, next to the score from the 100 human demos without transformed copies.

## User stories

1. As Khoi, I want a failure breakdown of the current best policy (approach, close, grasp, lift, by cube distance), so
   that the next steps are chosen from measurements.
2. As Khoi, I want 100 development seeds separate from the 100 held-out seeds, so that choosing between variants does
   not flatter the headline number.
3. As Khoi, I want the held-out seeds scored only at named checkpoints, so that the headline stays honest.
4. As Khoi, I want the policy to see the gripper relative to the cube's grasp point, so that "move toward the cube"
   looks the same wherever the cube is.
5. As Khoi, I want the policy to output small gripper moves, so that it does not have to learn the arm's geometry
   from 100 demos.
6. As Khoi, I want one controller that turns a gripper move into joint targets in both Python and C++, so that the
   deployed policy behaves the same in both.
7. As Khoi, I want demo actions taken from the motion the arm achieved, so that labels are consistent with what
   happened.
8. As Khoi, I want any action format to replay at least 95 of the 100 demos before a policy is trained on it, so that
   bad labels are caught before training.
9. As Khoi, I want the new design to clone the scripted expert at 90 of 100 or better first, so that a weak design is
   caught on perfect data.
10. As Khoi, I want action chunks executed with blending, so that my pauses and small errors do not stall or derail
    the arm.
11. As Khoi, I want the gripper command as its own yes/no output with a hold rule, so that the gripper cannot flutter.
12. As Khoi, I want my demos transformed to new cube positions and verified in the simulator, so that the policy sees
    many more positions without my recording again.
13. As Khoi, I want recovery data (start off the path, steer back), so that the policy can correct its own drift.
14. As Khoi, I want transformed demos to use only seeds below 10000, so that no test position leaks into training.
15. As Khoi, I want two scores reported: trained on the 100 human demos only, and trained on 100 plus transformed
    copies, so that the effect of augmentation is visible.
16. As Khoi, I want the "policy with assist" score as a labelled secondary number, so that approach and grasp failures
    can be told apart.
17. As Khoi, I want a correction mode (policy drives, I take over, only my corrections are recorded) available as a
    fallback, so that there is a next step if augmentation misses the target.
18. As Khoi, I want work to stop and the numbers brought to me if three correction rounds still miss 50 of 100, so
    that no effort is spent without a decision.
19. As Khoi, I want the C++ rollout to run the new policy and controller with the same outcomes as Python, so that the
    deployment claim still holds.
20. As Khoi, I want the README to say that the grasp segment of each demo is the assist's motion and that the
    training set is "100 human demos plus N transformed copies", so that nothing is overstated.
21. As Khoi, I want the cleaning and conversion steps that produce the training set to be repo commands with tests,
    so that the dataset can be rebuilt from the recorded sessions.
22. As Khoi, I want the image policy attempted only after the state policy passes, so that effort goes to the
    blocking problem first.
23. As Khoi, I want every reported score to come with the null checks in the same run, so that a broken harness
    fails loudly.

## Implementation decisions

- **Observation (state policy v2):** gripper position minus the cube's grasp point (3), gripper height above the
  table (1), gripper opening (1). Cube yaw is added only if the spawn varies it and the diagnosis shows it matters.
  Cube pose still comes from simulator state.
- **Action:** per step, a gripper move (3) and a gripper open/close logit (1). Whether the move is expressed against
  the current achieved gripper position or as a target relative to the cube is decided by the label gate (the format
  that replays more demos wins).
- **Controller:** damped-least-squares on the gripper site's position Jacobian, with a soft point-down term, written
  in Python and in C++ (MuJoCo's Jacobian function). It replaces the IK library only on the policy path; teleop is
  unchanged.
- **Execution:** chunk of 10 to 20 steps, re-planned every step or every few steps, overlapping predictions blended
  with exponential weights; gripper logit thresholded with hysteresis. L1 loss on moves, binary cross-entropy on the
  gripper.
- **Network:** MLP stays the default. A different network is allowed only if the expert-clone gate fails with the MLP.
- **ONNX contract v2:** input "observation" (B, N), output "actions" (B, H, 4). The controller and the blending live
  in the rollout loop, not in the network. The v1 contract and its parity result (ticket 14) are superseded.
- **Dataset:** recorded sessions and `runs/demos_100` stay read-only. Conversion (cleaning, achieved-motion labels,
  transformed copies, recovery data) writes new folders with a manifest that names the source, the method and the
  counts. A transformed episode records its source demo and is kept only if it lifts in the simulator.
- **Seeds:** demos and transformed copies below 10000; held-out 10000 to 10099 (checkpoints only); development 20000
  to 20099 (all experiments).
- **Corrections (fallback):** a live mode where the policy drives and Khoi's input, when engaged, overrides it; only
  the overridden segments are added to the training set, tagged by round.

## Testing decisions

Good tests assert behaviour through public interfaces with expected values from an independent source (the
simulator's state, forward kinematics, a known literal).

Seams (existing ones first; Khoi may veto):
1. **Evaluation harness** (exists): policy in, lifts out of 100 seeds with interval and null checks. Catches every
   end-to-end failure; does not say which part failed.
2. **Controller**: gripper move in, joint targets out; asserted through forward kinematics in the simulator (the
   gripper moved by the commanded amount within a tolerance, and stays within joint limits). Catches geometry and
   limit errors; misses dynamics over long horizons (the replay gate covers that).
3. **Dataset conversion**: dataset folder in, dataset folder out; asserted by replaying the output in the simulator
   (lifts counted from state). Catches bad labels and bad transformed copies.
4. **ONNX and C++ rollout** (exists): same seeds, same outcomes as Python.

Prior art: tests for the evaluation harness, the dataset module, live recording replay, and the two parity commands.

## Frozen thresholds (change only through a short G3)

- Label gate: open-loop replay of the converted human demos lifts at least 95 of 100.
- Design gate: cloning 80 scripted-expert demos scores at least 90 of 100 on development seeds.
- Checkpoints on held-out seeds: at least 20 of 100 from the 100 human demos; at least 50 of 100 with transformed
  copies. Target 50, stretch 80.
- C++ against Python with the v2 policy and controller: same outcome on 100 of 100 held-out seeds.
- Null checks stay 0 of 100 (do-nothing) and at most 2 of 100 (random) in every scoring run.
- Stop rule: under 50 of 100 after three correction rounds means stop and ask.

## Out of scope

- Diffusion or mixture-output policies (lifting this needs a fresh ask, and only if the diagnosis shows averaging
  between strategies is what is left).
- A residual policy on top of a scripted primitive (mostly script, not imitation).
- Recording again, recording with noise, or recording without the assist.
- The image policy until the state policy passes its final score.
- Paid compute, ROS 2, making the repo public (separate ask).

## Further notes

- Sources, cited from memory and not re-checked: ACT (Zhao et al. 2023), robomimic study (Mandlekar et al. 2021),
  MimicGen (Mandlekar et al. 2023), HG-DAgger (Kelly et al. 2019), DART (Laskey et al. 2017), Diffusion Policy (Chi
  et al. 2023). Treat as claims.
- All scores so far come from single training seeds. Final scores (ticket 25) use three training seeds.
