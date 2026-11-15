# 22: Transformed copies and recovery data
What to build: a generator that moves each human demo's cube-relative path to new cube positions (seeds below 10000), replays it through the controller and keeps only copies that lift; and a second mode that starts off the path and steers back onto it. Retrain on 100 human demos plus the copies; score on held-out seeds.
Blocked by: 21
Kind: HITL
- [ ] Every kept copy lifts in the simulator and records its source demo; a test checks no copy uses a held-out or development seed.
- [ ] Held-out score with interval and null checks shown to Khoi. Checkpoint: at least 50 of 100. Khoi decides whether ticket 23 is needed.

## Scope 2026-11-15 (Khoi: "choose the best option", Claude does all, no Codex)
Transformed copies only. The recovery mode (start off the path, steer back) was not built: ticket 21 had already
passed this ticket's checkpoint, and the copies alone reach 100 of 100 below.

## Built
- `python -m teleop_il.transform DATA OUT [--copies N]` (`src/teleop_il/transform.py`): each demo's achieved gripper
  path is turned about the arm base by the change in the cube's bearing and shifted radially by the change in its
  distance, blended in over the approach; replayed through the policy-path controller from the new seed's reset;
  kept only if the cube lifts (MuJoCo state). Copy seeds run from 1000 up, below 10000, skipping demo seeds.
- `train_v2 --extra DIR`: adds only the copies of training-split demos (validation stays human-only) and refuses a
  copies folder that was not made from `--data`.
- Tests: `tests/test_transform.py` (seeds, geometry, every kept copy lifts and names its source and passes the label
  gate, training-split filter). `make test`: 276 passed (verified, exit 0).

## Copies
`python -m teleop_il.transform runs/v2_human_20261115 runs/v2_copies_20261115 --copies 10` (verified, exit 0):
798 of 1000 lifted and kept; 96 of 100 demos have at least one copy; 798 distinct seeds. In a 1-copy probe the
copies failed more for larger moves, for positions near the base, and for demos that pushed the cube (9 of 23 kept
against 65 of 77). Nothing was tuned: 10 copies per demo was the first setting tried.

## Results (policy alone; 80 human training demos + 626 copies; chunk 20, ensemble m 0.01, 200 epochs)
`python -m teleop_il.train_v2 --data runs/v2_human_20261115 --extra runs/v2_copies_20261115 --seed S [--held-out --required 0.5]`

| training seed | development (human only) | development (+ copies) | held-out (human only, ticket 21) | held-out (+ copies) | 95% CI | null checks | held-out run folder |
|---|---|---|---|---|---|---|---|
| 0 | 97/100 | 100/100 | 96/100 | 100/100 | 0.96-1.00 | PASS | runs/v2_policy_20261115_185843 |
| 1 | 84/100 | 100/100 | 90/100 | 100/100 | 0.96-1.00 | PASS | runs/v2_policy_20261115_190242 |
| 2 | 79/100 | 100/100 | 84/100 | 100/100 | 0.96-1.00 | PASS | runs/v2_policy_20261115_190715 |

All three training seeds scored and reported (verified, PASS line and exit 0 each). Development run folders:
`runs/v2_policy_20261115_183708`, `..._184208`, `..._184605`. Checkpoint (50 of 100) passed; ticket 23 is not needed.
- [x] Every kept copy lifts in the simulator and records its source demo; a test checks no copy uses a held-out or development seed.
- [x] Held-out score with interval and null checks shown to Khoi. Checkpoint: at least 50 of 100. Ticket 23 not needed (300 of 300).
