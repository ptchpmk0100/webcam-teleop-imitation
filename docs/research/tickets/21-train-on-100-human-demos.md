# 21: Train policy v2 on the 100 human demos
What to build: train on the converted human demos, tune on development seeds only, then one scoring run on the held-out seeds. Use ticket 18's breakdown command on the result.
Blocked by: 18, 20
Kind: HITL
- [ ] Held-out score with interval, null checks, and the failure breakdown, shown to Khoi. Checkpoint: at least 20 of 100.
- [ ] This is the "100 human demos only" number for the README.

## Development-seed results 2026-11-15 (no held-out run yet)
Policy alone (no assist, no human), `train_v2`, chunk 20, development seeds 20000-20099:
- 93 demos that replay (`runs/v2_human_replayed_20261115`, 74 train): 83, 81, 85 of 100 (training seeds 0, 1, 2).
- all 100 cleaned demos (`runs/v2_human_20261115`, 80 train): 97, 84, 79 of 100.
- Validation: first-step move error about 4.5 mm (expert clone 0.9 mm), gripper command 93 percent right.
- Rollouts inspected (8 seeds): one close each, cube lifted 12-16 cm; the failure hovered and never closed.
- Demo seeds 600-723: no overlap with development or held-out seeds.
Next: Khoi picks the training set; then the one held-out checkpoint run (bar 20/100) with the failure breakdown.
