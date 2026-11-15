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

## Held-out checkpoint 2026-11-15 (Khoi: all 100 demos; run now)
Settings fixed before the run (all from the development seeds): `runs/v2_policy_20261115` all 100 cleaned demos (80
train, 20 validation, split by seed), chunk 20, ensemble m 0.01, 200 epochs, hidden 256. All three training seeds were
scored and are all reported; none was chosen.
`python -m teleop_il.train_v2 --data runs/v2_human_20261115 --seed S --held-out --required 0.2`, exit 0 each:

| training seed | held-out lifts (policy alone) | 95% CI | null checks | run folder |
|---|---|---|---|---|
| 0 | 96/100 | 0.90-0.98 | PASS | runs/v2_policy_20261115_175750 |
| 1 | 90/100 | 0.83-0.94 | PASS | runs/v2_policy_20261115_175900 |
| 2 | 84/100 | 0.76-0.90 | PASS | runs/v2_policy_20261115_180002 |

Mean 90 of 100: above the checkpoint (20), the PRD-v2 target (50) and the stretch (80). Failure breakdown: ticket 18
(largest class: never near). This is the "100 human demos only" number for the README (policy alone, no assist).
- [x] Held-out score with interval, null checks, and the failure breakdown, shown to Khoi. Checkpoint: at least 20 of 100.
