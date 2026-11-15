# 18: Diagnose why the current best policy fails
What to build: one command that trains the current best variant (relative input, delta targets) on the 100 demos and prints, on the 100 development seeds (20000-20099, new): lifts; a failure breakdown per episode (never near, near but no close, closed but missed, grasped but dropped or lifted too early); score by cube distance from the start pose; error by phase and joint on validation demos; and two hybrid scores (policy approach + scripted grasp, which is the "policy with assist" number, and scripted approach + policy grasp). No design change.
Blocked by: None (can start immediately)
Kind: AFK
- [ ] The command prints the table and exits non-zero if a null check fails or any breakdown row is missing.
- [ ] Development seeds are defined in the evaluation module with a test that they do not overlap demo or held-out seeds.
- [ ] The ticket ends with the table and one line naming the largest failure class.

## Closed 2026-11-15 (slimmed, Khoi's "yes for all")
Policy v1 is retired (policy v2 lifts 84-96 of 100 held-out), so the command no longer trains v1, and the hybrid
scores, per-joint errors and distance table are dropped. What remains is the failure breakdown, for any v2 run folder:
`python -m teleop_il.breakdown RUN_FOLDER [--held-out]`. Classes from MuJoCo state, first match wins: lifted; raised,
dropped or held too short; closed at the cube, not raised; near, never closed; never near. Exits 1 if the lifted count
differs from the run's own score on the same seeds, or a null check fails. Development seeds and their test were
added in ticket 21's work (`evaluation.DEV_SEEDS`). 9 tests (`tests/test_breakdown.py`).

Ticket 21's three held-out runs, verified (`python -m teleop_il.breakdown runs/v2_policy_20261115_<run> --held-out`,
exit 0 each):

| run (training seed) | lifted | raised, dropped or short | closed at cube, not raised | near, never closed | never near |
|---|---|---|---|---|---|
| 175750 (0) | 96 | 1 | 1 | 1 | 1 |
| 175900 (1) | 90 | 2 | 5 | 0 | 3 |
| 180002 (2) | 84 | 6 | 2 | 1 | 7 |
| total of 300 | 270 | 9 | 8 | 2 | 11 |

Largest failure class: never near (11 of 30 failures; seeds 10027, 10058 and 10063 fail under two or three of the
training seeds).
