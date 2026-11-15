# 25: Final state-policy scores
What to build: the final numbers from three training seeds: trained on 100 human demos only, and on 100 plus transformed copies (plus corrections if used); policy alone on held-out seeds, in Python and C++; policy with assist as a labelled secondary number; next to the teleop baseline and the expert.
Blocked by: 22 (or 23 if used), 24
Kind: HITL
- [ ] Every number printed by one command each, with interval and null checks. Khoi approves the numbers.

## Built 2026-11-15
- `make final-scores HUMAN="RUN RUN RUN" COPIES="RUN RUN RUN"` (`src/teleop_il/final_scores.py`): one command, all
  numbers, on the 100 held-out seeds; writes a new `runs/final_scores_<time>/`; exit 1 if a null check fails or C++
  and Python disagree on any seed.
- `assist.AssistedPolicy`: the policy drives; when it commands closed inside the assist zone, the teleop grasp assist
  finishes the grasp and hands back with the gripper held closed (tests in `tests/test_assisted_policy.py`).
- `make policy-demo RUN=...`: live window of the policy alone, for Khoi's own demonstration recording.

## Numbers (verified: the command above, exit 0, "final scores: PASS"; `runs/final_scores_20261115_200602`)
Policy v2, chunk 20, ensemble m 0.01; 80 human demos in training (20 validation), copies: + 626.

| training set | training seed | policy alone, Python (headline) | policy alone, C++ | C++ = Python | policy with assist (secondary) | run |
|---|---|---|---|---|---|---|
| 100 human demos | 0 | 96/100 (0.90-0.98) | 96/100 | 100/100 | 96/100 | v2_policy_20261115_175750 |
| 100 human demos | 1 | 90/100 (0.83-0.94) | 90/100 | 100/100 | 87/100 | v2_policy_20261115_175900 |
| 100 human demos | 2 | 84/100 (0.76-0.90) | 84/100 | 100/100 | 89/100 | v2_policy_20261115_180002 |
| 100 human demos | mean of 3 | 90.0 | 90.0 | | 90.7 | |
| 100 human demos + copies | 0 | 100/100 (0.96-1.00) | 100/100 | 100/100 | 100/100 | v2_policy_20261115_185843 |
| 100 human demos + copies | 1 | 100/100 (0.96-1.00) | 100/100 | 100/100 | 100/100 | v2_policy_20261115_190242 |
| 100 human demos + copies | 2 | 100/100 (0.96-1.00) | 100/100 | 100/100 | 100/100 | v2_policy_20261115_190715 |
| 100 human demos + copies | mean of 3 | 100.0 | 100.0 | | 100.0 | |

Scripted expert 100/100. Null checks: Python PASS; C++ do-nothing 0/100, random 0/100.
Teleop baseline (claim, counted live at record time, not recomputable because failed attempts were not saved): human
with grasp assist 30 of 30, median 8.0 s (ticket 10); 56 of 58, median 6.9 s (ticket 11, largest session).
The assist does not help the human-only policies (87-96 against 84-96): their failures are mostly in the approach
(ticket 18: "never near"), which the assist does not touch.
- [x] Every number printed by one command, with interval and null checks.
- [ ] Khoi approves the numbers.
