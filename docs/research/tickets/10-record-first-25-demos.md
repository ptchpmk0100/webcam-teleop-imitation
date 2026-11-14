# 10: Record first 25 human demos and teleop baseline
What to build: a record mode on the live demo that saves one episode per attempt, keeps only successes, discards episodes with tracking loss, and logs Khoi's success rate and time per attempt.
Blocked by: 04, 06, 07
Kind: HITL
- [ ] 25 valid episodes pass dataset validation.
- [ ] Khoi's teleop success rate and median time are printed.

Prep 2026-11-15 (from try5 attempt 1, carried from ticket 04): the assist fired 3 times and grasped each time (aligned within 1 cm, descended to grasp height, closed), but (1) G is a toggle and was open at hand-back twice, so the gripper opened at once and dropped the cube; (2) after the third grasp the cube was lifted only to 3.9 cm (needs 5 for 1 s), then lowered. Fixed: after a hand-back the gripper stays closed until the G key changes (one press opens it; the window shows the key as closed meanwhile); the panel shows the cube's height and the hold time against the 5 cm and 1 s rule. Tests first; 3 of 3 mutations caught.
