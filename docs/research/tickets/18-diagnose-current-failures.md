# 18: Diagnose why the current best policy fails
What to build: one command that trains the current best variant (relative input, delta targets) on the 100 demos and prints, on the 100 development seeds (20000-20099, new): lifts; a failure breakdown per episode (never near, near but no close, closed but missed, grasped but dropped or lifted too early); score by cube distance from the start pose; error by phase and joint on validation demos; and two hybrid scores (policy approach + scripted grasp, which is the "policy with assist" number, and scripted approach + policy grasp). No design change.
Blocked by: None (can start immediately)
Kind: AFK
- [ ] The command prints the table and exits non-zero if a null check fails or any breakdown row is missing.
- [ ] Development seeds are defined in the evaluation module with a test that they do not overlap demo or held-out seeds.
- [ ] The ticket ends with the table and one line naming the largest failure class.
