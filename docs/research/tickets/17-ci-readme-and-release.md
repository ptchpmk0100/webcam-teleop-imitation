# 17: CI, README and showable package
What to build: CI that runs the tests and the recorded-clip regression checks, a README with one command to test, one to demo, the headline number with its command, and the video.
Blocked by: 05, 12, 14
Kind: HITL
- [ ] CI is green on the pushed branch.
- [ ] Every README number cites a command.
- [ ] Making the repo public is asked separately and is not part of this ticket.

**Changed 2026-11-15 (PRD-v2):** blocked by 05, 25 (was 05, 12, 14).
- [ ] README states: the grasp segment of each demo is the assist's motion; the training set is "100 human demos plus N transformed copies"; both scores (human-only, augmented).

## Progress 2026-11-15 (the parts that need neither Khoi nor ticket 05)
- CI: `.github/workflows/tests.yml` runs `make test` on push to `main` and `integration` and on pull requests
  (Ubuntu 24.04, offscreen rendering with EGL). First run: see the handoff.
- README rewritten: result table (human only 96/90/84; with copies 100/100/100; C++ equal), each number citing
  `make final-scores`; states that the grasp segment of each demo is the assist's motion and what the training set
  is; the video section is a placeholder until Khoi's own screen recording of `make policy-demo` exists.
- Open: the video (Khoi), ticket 05, Khoi's approval of the ticket 25 numbers, and making the repo public (a
  separate ask).
