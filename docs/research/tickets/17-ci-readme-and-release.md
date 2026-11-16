# 17: CI, README and showable package
What to build: CI that runs the tests and the recorded-clip regression checks, a README with one command to test, one to demo, the headline number with its command, and the video.
Blocked by: 05, 12, 14
Kind: HITL
- [x] CI is green on the pushed branch.
- [x] Every README number cites a command (the teleop baseline is a labelled claim: it cannot be recomputed).
- [ ] Making the repo public is asked separately and is not part of this ticket.

**Changed 2026-11-15 (PRD-v2):** blocked by 05, 25 (was 05, 12, 14).
- [ ] README states: the grasp segment of each demo is the assist's motion; the training set is "100 human demos plus N transformed copies"; both scores (human-only, augmented).

## Progress 2026-11-15 (the parts that need neither Khoi nor ticket 05)
- CI: `.github/workflows/tests.yml` runs `make test` on push to `main` and `integration` and on pull requests
  (Ubuntu 24.04, offscreen rendering with EGL). Green on `integration` at 7c9c6d4: run 43754820435, 279 passed
  (verified, `gh run view`, conclusion success). Two runs failed first and found real problems: `fetch_assets.sh`
  broke on a fresh checkout (relative path after `cd`), and the v1 parity test assumed training is bit-identical
  across CPUs.
- README rewritten: result table (human only 96/90/84; with copies 100/100/100; C++ equal), each number citing
  `make final-scores`; states that the grasp segment of each demo is the assist's motion and what the training set
  is; the video section is a placeholder until Khoi's own screen recording of `make policy-demo` exists.
- Open: the video (Khoi), ticket 05, Khoi's approval of the ticket 25 numbers, and making the repo public (a
  separate ask).

## Update 2026-11-16
- Ticket 25 numbers approved by Khoi; ticket 16 done (image policy 97/100 held-out, README section).
- Video: Khoi's 66 s recording of `make policy-demo` is committed as `docs/media/policy-demo.mp4` and linked from the
  README (gitignore exception for `docs/media/*.mp4`). Making the repo public: Khoi will ask later; not done.
