# Handoff and progress, 2026-11-15

Read this first when resuming. Then `docs/research/DECISIONS.md` (all decisions, newest at the bottom) and the
ticket files in `docs/research/tickets/` (each closed ticket ends with its evidence).

## Where things stand
- Branch `integration`, pushed to github.com/ptchpmk0100/webcam-teleop-imitation (private). Not merged to `main`.
- `make test`: 222 passed (run with `env -u PYTHONPATH`). Expert check: 100/100. Null checks: PASS.
- **The policy was re-planned on 2026-11-15** after 0/100 on 100 clean demos. Read `PRD-v2.md`, the "Policy
  re-plan" entry in DECISIONS.md and "Board v2" in `tickets/00-board.md`. Tickets 18 to 25 are the new journey.
- Planning was done on Fable 5.2. **Execution (ticket 18 onward) runs on Opus 5.5.**

| Ticket | Status | Key result |
|---|---|---|
| 01-04, 06-11, 13, 15 | done | teleop, dataset, expert 100/100, 100 clean demos in `runs/demos_100`, C++ physics parity |
| 05 Week-1 video + scale check | **open, Khoi** | 60 s screen recording in a practice session (`make demo`); ruler check |
| 12 Train + score state policy | closed, superseded | official 0/100; best variant (relative input + delta targets) 5/100 |
| 14 C++ ONNX policy parity | done for contract v1 | v2 is ticket 24 |
| 18 Diagnose current failures | **next, AFK** | development seeds 20000-20099, failure breakdown, hybrid scores |
| 19 Controller + label gate | **next, AFK** | replay of converted demos must lift at least 95/100 |
| 20 New design clones the expert | after 19, AFK | at least 90/100 on development seeds |
| 21 Train on 100 human demos | after 18, 20, HITL | checkpoint 20/100 held-out |
| 22 Transformed + recovery demos | after 21, HITL | checkpoint 50/100 held-out |
| 23 Correction rounds | only if 22 misses, HITL | up to three rounds with Khoi |
| 24 C++ controller + ONNX v2 | after 20, AFK | outcomes agree 100/100 |
| 25 Final scores | after 22/23, 24, HITL | both numbers, three training seeds |
| 16 Image policy | after 25 | needs Colab or Kaggle |
| 17 CI, README, release | after 05, 25 | making the repo public needs a separate ask |

## Next steps
Claude (on Opus 5.5): start tickets 18 and 19 (no blockers, can run in parallel). Held-out seeds 10000-10099 are
scored only at the checkpoints in 21, 22, 23 and 25; everything else uses development seeds.
Khoi: ticket 05 when convenient (never screen-record during a record session). No recording is needed unless
ticket 23 is reached.

Experiment scripts from the 0/100 investigation (cleaning, relabelling, lever screen, failure probe) are copied to
`runs/scratch_scripts_20261115/` (local, not committed). Ticket 19 turns the cleaning into a repo command.

## Decisions to keep in force (details in DECISIONS.md)
- Headline score: the policy alone (no assist, no human). Policy with assist only as a labelled secondary number.
  The assist is a data-collection crutch; the end goal is without it.
- Demos keep only lifts without tracking loss over 0.3 s. Demo seeds stay below 10000 (held-out seeds 10000-10099).
- Standing rules: no `rm` or overwrite-in-place, each stage writes a new folder; planning files only in
  `docs/research/`; git create/commit/push only in this repo; making the repo public, issues, paid compute, any
  outward action need a fresh ask; claims labelled verified (command, exit N) or claim (source).

## Data that exists only on this machine (runs/ is not committed)
- Human demos: `runs/demos_20261115_004601` (9), `..._005408` (4), `..._005620` (9), `..._010151` (3), merged in
  `runs/demos_25`. `..._004332` and `..._011905` are empty sessions (0 kept). Session logs `runs/try1.npz`-`try10.npz`.
  **Back up `runs/` (for example to a USB drive or cloud folder): losing it loses the demos.**
- Policies: `runs/state_policy_20261114_235944` (expert, 59/100), `runs/state_policy_20261115_110826` (100 human, 0/100).
- Sessions `runs/demos_20261115_095943` (25), `..._104242` (9), `..._104619` (53); cleaned per session in `runs/clean_20261115_*`; merged in **`runs/demos_100`** (100 episodes, the training source).
- Assets fetched by `make setup` into `assets/` (menagerie, hand model, ONNX Runtime 1.32.0, all pinned).

## Commands
`make test` | `make demo` (practice) | `make demo ARGS="--record --log runs/tryN.npz"` | `make trial` (10 counted) |
`make diagnose LOG=...` | `make parity` (C++ physics) | `make policy-parity` (C++ ONNX policy) |
`uv run python -m teleop_il.dataset check|merge ...` | `uv run python -m teleop_il.train_state --data DIR` |
`uv run python -m teleop_il.train_image --data DIR --images DIR` | `uv run python -m teleop_il.replay DATA OUT`.

## Known noise and quirks
- Qt prints "Cannot find font directory" warnings at start: harmless.
- The desktop's Vietnamese input method is bypassed for the demo window; the panel's "last key" shows what arrived.
- If the webcam will not open, another app holds it (once it was Microsoft Edge): close it.
- Rendering one state twice differs by up to 1 colour level (GPU noise); image tests allow 2.
