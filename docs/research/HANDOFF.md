# Handoff and progress, 2026-11-15

Read this first when resuming. Then `docs/research/DECISIONS.md` (all decisions, newest at the bottom) and the
ticket files in `docs/research/tickets/` (each closed ticket ends with its evidence).

## Where things stand
- Branch `integration`, pushed to github.com/ptchpmk0100/webcam-teleop-imitation (private), CI green. Not merged to `main`.
- `make test` (run with `env -u PYTHONPATH`): see the latest commit message for the count. Expert check: 100/100.
  Null checks: PASS.
- **The policy was re-planned on 2026-11-15** after 0/100 on 100 clean demos. Read `PRD-v2.md`, the "Policy
  re-plan" and "Execution answers" entries in DECISIONS.md and "Board v2" in `tickets/00-board.md`.
- **Policy v2 works:** trained on the 100 human demos only, the policy alone lifts 96, 90 and 84 of the 100 held-out
  seeds (training seeds 0, 1, 2; ticket 21). C++ runs it with the same outcome on 100/100 (ticket 24).
- **With transformed copies (ticket 22): 100, 100, 100 of 100 held-out seeds** (80 human demos + 626 copies in
  training, seeds 0, 1, 2), the same in C++ (ticket 25).
- Planning was done on Fable 5.2. **Execution (ticket 18 onward) runs on Opus 5.5** (ticket 22 ran on Fable 5.2 as
  an exception Khoi allowed).

| Ticket | Status | Key result |
|---|---|---|
| 01-04, 06-11, 13, 15 | done | teleop, dataset, expert 100/100, 100 clean demos in `runs/demos_100`, C++ physics parity |
| 05 Week-1 video + scale check | **closed by Khoi's approval 2026-11-16** | ruler logs `runs/try14.npz`, `try15.npz`: gain matches design within noise, no cm figure (see ticket 05); live-teleop clip optional |
| 12 Train + score state policy | closed, superseded | official 0/100; best variant (relative input + delta targets) 5/100 |
| 14 C++ ONNX policy parity | done for contract v1 | v2 is ticket 24 |
| 18 Diagnose failures | **done (slimmed)** | `python -m teleop_il.breakdown RUN [--held-out]`; largest class on held-out: never near (11 of 30) |
| 19 Controller + label gate | **done** | gate judged on no-push demos (Khoi): 77/77 PASS; all demos 93/100; expert 100/100 |
| 20 New design clones the expert | **done** | 100/100 development seeds (v1: 59), chunk 20, 3 of 3 training seeds |
| 21 Train on 100 human demos | **done** | held-out, policy alone: 96, 90, 84 (seeds 0-2), mean 90; `runs/v2_human_20261115` |
| 22 Transformed copies | **done** (recovery mode dropped) | held-out, policy alone: 100, 100, 100 (seeds 0-2); 798 copies in `runs/v2_copies_20261115` |
| 23 Correction rounds | not needed | 22 passed its checkpoint |
| 24 C++ controller + ONNX v2 | **done** | `make policy-v2-parity RUN=...`: outcomes agree 100/100, controller gap 7e-15 |
| 25 Final scores | **done; Khoi approved 2026-11-16** | `make final-scores`: human only 96/90/84, with copies 100/100/100, C++ equal on every seed; with assist 96/87/89 and 100/100/100 |
| 16 Image policy | **done 2026-11-16** | image + joints, no cube pose; Kaggle T4 40 epochs; policy alone 97/100 held-out and 97/100 development (CI 0.92-0.99), seed 0 only; `python -m teleop_il.train_image_v2 score runs/image_v2_policy_kaggle_20261116 [--held-out]` |
| 17 CI, README, release | **CI green, README and video (`docs/media/policy-demo.mp4`) committed**; open: public release (Khoi asks later) | `.github/workflows/tests.yml`, run 43754820435: 279 passed; making the repo public needs a separate ask |

## Next steps
Ticket 25: the table is in `tickets/25-final-scores.md` (from `make final-scores`, `runs/final_scores_20261115_200602`).
All original tickets are closed except 17's last item: making the repo public (Khoi asks later). 16 is done
(see the table; Kaggle run in `runs/image_v2_policy_kaggle_20261116`, arrays in `runs/image_v2_arrays_copies_20261116`,
upload folder `runs/kaggle_image_v2_20261116`). A human-only image run was started locally and stopped at epoch 38
of 100 (not needed for the plan; its folder `runs/image_v2_policy_20261116_*` has no policy.pt).
Held-out seeds 10000-10099 are scored only at the checkpoints in 21, 22, 23 and 25; everything else uses
development seeds.
Khoi: ticket 05 when convenient (never screen-record during a record session). The demonstration video is Khoi's own screen recording
of the live window: `make policy-demo RUN=runs/v2_policy_20261115_185843` (policy alone, real time, keys
N / SPACE / Q). `runs/policy_copies_dev_20261115.mp4` is a rendered placeholder only. The live window was checked
off-screen only; its first run on the desktop is Khoi's.

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
- v2 data: `runs/v2_human_20261115` (100 cleaned demos from the 7 sessions, the training set),
  `runs/v2_human_replayed_20261115` (the 93 that replay). v2 held-out policies: `runs/v2_policy_20261115_175750`
  (seed 0, 96), `..._175900` (seed 1, 90), `..._180002` (seed 2, 84).
- Ticket 22: `runs/v2_copies_20261115` (798 transformed copies of the 100 demos). Policies with copies, held-out:
  `runs/v2_policy_20261115_185843` (seed 0), `..._190242` (seed 1), `..._190715` (seed 2), each 100/100; development:
  `..._183708`, `..._184208`, `..._184605`, each 100/100.
- Assets fetched by `make setup` into `assets/` (menagerie, hand model, ONNX Runtime 1.32.0, all pinned).

## Commands
`make test` | `make demo` (practice) | `make demo ARGS="--record --log runs/tryN.npz"` | `make trial` (10 counted) |
`make diagnose LOG=...` | `make parity` (C++ physics) | `make policy-parity` (C++ ONNX policy) |
`uv run python -m teleop_il.dataset check|merge ...` | `uv run python -m teleop_il.train_state --data DIR` |
`uv run python -m teleop_il.train_image --data DIR --images DIR` | `uv run python -m teleop_il.replay DATA OUT`.
Policy v2: `python -m teleop_il.convert build|gate ...` | `python -m teleop_il.train_v2 --data DIR [--seed S] [--held-out]` |
`python -m teleop_il.breakdown RUN [--held-out]` | `make policy-v2-parity RUN=runs/v2_policy_...` |
`python -m teleop_il.transform DATA OUT [--copies N]` | `train_v2 ... --extra COPIES` | `make policy-demo RUN=...` |
`make final-scores HUMAN="RUN RUN RUN" COPIES="RUN RUN RUN"` (held-out; ticket 25 checkpoint only).

## Known noise and quirks
- Qt prints "Cannot find font directory" warnings at start: harmless.
- The desktop's Vietnamese input method is bypassed for the demo window; the panel's "last key" shows what arrived.
- If the webcam will not open, another app holds it (once it was Microsoft Edge): close it.
- Rendering one state twice differs by up to 1 colour level (GPU noise); image tests allow 2.
