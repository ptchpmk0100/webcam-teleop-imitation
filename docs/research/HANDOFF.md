# Handoff and progress, 2026-11-15

Read this first when resuming. Then `docs/research/DECISIONS.md` (all decisions, newest at the bottom) and the
ticket files in `docs/research/tickets/` (each closed ticket ends with its evidence).

## Where things stand
- Branch `integration`, pushed to github.com/ptchpmk0100/webcam-teleop-imitation (private). Not merged to `main`.
- `make test`: 219 passed. Expert check `python -m teleop_il.expert`: 100/100. Null checks: PASS.
- **Progress: 12 of 17 tickets done (71 percent by count; about 65 percent by effort**, because the two remaining
  training tickets carry the main risk: the policy alone must learn the grasp from human demos).

| Ticket | Status | Key result |
|---|---|---|
| 01 Env tracer | done | success read from MuJoCo state (cube bottom above 5 cm for 1 s) |
| 02 Eval harness | done | 100 held-out seeds (10000-10099), Wilson CI, null checks 0/100 |
| 03 Teleop core | done | relative mapping, One-Euro filter, IK with mink |
| 04 Live webcam demo | done (G5) | redesigned teleop: hand = left-right/up-down, W/S = forward-back, G = gripper, grasp assist; trial 6/10 |
| 05 Week-1 video + scale check | **open, Khoi** | 60 s screen recording; ruler scale check (left-right, up-down) |
| 06 Scripted expert | done | 100/100 held-out |
| 07 Dataset module | done | npz per episode, versioned manifest, split by cube seed, merge and check commands |
| 08 Replay + image render | done | images from stored states, 96 x 128 policy camera |
| 09 State policy code | done | MLP, 10-step action chunks, ONNX; 80 expert demos give 59/100 |
| 10 First 25 demos | done (G5) | `runs/demos_25`: 25 valid; baseline 30/30 lifted, median 8.0 s |
| 11 Record to 100 | **in progress, Khoi** | 25 of 100; checkpoint at 50 |
| 12 Train + score state policy | waiting for 11 | 25-demo checkpoint: 0/100 (approaches, rarely closes) |
| 13 C++ physics parity | done | bit-identical to Python on 130 sequences |
| 14 C++ ONNX policy parity | done | C++ = PyTorch within 6e-7, 100/100 outcomes agree, C++ null checks 0/100 |
| 15 Image policy code | done | CNN + spatial softmax, ONNX, `train_image` command |
| 16 Train + score image policy | waiting for 11 | needs Colab or Kaggle (CPU is about 10 s per epoch per 20 episodes) |
| 17 CI, README, release | waiting for 05, 12 | making the repo public needs a separate ask |

## Next steps
Khoi:
1. Record demos (ticket 11), one session at a time; the log number goes up each time:
   `make demo ARGS="--record --log runs/try11.npz"`. Each session gets a new folder `runs/demos_<time>/` and continues
   after the highest seed recorded so far (next: 628). Tips that worked: W/S for cubes far forward or back; after the
   assist hands back the gripper stays closed, raise the hand until the panel shows the 1 s hold; keep the hand in the
   lamp light (5 of 30 attempts were discarded for losing the hand more than 0.3 s).
2. Ticket 05 when convenient: Ctrl+Alt+Shift+R records the screen; ruler check with `--log`.

Claude, when Khoi reports:
1. At 50 kept demos: merge all sessions into a new folder
   (`uv run python -m teleop_il.dataset merge runs/demos_50 runs/demos_2026...`; never include `runs/demos_25`, it
   is itself a merge), train (`uv run python -m teleop_il.train_state --data runs/demos_50`), report the policy-alone
   score and what it does on failures (approach distance, gripper closing, lift).
2. At 100: same as ticket 12, plus the secondary "policy with assist" number (decided: reported, labelled, not the
   headline). If the policy still does not close the gripper, candidates for ticket 12: weight the grasp segment in
   the loss, add the gripper's commanded state or a phase cue to the observation, more epochs, DART-style noise.
3. Ticket 16 needs a Colab or Kaggle bundle (Khoi runs it by hand; no tokens to Claude).

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
- Policies: `runs/state_policy_20261114_235944` (expert, 59/100), `runs/state_policy_20261115_010232` (25 human, 0/100).
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
