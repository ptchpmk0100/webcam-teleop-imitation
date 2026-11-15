# Board (journey for PRD.md), 2026-11-14

Tickets close when done and are deleted or closed, not kept as documentation. QA may add tickets and blockers at any time.

## Graph (blocked by)
01 -> 02, 03, 07
01 + 02 -> 06, 13
03 -> 04 -> 05 (week-1 critical path)
06 + 07 -> 08, 09
04 + 06 + 07 -> 10 -> 11
08 + 09 -> 15
09 + 13 -> 14
02 + 09 + 11 -> 12
02 + 11 + 15 -> 16
05 + 12 + 14 -> 17

Critical path to the week-1 video: 01, 03, 04, 05. In parallel after 01: 02, 03, 07; after 02: 06, 13.

## Frozen thresholds (change only through a short G3)
- Expert policy (06): at least 90 of 100 held-out seeds succeed. Null policy (do nothing) 0 of 100; random-action policy at most 2 of 100.
- Teleop (03): assertions on properties read through forward kinematics (not snapshots of the code's own output); end-effector error at most 5 mm inside the proven rectangle.
- ONNX parity (09, 15): max absolute difference at most 1e-4 against PyTorch on recorded observations.
- C++ parity (13, 14): same success outcome on at least 98 of 100 seeds as Python; physics-only parity: cube height within 5 mm after the scripted sequence.
- Tiny overfit check (09, 15): training loss falls below 10 percent of its starting value on a 5-episode set.
- Spawn rectangle: forward 0.17 to 0.30 m, sideways -0.15 to +0.15 m. Started at 0.10 to 0.30; ticket 06 tightened the near edge to what the expert proves (all 17 failures of the first run were closer than 0.165 m).
- Camera (04): 25 FPS or better in the recording mode, measured and printed.

## Run budget
- AFK ticket: at most 3 failed verifier cycles, then park it and report. One ticket per fresh context. Usage gauge is unknown (autopilot ticket 02), so G9 is not enforced.
- No paid compute. No Colab, Kaggle, public repo or issue creation without a fresh ask.

# Board v2 (journey for PRD-v2.md), 2026-11-15

Done and unchanged: 01-11, 13, 15. Open with Khoi: 05. Closed as superseded: 12. Done for contract v1 only: 14.

## Graph (blocked by)
(none) -> 18, 19
19 -> 20
18 + 20 -> 21 -> 22 -> 23 (only if 22 misses 50 of 100)
20 -> 24
22 (or 23) + 24 -> 25
25 -> 16
05 + 25 -> 17

Critical path: 19, 20, 21, 22, 25. In parallel: 18 with 19 and 20; 24 with 21 and 22.
Kinds: AFK 18, 19, 20, 24. HITL 21, 22, 23, 25, 16, 17.

## Frozen thresholds v2 (change only through a short G3)
- Label gate (19): replay of converted human demos lifts at least 95 of 100.
- Design gate (20): at least 90 of 100 on development seeds from 80 expert demos.
- Held-out checkpoints: 21 at least 20 of 100; 22 at least 50 of 100. Target 50, stretch 80.
- C++ parity v2 (24): same outcome on 100 of 100 held-out seeds.
- Seeds: demos and copies below 10000; held-out 10000-10099 only at checkpoints (21, 22, 23, 25); development
  20000-20099 for everything else.
- Stop rule: under 50 of 100 after three correction rounds: stop and ask.
- Run budget and standing rules as in the first board.
