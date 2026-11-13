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
- Spawn rectangle: starts at forward 0.10 to 0.30 m, sideways -0.15 to +0.15 m, tightened by 06 to what the expert proves.
- Camera (04): 25 FPS or better in the recording mode, measured and printed.

## Run budget
- AFK ticket: at most 3 failed verifier cycles, then park it and report. One ticket per fresh context. Usage gauge is unknown (autopilot ticket 02), so G9 is not enforced.
- No paid compute. No Colab, Kaggle, public repo or issue creation without a fresh ask.
