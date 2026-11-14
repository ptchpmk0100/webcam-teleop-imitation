# 06: Scripted expert and proven spawn rectangle
What to build: a scripted pick-and-lift policy using IK, run through the eval harness, that proves which cube-spawn rectangle is truly graspable and liftable (real contacts), and writes expert episodes once the dataset module exists.
Blocked by: 01, 02
Kind: AFK
- [ ] Eval command exits nonzero unless the expert reaches at least 90 of 100 held-out seeds.
- [ ] The proven rectangle is printed and recorded; if the SO-101 gripper cannot lift the cube, the ticket parks and Khoi decides (UR5e fallback).

Status 2026-11-14: built (verifier red first, then green). Check command: `env -u PYTHONPATH uv run python -m teleop_il.expert` (exit 0 only at 90 of 100 held-out seeds, judged from MuJoCo cube height).
- First run on the original rectangle (forward 0.10 to 0.30 m): 83/100, exit 1. All 17 failures had the cube closer than 0.165 m to the base (vertical grasp not reachable there).
- Spawn rectangle tightened to forward 0.17 to 0.30 m, sideways -0.15 to 0.15 m (env constant `SPAWN_X`). Run on it: 100/100 (95 percent interval 0.96 to 1.00), exit 0.
- The SO-101 gripper CAN lift the 4 cm cube (probe: 8 of 9 seeds lifted it 10 cm), so the UR5e fallback is not needed.
- Finding for teleop: the gripper is asymmetric. The fixed jaw sits about 2 cm to one side of the gripper frame, so aiming the frame at the cube centre lands that jaw on top of the cube. The expert aims 1 cm off along the closing axis. A human aiming by eye hits the same problem; a tool-point offset in teleop is a candidate fix.
- Mutation checks on the expert (copy in scratchpad): no jaw offset 54/100 (exit 1); more mutations listed in the commit.
- Not done yet: writing expert episodes (needs the dataset module, ticket 07).
