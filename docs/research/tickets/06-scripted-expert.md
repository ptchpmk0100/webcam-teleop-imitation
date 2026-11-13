# 06: Scripted expert and proven spawn rectangle
What to build: a scripted pick-and-lift policy using IK, run through the eval harness, that proves which cube-spawn rectangle is truly graspable and liftable (real contacts), and writes expert episodes once the dataset module exists.
Blocked by: 01, 02
Kind: AFK
- [ ] Eval command exits nonzero unless the expert reaches at least 90 of 100 held-out seeds.
- [ ] The proven rectangle is printed and recorded; if the SO-101 gripper cannot lift the cube, the ticket parks and Khoi decides (UR5e fallback).
