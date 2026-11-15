# 22: Transformed copies and recovery data
What to build: a generator that moves each human demo's cube-relative path to new cube positions (seeds below 10000), replays it through the controller and keeps only copies that lift; and a second mode that starts off the path and steers back onto it. Retrain on 100 human demos plus the copies; score on held-out seeds.
Blocked by: 21
Kind: HITL
- [ ] Every kept copy lifts in the simulator and records its source demo; a test checks no copy uses a held-out or development seed.
- [ ] Held-out score with interval and null checks shown to Khoi. Checkpoint: at least 50 of 100. Khoi decides whether ticket 23 is needed.
