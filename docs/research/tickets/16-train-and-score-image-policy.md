# 16: Train and score the image policy on human demos
What to build: render images for the human dataset by replay, train the image policy (Colab if the local GPU is too small), score it over the 100 held-out seeds. If it fails, report that honestly.
Blocked by: 02, 11, 15
Kind: HITL
- [ ] Success rate with confidence interval printed; the README line says which policy (state or image).

**Changed 2026-11-15 (PRD-v2):** blocked by 25 (was 02, 11, 15). Uses the v2 design (gripper-move actions, chunk
execution, the same training set as the final state policy). Not started until the state policy passes ticket 25.
