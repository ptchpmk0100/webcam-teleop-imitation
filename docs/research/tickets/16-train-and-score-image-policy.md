# 16: Train and score the image policy on human demos
What to build: render images for the human dataset by replay, train the image policy (Colab if the local GPU is too small), score it over the 100 held-out seeds. If it fails, report that honestly.
Blocked by: 02, 11, 15
Kind: HITL
- [x] Success rate with confidence interval printed; the README line says which policy (state or image).

**Changed 2026-11-15 (PRD-v2):** blocked by 25 (was 02, 11, 15). Uses the v2 design (gripper-move actions, chunk
execution, the same training set as the final state policy). Not started until the state policy passes ticket 25.

## Result 2026-11-16 (verified: the commands below, exit 0)
- Code: `image_policy_v2.py`, `train_image_v2.py` (prepare, fit, score), tests in `tests/test_image_policy_v2.py` (8 pass).
  The v2 design with a camera image instead of the cube pose; the network gets the image plus joints and gripper
  position (forward kinematics), never the cube pose; same rollout (`PolicyV2` with an `observe` hook).
- Data: `runs/v2_images_human_20261116` (100 demos), `runs/v2_images_copies_20261116` (798 copies), arrays in
  `runs/image_v2_arrays_copies_20261116` (80 demos + 626 copies = 97,179 training frames; same split as the state policy).
- Training: Kaggle T4, 40 epochs, 1205 s, shift augmentation 4 px, seed 0; loss 1.33 -> 0.42. Files in
  `runs/image_v2_policy_kaggle_20261116`.
- Scores, policy alone: development seeds 97/100 (CI 0.92-0.99); held-out seeds 97/100 (CI 0.92-0.99), scored once,
  with no choice made on held-out data. Command: `python -m teleop_il.train_image_v2 score RUN [--held-out]`.
- Test change: the board's tiny-overfit threshold (loss under 10 percent of start) is met with a small CNN (8 channels,
  300 epochs, no augmentation) because the full-size CNN is too slow on this CPU for a unit test. Threshold unchanged.
- Limits: one training seed; no C++ port; not in `make final-scores`.
- Open: a human-only image run is training locally (100 epochs, seed 0) and will be scored once on held-out.
