# Demonstrations

| folder | what it is |
|---|---|
| `human_demos_100/` | 100 demos recorded with the webcam teleop (hand tracking, grasp assist) and cleaned: only lifts without hand-tracking loss over 0.3 s. One `episode_*.npz` each plus `manifest.json`. |
| `transformed_copies_798/` | 798 copies made from them: each demo's gripper path moved to a new cube position, replayed in the simulator, kept only if the cube lifted (`python -m teleop_il.transform`). |

An episode holds simulator states only (joint positions and velocities, cube pose, joint targets, flags, the seed and the
start state). No webcam image and no hand landmark is stored. **The grasp segment of each demo is the grasp assist's
motion, not the human's.** Load them with `teleop_il.dataset.read_dataset`.

Rendered images for the image policy (about 1 GB) are not stored; `python -m teleop_il.replay DATA OUT` draws them from
these states.
