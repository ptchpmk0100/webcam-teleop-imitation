# Trained models

Every model here was trained in this repository's code on the 100 human demos in [`../datasets/`](../datasets/). Scores
are on the 100 held-out cube positions (seeds 10000-10099) the policies never saw, **policy alone** (no grasp assist,
no hand, no webcam). The numbers and the commands that print them are in the top-level [README](../README.md); the
raw output is in [`../results/`](../results/).

| folder | what it is | held-out lifts |
|---|---|---|
| `state_policy_v2/human_seed{0,1,2}` | state policy (cube-relative numbers in, 20 gripper moves out), 80 human demos, 3 training seeds | 96, 90, 84 of 100 |
| `state_policy_v2/copies_seed{0,1,2}` | the same, plus 626 transformed copies of the 80 training demos, 3 training seeds | 100, 100, 100 of 100 |
| `image_policy_v2/` | camera image + arm joints in (no cube pose), same output; trained once on Kaggle (T4, 40 epochs) | 97 of 100 |

Each folder holds `policy.onnx` (what the rollouts run), `policy.pt` (PyTorch weights) and the run's JSON
(`run.json`, or `fit.json` and `score_*.json` for the image policy). `state_policy_v2/SOURCES.txt` maps each folder to
the original `runs/` folder name that the results files refer to.

```
make policy-demo RUN=models/state_policy_v2/copies_seed0        # watch the policy alone, live
python -m teleop_il.train_image_v2 score models/image_policy_v2 --n 10    # image policy; writes a new score file here
```

ONNX contracts are in the module docstrings of `src/teleop_il/policy_v2.py` and `src/teleop_il/image_policy_v2.py`.
