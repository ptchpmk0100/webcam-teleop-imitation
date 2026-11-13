# 12: Train and score the state policy on human demos
What to build: train the state MLP on the human dataset (local GPU, or a Colab script and bundle if needed), export ONNX, score over the 100 held-out seeds, and look for mode averaging.
Blocked by: 02, 09, 11
Kind: HITL
- [ ] Success rate with confidence interval printed, next to the teleop baseline and the expert ceiling.
- [ ] Null checks pass in the same run.
