# 09: State policy training and ONNX export
What to build: an MLP on state predicting an action chunk of about 10 steps, trained locally on scripted-expert episodes, exported to ONNX.
Blocked by: 06, 07
Kind: AFK
- [ ] Test command exits 0: loss falls below 10 percent of its start on a 5-episode set; ONNX output matches PyTorch within 1e-4 on recorded observations.
- [ ] The trained-on-expert policy is scored through the eval harness and printed (not a pass criterion, a pipeline smoke result).

Done 2026-11-15. Acceptance `pytest tests/test_policy.py` exits 0 (8 passed; full suite 192): on 5 expert episodes the training loss falls below 10 percent of its start; ONNX Runtime output matches PyTorch within 1e-4 on all recorded observations; the ONNX interface is fixed for the C++ side ("observation" float32 (batch, 13) = 6 joint positions + cube pose, "actions" float32 (batch, 10, 6) absolute joint targets, normalisation inside the graph). Policy: MLP 13-256-256-60, behaviour cloning on normalised 10-step chunks, re-plans every step and executes the first action. Exported with PyTorch's torch.export-based ONNX exporter (the legacy one is deprecated and left the chunk dimensions symbolic). Dependencies: CPU-only torch 2.15, onnx, onnxruntime, onnxscript in a "train" group. Mutations: 4 of 4 caught.
Pipeline smoke result (not a pass criterion): `python -m teleop_il.train_state --expert-episodes 100` (80 train, 20 validation episodes split by seed, 200 epochs, 20 s on CPU): loss 1.015 to 0.002, validation mean abs error 0.0087 rad, **ONNX policy 59/100 on the held-out seeds (95% CI 0.49-0.68)** against the expert's 100/100. On only 5 episodes it lifted 1 of its 5 training seeds although it fits their actions closely: closed-loop compounding error, likely helped by the state not telling the expert's timed phases apart (descend, close and lift start from similar states). For ticket 12 to look at, not fixed here.
Status: DONE.
