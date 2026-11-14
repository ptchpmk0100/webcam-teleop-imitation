# 15: Image policy training code
What to build: a small CNN plus MLP on rendered image and joints with action chunking, trained locally on scripted-expert episodes, exported to ONNX.
Blocked by: 08, 09
Kind: AFK
- [ ] Test command exits 0: the 5-episode overfit and ONNX parity checks from the frozen thresholds.

Done 2026-11-15. Acceptance `pytest tests/test_image_policy.py` exits 0 (7 passed; full suite 205): on 5 expert episodes and their rendered images the training loss falls below 10 percent of its start within 10 epochs (about 20 s on CPU; a prototype reached 1.2 percent after 50); ONNX Runtime matches PyTorch within 1e-4 on recorded images and joints; the trained policy reproduces the recorded joint targets (mean error below 0.05 rad); the ONNX interface is "image" float32 (batch, 96, 128, 3) raw RGB plus "joints" float32 (batch, 6) to "actions" (batch, 10, 6); the policy ignores the cube pose in the observation and reacts to the cube moving in the picture. Model: 3 conv layers (32 channels) and a spatial-softmax keypoint layer, then an MLP with the joints; pixel scaling and normalisation inside the graph. Determinism checks allow 1e-4 because the GPU render noise from ticket 08 reaches the output (about 3e-6). Mutations: 4 of 4 caught (no de-normalisation, no optimiser step, blank image instead of the camera, velocities instead of joint positions).
`python -m teleop_il.train_image` (for ticket 16) smoke-run end to end on 20 expert episodes, 3 epochs, 5 seeds: loss 1.004 to 0.056, validation error 0.044 rad, 0/5 lifted (a pipeline check only; a CPU run is about 10 s per epoch per 20 episodes).
Status: DONE.
