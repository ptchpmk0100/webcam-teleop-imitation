# 09: State policy training and ONNX export
What to build: an MLP on state predicting an action chunk of about 10 steps, trained locally on scripted-expert episodes, exported to ONNX.
Blocked by: 06, 07
Kind: AFK
- [ ] Test command exits 0: loss falls below 10 percent of its start on a 5-episode set; ONNX output matches PyTorch within 1e-4 on recorded observations.
- [ ] The trained-on-expert policy is scored through the eval harness and printed (not a pass criterion, a pipeline smoke result).
