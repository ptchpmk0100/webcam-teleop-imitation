# 07: Dataset module
What to build: one episode per file with full MuJoCo state, joint targets, gripper, cube pose, seed, success and tracking flags; a versioned manifest; read, validate and split by episode.
Blocked by: 01
Kind: AFK
- [ ] Test command exits 0: write-read round trip is exact; a malformed manifest fails loudly; train and validation share no episode.
