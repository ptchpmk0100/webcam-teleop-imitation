# 07: Dataset module
What to build: one episode per file with full MuJoCo state, joint targets, gripper, cube pose, seed, success and tracking flags; a versioned manifest; read, validate and split by episode.
Blocked by: 01
Kind: AFK
- [ ] Test command exits 0: write-read round trip is exact; a malformed manifest fails loudly; train and validation share no episode.

Done 2026-11-14. Acceptance command `pytest tests/test_dataset.py` exits 0 (17 passed; full suite 168 passed). Format: one compressed .npz per episode (one row per 20 Hz step, recorded before the step's joint targets: time, full qpos and qvel, joint targets with the gripper, cube pose, tracking-lost and assist-active flags; plus seed, source, success, flagged, assist count and the initial simulator snapshot), manifest.json with format_version 1, per-episode sha256 and step count; the folder is created once and never written again. Checks: exact round trip including dtypes; the stored initial state restores the simulation; 8 kinds of malformed manifest or file each fail with a message naming the problem; split is by episode and grouped by cube seed, so retries of one cube position never straddle train and validation. Mutations: 6 of 6 caught (checksum, split grouping, dtype, seed check, recording after the step, duplicates). Expert episodes record as successes (seeds 3 and 4).
Status: DONE.
