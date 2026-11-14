# 08: Replay and image render
What to build: replay a stored episode from its recorded state and actions, and render camera images into a new folder.
Blocked by: 06, 07
Kind: AFK
- [ ] Test command exits 0: replaying an expert episode reproduces its success flag and final cube pose within 5 mm; the output folder is new and inputs stay unchanged.
- [ ] Rendered images are not blank (checked).

Done 2026-11-14. Acceptance `pytest tests/test_replay.py` exits 0 (8 passed, stable over 3 runs; full suite 176). Replay re-simulates from the stored initial state and joint targets and matches the recorded cube path within 5 mm at every step (expert episodes); a gripper that never closes does not lift, so replay really runs the actions. Images are rendered from each step's STORED state (human demos are simulated live in 5 ms slices on camera timing and cannot be re-run at 20 Hz), 96 x 128, into a new folder with its own manifest (camera, size, sha256 of the source manifest); the dataset folder is unchanged (hashes before and after). Not blank: pixel std above 10, the green cube visible, motion between first and last frame.
Scene changes (visual only; expert still 100/100, null checks pass): a top light and a lighter floor (images were nearly flat, std 7 of 255, now about 36); a named "policy" camera in the scene, front-right of the arm, chosen from 3 candidates because it keeps the whole arm, both spawn corners and a lifted cube in view.
Found and fixed: Env.render drew the arm one 5 ms substep stale (MuJoCo body positions after mj_step), so evaluation images differed from training images on 400-500 pixels by up to 166 levels; it now recomputes positions on a scratch copy. Image comparisons allow 2 levels: rendering one state twice differs by up to 1 level on about 20 pixels (GPU noise, measured). Mutations: 6 of 7 caught; the 7th (Env.render using the free camera) was caught after adding the evaluation-equals-training test.
Status: DONE.
