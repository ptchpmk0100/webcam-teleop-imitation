# 05: Week-1 video and ChArUco scale check
What to build: a demo video of 60 s or less of the live webcam driving the arm with pretrained models only, and one ChArUco scale check of hand displacement against arm motion recorded as a number.
Blocked by: 04
Kind: HITL
- [ ] Video of 60 s or less exists (not committed; mp4 is gitignored).
- [ ] The scale-check number and how it was measured are recorded in the research notes.

Changed 2026-11-14 (short G3, Khoi: "everything ok"): the scale check uses a ruler instead of a printed ChArUco board, and covers left-right and up-down only (forward-back is on the W/S keys since the redesign): move the hand 10 cm sideways and 10 cm up along a ruler with a session log running; the arm motion is read from the log. The video is recorded with the desktop's built-in screen recorder (Ctrl+Alt+Shift+R) and shows the redesigned teleop (grill Q16).

## Closed 2026-11-16 by Khoi's approval (short G3: "mark as approve and next")
- Scale check: Khoi did the ruler protocol (10 cm sideways and up, return, hold 2 s; sessions `runs/try14.npz`,
  `runs/try15.npz`). The log has no ruler markers, so a centimetre figure was not computed. What the logs show
  (verified, `make diagnose` and a fit of arm position on hand image position over engaged runs): left-right arm
  motion follows the hand with 0.69 m per image width on try14 (correlation 0.98; design 0.6) and 0.44 on try15
  (correlation 0.73); up-down -0.45 m per image height on try15 (correlation -0.92; design -0.5). Lag 0 to 30 ms.
  Up-down on try14 was clipped by the workspace limits. The ChArUco/ruler check was only a sanity check (grill Q16).
- Video: see ticket 17 (the live-teleop clip, if recorded, is Khoi's own screen recording; mp4 not committed).
