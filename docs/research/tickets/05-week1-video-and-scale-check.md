# 05: Week-1 video and ChArUco scale check
What to build: a demo video of 60 s or less of the live webcam driving the arm with pretrained models only, and one ChArUco scale check of hand displacement against arm motion recorded as a number.
Blocked by: 04
Kind: HITL
- [ ] Video of 60 s or less exists (not committed; mp4 is gitignored).
- [ ] The scale-check number and how it was measured are recorded in the research notes.

Changed 2026-11-14 (short G3, Khoi: "everything ok"): the scale check uses a ruler instead of a printed ChArUco board, and covers left-right and up-down only (forward-back is on the W/S keys since the redesign): move the hand 10 cm sideways and 10 cm up along a ruler with a session log running; the arm motion is read from the log. The video is recorded with the desktop's built-in screen recorder (Ctrl+Alt+Shift+R) and shows the redesigned teleop (grill Q16).
