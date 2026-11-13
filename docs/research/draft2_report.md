# ATS keyword study and project choice (draft 2, 2026-11-13)

## Method
- **Sources:**
  - Draft 1: 17 JDs, from the job-search repo's `documents/applications/*/job_posting.md` (11) and the "posting" sections of the Notion Applications Pipeline (6).
  - Draft 2: those 17 plus 33 postings fetched from the job links collected by the Daily Job Digest (scraper file `job_scraper/seen_jobs.json`, 2026-10-28 to 11-08).
- **Filters:**
  - The title must match the robot/vision term list.
  - Student roles only (Praktikum, intern, Werkstudent, thesis).
  - Draft-1 duplicates removed.
  - The 60 most recent were fetched with plain GET requests.
  - Posting text was extracted from JobPosting JSON-LD or LinkedIn's description block.
  - "About us", benefits and how-to-apply sections were cut.
  - A full-text filter ran again on the remaining text.
- **Outcome of the 72 candidate links:** 33 fetched and kept, 17 failed (Indeed login wall ×11, JavaScript-only pages ×3, expired ×2, no description ×1), 5 duplicates skipped, 5 non-technical jobs skipped, 12 not fetched because of the 60-link cap. See `evidence/links2.tsv`.
- **Counting:** each keyword counts once per JD, using the approved synonym regexes in `evidence/keywords.tsv`. German and English terms are merged.
- **CV baseline:** `cv/main_master.tex` plus `cv/main_master_de.tex` in the job-search repo.

### Reproduce
The JD texts are local only, in `docs/research/local/jd_texts/` (gitignored, because the employer texts are not ours to republish). From a new, empty folder:
```
bash ../docs/research/evidence/count.sh ../docs/research/evidence/keywords.tsv ../docs/research/local/jd_texts
```
`count.sh` writes `matrix.tsv` into the current folder. Compare it with `evidence/matrix2.tsv`. On 2026-11-13 the two matched on every keyword.

## Top 25 hard-skill keywords (N=50)
| # | Keyword | JDs | % | On CV |
|---|---|---|---|---|
| 1 | Python | 38 | 76 | yes |
| 2 | C++ | 25 | 50 | listed, but no project uses it |
| 3 | Testing / validation | 24 | 48 | yes |
| 4 | Computer vision / Bildverarbeitung | 23 | 46 | yes |
| 5 | ROS / ROS 2 | 19 | 38 | yes |
| 6 | Sensors / sensor fusion | 19 | 38 | **no** |
| 7 | Hardware / electronics | 19 | 38 | yes |
| 8 | Simulation | 15 | 30 | yes |
| 9 | Machine learning | 13 | 26 | yes |
| 10 | Camera / RGB data | 13 | 26 | yes |
| 11 | Detection / recognition / tracking | 13 | 26 | yes |
| 12 | Mobile / autonomous robots | 12 | 24 | yes |
| 13 | 3D / point clouds | 11 | 22 | yes |
| 14 | Manipulation / grasping | 11 | 22 | yes |
| 15 | PyTorch | 10 | 20 | **no** |
| 16 | Humanoid robots | 10 | 20 | **no** |
| 17 | Deep learning | 9 | 18 | yes |
| 18 | Human-robot interaction | 9 | 18 | **no** |
| 19 | Deployment | 9 | 18 | **no** |
| 20 | Blender / CAD | 8 | 16 | **no** |
| 21 | Linux | 8 | 16 | yes |
| 22 | LiDAR | 7 | 14 | **no** |
| 23 | SLAM / navigation | 7 | 14 | **no** |
| 24 | Cloud / GPU | 7 | 14 | yes |
| 25 | Medical / surgical | 7 | 14 | **no** |

- **Below the cut:** TensorFlow, OpenCV, Robot learning, Industrial robots and Robot programming (6 each); Datasets (5); Calibration, Segmentation, Teleoperation, Git and Control (4 each); Docker, RL, LLM, Fine-tuning and named simulators (3 each); Pose estimation (0).
- **Soft or generic:** Robotics 40, German 33, English 30, AI/KI 28, Hands-on 26, Independent 24, Documentation 19, Teamwork 18, Communication 14, Analytical 14, MS Office 7.

## Ideas compared
- **Weighted** is the sum of the JD counts of the top-25 keywords an idea covers.
- **Gap fill** is the weighted sum of the CV-missing keywords it covers; C++ counts as missing, because no project backs it.
- Linux and Git are excluded for every idea.

| Idea | What | Weighted | Hits / 25 | Gap fill | Week-1 demo |
|---|---|---|---|---|---|
| **A + C++ node (chosen)** | webcam hand-teleop → demo dataset → imitation policy in MuJoCo; C++ ONNX inference | **219** | **14** | 37 | high |
| D | real-time C++ perception node + ROS 2 bridge to MuJoCo | 214 | 12 | 34 | low |
| B | detector trained on synthetic MuJoCo images, measured sim-to-real gap, grasping | 196 (221 with a C++ node) | 13 | 19 | medium |
| E | EKF sensor fusion (camera markers, IMU, odometry, simulated LiDAR) for a MuJoCo mobile robot | 196 | 11 | **58** | high |
| C | anomaly-detection inspection cell (MVTec AD) + sim OK/NOK sorting | 185 | 12 | 19 | high |

### Idea A, chosen
- **Keywords covered:** Python 38, C++ 25, Testing 24, Computer vision 23, Simulation 15, ML 13, Camera 13, Detection/tracking 13, Manipulation 11, PyTorch 10, Deep learning 9, HRI 9, Deployment 9, Cloud/GPU 7.
  - Below the cut: Robot learning 6, Datasets 5, Teleoperation 4, Calibration 4, named simulator (MuJoCo).
- **CV lines to aim for:**
  - "Teleop tracking error X mm (MuJoCo ground truth)."
  - "Policy succeeds in Y% of 100 seeded episodes, against a Z% scripted baseline."
  - "C++ ONNX node at N FPS."
- **Risk:** a policy trained on jittery hand demonstrations may reach a low success rate. Mitigation: also record scripted demonstrations and report both results honestly.

### Parked ideas
- **E** is the recommended second project. Sensor fusion is the most-asked keyword missing from the CV (19/50 JDs). Its LiDAR must be described as simulated.
- **B:** sim-to-real gap as the headline number. Its risk is that MuJoCo's simple rendering makes the gap large.
- **C:** inspection appears in only 2 of 50 JDs, so the ATS evidence for it is weak.
- **D:** best C++ and ROS 2 coverage, but a C++ ramp-up blocks the week-1 demo, and it conflicts with the no-ROS-2 choice.
