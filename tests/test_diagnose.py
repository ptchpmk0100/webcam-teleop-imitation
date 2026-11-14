"""Session log and lag estimate: the tools that turn 'it feels delayed' into a number."""
import re

import numpy as np
import pytest

from teleop_il.diagnose import SessionLog, estimate_lag_s, summarize


def test_the_lag_between_a_hand_signal_and_a_delayed_arm_signal_is_recovered():
    t = np.arange(0, 20, 1 / 30)
    hand = np.sin(2 * np.pi * 0.4 * t) + 0.5 * np.sin(2 * np.pi * 0.9 * t + 1.0)
    delay = 0.25
    arm = np.interp(t - delay, t, hand, left=hand[0])
    assert estimate_lag_s(t, hand, arm) == pytest.approx(delay, abs=0.02)


def test_an_arm_that_follows_with_no_delay_has_zero_lag():
    t = np.arange(0, 20, 1 / 30)
    hand = np.sin(2 * np.pi * 0.5 * t)
    assert estimate_lag_s(t, hand, 0.08 * hand) == pytest.approx(0.0, abs=0.02)


def test_a_signal_that_does_not_move_has_no_lag_estimate():
    t = np.arange(0, 5, 1 / 30)
    assert np.isnan(estimate_lag_s(t, np.zeros_like(t), np.zeros_like(t)))


def make_log(n=60):
    log = SessionLog()
    for i in range(n):
        info = {"raw": np.array([0.5, 0.5, 2.0]), "filtered": np.array([0.5, 0.5, 2.0]),
                "target": np.array([0.2, 0.0, 0.09]), "pinch_gap_m": 0.05}
        log.add(t=i / 30, detected=i % 10 != 0, engaged=True, info=info if i % 10 else {},
                ee=np.array([0.2, 0.0, 0.09]), gripper=1.2, key_closed=False)
    return log


def test_a_saved_log_reads_back_with_nan_where_the_hand_was_missing(tmp_path):
    path = tmp_path / "run1.npz"
    make_log().save(path)
    data = np.load(path)
    assert data["t"].shape == (60,) and data["raw"].shape == (60, 3) and data["ee"].shape == (60, 3)
    assert np.isnan(data["raw"][0]).all() and not np.isnan(data["raw"][1]).any()
    assert data["detected"].sum() == 54


def test_saving_a_log_never_overwrites_an_existing_file(tmp_path):
    path = tmp_path / "run1.npz"
    make_log().save(path)
    with pytest.raises(FileExistsError):
        make_log().save(path)


def test_the_summary_reports_rate_detection_and_pinch_gap(tmp_path):
    path = tmp_path / "run1.npz"
    make_log().save(path)
    text = summarize(path)
    assert "60 frames" in text and "detected 90%" in text and "pinch gap" in text


def test_the_summary_measures_the_hand_to_arm_delay_with_the_mapping_signs(tmp_path):
    """Hand right (u smaller) = arm -y; hand up (v smaller) = arm +z. The arm trails the hand by 0.2 s here."""
    t = np.arange(0, 20, 1 / 30)
    u = 0.5 + 0.1 * np.sin(2 * np.pi * 0.4 * t); v = 0.5 + 0.1 * np.sin(2 * np.pi * 0.3 * t + 1)
    late = lambda x: np.interp(t - 0.2, t, x, left=x[0])
    log = SessionLog()
    for i in range(len(t)):
        ee = np.array([0.2, 0.8 * (late(u)[i] - 0.5), 0.09 - 0.8 * (late(v)[i] - 0.5)])
        log.add(t=t[i], detected=True, engaged=True, info={"raw": np.array([u[i], v[i], 2.0]), "pinch_gap_m": 0.08},
                ee=ee, gripper=1.2, key_closed=False)
    path = tmp_path / "run.npz"
    log.save(path)
    lags = [int(x) for x in re.findall(r"real position: (\d+) ms", summarize(path))]
    assert len(lags) == 2 and all(abs(x - 200) <= 20 for x in lags)


def test_a_saved_log_keeps_the_full_hand_landmarks_so_the_session_can_be_replayed(tmp_path):
    log = SessionLog()
    img, world = np.random.default_rng(0).random((21, 3)), np.random.default_rng(1).random((21, 3))
    log.add(t=0.0, detected=True, engaged=True, info={}, ee=np.zeros(3), gripper=1.2, key_closed=False, image=img, world=world)
    log.add(t=0.03, detected=False, engaged=True, info={}, ee=np.zeros(3), gripper=1.2, key_closed=False)
    path = tmp_path / "run.npz"
    log.save(path)
    data = np.load(path)
    assert np.array_equal(data["image"][0], img) and np.array_equal(data["world"][0], world)
    assert np.isnan(data["image"][1]).all() and data["image"].shape == (2, 21, 3)
