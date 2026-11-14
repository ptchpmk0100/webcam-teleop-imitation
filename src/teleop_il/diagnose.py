"""Session log and lag estimate: turns "it feels delayed" into numbers. Logs are written to new files, never over old ones."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_NAN3 = np.full(3, np.nan)


class SessionLog:
    """One row per camera frame: raw and filtered hand features, commanded target, real gripper position, pinch gap."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def last_ee(self) -> np.ndarray:
        return self._rows[-1]["ee"]

    @property
    def last_image(self) -> np.ndarray:
        return self._rows[-1]["image"]

    def image_at(self, index: int) -> np.ndarray:
        return self._rows[index]["image"]

    def add(self, *, t: float, detected: bool, engaged: bool, info: dict, ee, gripper: float, key_closed: bool,
            image=None, world=None) -> None:
        self._rows.append({
            "t": t, "detected": detected, "engaged": engaged, "gripper": gripper, "key_closed": key_closed,
            "raw": np.asarray(info.get("raw", _NAN3), float), "filtered": np.asarray(info.get("filtered", _NAN3), float),
            "target": np.asarray(info.get("target", _NAN3), float), "ee": np.asarray(ee, float),
            "pinch_gap_m": float(info.get("pinch_gap_m", np.nan)),
            "image": np.full((21, 3), np.nan) if image is None else np.asarray(image, float),    # full landmarks: lets a
            "world": np.full((21, 3), np.nan) if world is None else np.asarray(world, float),    # session be replayed offline
        })

    def save(self, path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists; logs go to new files")
        columns = {k: np.array([r[k] for r in self._rows]) for k in self._rows[0]}
        with open(path, "xb") as f:                    # "x": fails rather than overwrites
            np.savez(f, **columns)


def estimate_lag_s(t, hand, arm, max_lag_s: float = 1.0, grid_hz: float = 100.0) -> float:
    """How many seconds the arm signal trails the hand signal (cross-correlation on a uniform grid); NaN if either is flat."""
    t, hand, arm = (np.asarray(a, float) for a in (t, hand, arm))
    ok = ~(np.isnan(hand) | np.isnan(arm))
    if ok.sum() < 10 or np.ptp(hand[ok]) < 1e-9 or np.ptp(arm[ok]) < 1e-9:
        return float("nan")
    grid = np.arange(t[ok][0], t[ok][-1], 1.0 / grid_hz)
    h = np.interp(grid, t[ok], hand[ok]); a = np.interp(grid, t[ok], arm[ok])
    h, a = h - h.mean(), a - a.mean()
    best, best_lag = -np.inf, 0
    for k in range(int(max_lag_s * grid_hz) + 1):
        c = float(np.dot(h[: len(h) - k], a[k:])) / ((len(h) - k) * h.std() * a.std() + 1e-12)
        if c > best:
            best, best_lag = c, k
    return best_lag / grid_hz


def summarize(path) -> str:
    d = np.load(path)
    t, det = d["t"], d["detected"].astype(bool)
    n = len(t)
    lines = [f"{n} frames in {t[-1] - t[0]:.1f} s = {(n - 1) / max(t[-1] - t[0], 1e-9):.1f} FPS; hand detected {100 * det.mean():.0f}%"]
    live = d["engaged"].astype(bool) & det
    if live.sum() > 30:
        # hand right in the image (u smaller) = arm -y, so the arm's y follows +u; up (v smaller) = arm +z, so z follows -v
        for name, hand_sig, arm_sig in (("left-right", d["raw"][live, 0], d["ee"][live, 1]),
                                        ("up-down", -d["raw"][live, 1], d["ee"][live, 2])):
            lag = estimate_lag_s(t[live], hand_sig, arm_sig)
            lines.append(f"lag, hand {name} to the arm's real position: " + ("not enough motion" if np.isnan(lag) else f"{1000 * lag:.0f} ms"))
        rest = np.abs(np.diff(d["raw"][live], axis=0)).max(axis=1) < 0.002
        if rest.sum() > 10:
            lines.append(f"frame-to-frame jitter of the raw hand position when still: {1000 * np.nanstd(np.diff(d['raw'][live][:, :2], axis=0)[rest]):.2f} thousandths of the image")
    gap = d["pinch_gap_m"][det]
    gap = gap[~np.isnan(gap)]
    if len(gap):
        closed = d["gripper"] < 0
        lines.append(f"pinch gap (thumb to index tip, metres): min {gap.min():.3f}, median {np.median(gap):.3f}, max {gap.max():.3f}; "
                     f"gripper closed {100 * closed.mean():.0f}% of frames, closed by key {100 * d['key_closed'].mean():.0f}%")
        lines.append("  (the gripper closes below 0.030 and opens above 0.050)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise a session log written by `make demo` with --log.")
    ap.add_argument("log", type=Path)
    print(summarize(ap.parse_args(argv).log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
