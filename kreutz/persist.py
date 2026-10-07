"""Does a fired track keep going? The archival search's main filter.

Live, a track has to be judged on the frames it fired from. In the archive
the frames after it already exist, and a comet keeps showing up where its
motion says it will while noise, star residue and coronal rays don't. This
is the check that, done by hand, threw out every false live alert on
2026-10-05.

The path is extrapolated from a straight-line fit to the track's own
positions, and in each following frame the peak significance within
PATH_R px of it is taken. The same is done along CONTROL_N parallel paths
offset CONTROL_OFFSET px to the side, in the same frames, which gives a
noise baseline for exactly that patch of sky and those frames: the score
is how far the real path's mean peak stands above the controls', in units
of their spread.
"""

from datetime import datetime

import numpy as np

from .frames import Sequence

AHEAD = 6
PATH_R = 2
CONTROL_N = 12
CONTROL_OFFSET = 15


def fit(track):
    """Straight-line motion through the track's positions: (x, y) at the
    last window's time, and (vx, vy) per frame."""
    t = np.arange(len(track)) - (len(track) - 1)
    xs = np.array([c["x"] for c in track])
    ys = np.array([c["y"] for c in track])
    if len(track) >= 2:
        vx, x0 = np.polyfit(t, xs, 1)
        vy, y0 = np.polyfit(t, ys, 1)
    else:
        vx, vy, x0, y0 = track[-1]["vx"], track[-1]["vy"], xs[-1], ys[-1]
    # A window's velocity is the better estimate when the chain is short.
    if len(track) < 3:
        vx, vy = track[-1]["vx"], track[-1]["vy"]
    return x0, y0, vx, vy


def _peak(sig, x, y, H):
    r, c = int(round(H - 1 - y)), int(round(x))
    win = sig[r - PATH_R:r + PATH_R + 1, c - PATH_R:c + PATH_R + 1]
    if win.size == 0 or not np.isfinite(win).any():
        return np.nan
    return float(np.nanmax(win))


def persistence(seq: Sequence, last: int, track):
    """(score, path peaks, control mean, control std): score ~0 for noise,
    several for something that keeps going. NaN if there aren't frames
    after the track."""
    x0, y0, vx, vy = fit(track)
    H = seq.shape[0]
    speed = np.hypot(vx, vy) or 1.0
    # Controls run parallel to the track, offset perpendicular to it.
    px, py = -vy / speed, vx / speed
    # +-15, +-30, ... px either side
    offsets = [CONTROL_OFFSET * (1 + j // 2) * (1 if j % 2 == 0 else -1)
               for j in range(CONTROL_N)]
    path, ctrl = [], [[] for _ in offsets]
    for k in range(1, AHEAD + 1):
        i = last + k
        if i >= len(seq.paths):
            break
        if seq.times[i] - seq.times[last] != (seq.times[1] - seq.times[0]) * k:
            break  # a gap: the extrapolation doesn't hold across it
        sig = seq.significance(i)
        x, y = x0 + vx * k, y0 + vy * k
        path.append(_peak(sig, x, y, H))
        for j, o in enumerate(offsets):
            ctrl[j].append(_peak(sig, x + px * o, y + py * o, H))
    if not path or np.all(np.isnan(path)):
        return float("nan"), path, float("nan"), float("nan")
    p = np.nanmean(path)
    c = np.array([np.nanmean(cc) for cc in ctrl if np.isfinite(cc).any()])
    if len(c) < 3:
        return float("nan"), path, float("nan"), float("nan")
    return float((p - c.mean()) / (c.std() + 1e-6)), path, float(c.mean()), float(c.std())
