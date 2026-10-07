"""Measure an object frame by frame and print a Sungrazer report block.

    docker compose run --rm kreutz kreutz.report /data/live/alerts/<alert>.json
    docker compose run --rm kreutz kreutz.report 2026-10-06T14:00 1954 1218 -9 -2

Follows the object through the significance maps (render.follow: 4-sigma
peak within a few px of the prediction, velocity refit as it goes), then
refines each position to the brightness-weighted centroid of the
background-subtracted raw frame around that peak, and prints one line per
frame in the form Sungrazer's report form takes:

    HHMM  x  y

in CCOR-1 L1b image coordinates (x right, y down from the top). The form
wants at least five consecutive frames in a straight line at a steady
speed; fewer gets a warning here. The submission itself is yours to make.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta

import numpy as np

from .frames import BG_FRAMES, BG_SKIP, STAR_FRAMES, Sequence, frame_paths
from .render import follow

MIN_POINTS = 5
CENTROID_R = 3


def centroid(seq, i, x, y):
    """Brightness-weighted centroid of the background-subtracted frame in a
    small box around (x, y), report coords; falls back to (x, y)."""
    H = seq.shape[0]
    r, c = int(round(H - 1 - y)), int(round(x))
    d = seq.raw(i) - seq._background(i)
    win = d[r - CENTROID_R:r + CENTROID_R + 1, c - CENTROID_R:c + CENTROID_R + 1]
    if win.shape != (2 * CENTROID_R + 1,) * 2 or not np.isfinite(win).any():
        return x, y
    w = np.nan_to_num(win) - np.nanmedian(win)
    w[w < 0] = 0
    if w.sum() <= 0:
        return x, y
    rr, cc = np.mgrid[-CENTROID_R:CENTROID_R + 1, -CENTROID_R:CENTROID_R + 1]
    dr, dc = (w * rr).sum() / w.sum(), (w * cc).sum() / w.sum()
    return c + dc, H - 1 - (r + dr)


def measure(seq, i0, x, y, vx, vy, back=12, ahead=12, lost=2):
    """[(time, x, y, sigma)] for every frame the object was seen in."""
    track = follow(seq, i0, x, y, vx, vy, back, ahead, lost)
    out = []
    for i in sorted(track):
        px, py, sig = track[i]
        if sig is None:
            continue
        cx, cy = centroid(seq, i, px, py)
        out.append((seq.times[i], cx, cy, sig))
    return out


def block(points):
    """The report lines, plus a one-line verdict on straightness."""
    lines = [f"{t:%H%M} {x:.0f} {y:.0f}" for t, x, y, _ in points]
    note = ""
    if len(points) >= 3:
        k = np.array([(t - points[0][0]) / timedelta(minutes=15) for t, *_ in points])
        xs = np.array([p[1] for p in points]); ys = np.array([p[2] for p in points])
        rx = xs - np.polyval(np.polyfit(k, xs, 1), k)
        ry = ys - np.polyval(np.polyfit(k, ys, 1), k)
        note = f"straight-line residual {np.hypot(rx, ry).max():.1f} px over {len(points)} frames"
    return lines, note


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", nargs="+", help="an alert .json, or TIME X Y VX VY")
    ap.add_argument("--back", type=int, default=12)
    ap.add_argument("--ahead", type=int, default=12)
    args = ap.parse_args()
    if len(args.what) == 1:
        d = json.load(open(args.what[0]))
        c = (d["track"] if isinstance(d, dict) else d)[-1]
        t, x, y, vx, vy = datetime.fromisoformat(c["time"][:19]), c["x"], c["y"], c["vx"], c["vy"]
    else:
        t, x, y, vx, vy = datetime.fromisoformat(args.what[0]), *map(float, args.what[1:5])
    t = t.replace(second=25)
    warm = timedelta(minutes=15) * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.back + 2)
    seq = Sequence(frame_paths(t - warm, t + timedelta(minutes=15) * (args.ahead + 1)))
    pts = measure(seq, seq.times.index(t), x, y, vx, vy, args.back, args.ahead)
    lines, note = block(pts)
    print(f"# CCOR1, {pts[0][0]:%Y-%m-%d} UTC, {len(pts)} frames; {note}")
    print("\n".join(lines))
    if len(pts) < MIN_POINTS:
        print(f"# only {len(pts)} frames; Sungrazer wants at least {MIN_POINTS}", file=sys.stderr)


if __name__ == "__main__":
    main()
