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
speed; fewer gets a warning here, and so does an object that doesn't move
(vet.STATIONARY). The submission itself is yours to make.
"""

import argparse
import json
import sys
from datetime import datetime

import numpy as np

from .frames import BG_FRAMES, BG_SKIP, CADENCE, INSTRUMENT, STAR_FRAMES, Sequence, frame_paths
from .render import follow
from .vet import STATIONARY

MIN_POINTS = 5
CENTROID_R = 3
# Measured frames have to lie on a line. The largest residual allowed grows
# with the arc, since a comet accelerates: 0.5 px over 5 frames and 1.2-1.5
# px over 14 for this season's real comets, 3.1 px over 6 for the glare
# feature of 2026-10-11.
RESIDUAL_PER_FRAME = 0.3
FORM_URL = "https://sungrazer.nrl.navy.mil/report"


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
    """[(time, x, y, sigma)] for every frame the object was seen in; empty
    if it was seen in none."""
    try:
        track = follow(seq, i0, x, y, vx, vy, back, ahead, lost)
    except ValueError:  # not seen anywhere in the span
        return []
    out = []
    for i in sorted(track):
        px, py, sig = track[i]
        if sig is None:
            continue
        cx, cy = centroid(seq, i, px, py)
        out.append((seq.times[i], cx, cy, sig))
    return out


def fit(points):
    """A straight line through measured points [(time, x, y, sigma)]:
    (vx, vy) in px/frame and the largest residual in px; zeros for fewer
    than two points."""
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    k = np.array([(t - points[0][0]) / CADENCE for t, *_ in points])
    xs = np.array([p[1] for p in points]); ys = np.array([p[2] for p in points])
    px, py = np.polyfit(k, xs, 1), np.polyfit(k, ys, 1)
    resid = np.hypot(xs - np.polyval(px, k), ys - np.polyval(py, k)).max()
    return float(px[0]), float(py[0]), float(resid)


def straight(points):
    """Do the points lie on a line (largest residual within
    RESIDUAL_PER_FRAME a point)?"""
    return fit(points)[2] <= RESIDUAL_PER_FRAME * len(points)


def inliers(points):
    """(the points on the line, the one left out or None): all of them when
    they lie on a line; else, from MIN_POINTS on, all but the one whose
    removal puts the rest on one, if one does; else all of them. One
    frame's measurement can land on a star or a noise peak beside a faint
    comet: CCOR-1's straight-line rejections were six, three of them real
    comets (2026-11-09 01:45, 11-22 08:45 twice, 11-28 11:00, reported by
    Barraclough), each bent by one frame, and replayed with this
    (scripts/live-replay.py) each confirms at the rejection's own frame,
    0.8-1.1 px over five with that frame left out (00:45, 07:45, 09:00);
    the other two (10-13 07:15, 10-17 03:45) the watcher no longer alerts
    on at all."""
    points = list(points)
    if len(points) < MIN_POINTS or straight(points):
        return points, None
    j = min(range(len(points)), key=lambda k: fit(points[:k] + points[k + 1:])[2])
    rest = points[:j] + points[j + 1:]
    return (rest, points[j]) if straight(rest) else (points, None)


def block(points):
    """The report lines, plus a one-line verdict on straightness. A frame
    off the line that inliers() leaves out isn't reported."""
    use, out = inliers(points)
    lines = [f"{t:%H%M} {x:.0f} {y:.0f}" for t, x, y, _ in use]
    note = ""
    if len(use) >= 3:
        note = f"straight-line residual {fit(use)[2]:.1f} px over {len(use)} frames"
        if out is not None:
            note += f", the {out[0]:%H:%M} frame off the line left out"
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
    warm = CADENCE * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.back + 2)
    seq = Sequence(frame_paths(t - warm, t + CADENCE * (args.ahead + 1)))
    i = seq.index(t)
    t = seq.times[i]
    pts = measure(seq, i, x, y, vx, vy, args.back, args.ahead)
    if not pts:
        print(f"# not seen in any frame within {args.back} before and {args.ahead} after {t:%Y-%m-%d %H:%M} UTC "
              "along that motion (no 4-sigma peak near the predicted position)", file=sys.stderr)
        sys.exit(1)
    lines, note = block(pts)
    # the vetting GIF belongs to the alert, also when given its .confirmed.json
    gif = args.what[0].replace(".confirmed.json", ".json").replace(".json", ".gif") if len(args.what) == 1 else None
    if gif and gif.startswith("/data/"):
        gif = gif[1:]  # the container's /data is the project's data/; print it as seen from there
    print(f"# {INSTRUMENT.telescope}, {pts[0][0]:%Y-%m-%d} UTC, {len(pts)} frames; {note}")
    vx, vy, _ = fit(pts)
    still = len(pts) >= 3 and np.hypot(vx, vy) < STATIONARY
    print(f"# motion ({vx:+.1f},{vy:+.1f}) px/frame" + (f": stationary (under {STATIONARY} px/frame), not a comet" if still else ""))
    if gif:
        print(f"# vetting animation: {gif} (real if it holds still while stars drift)")
    print(f"# form: {FORM_URL}  telescope {INSTRUMENT.telescope}, group Kreutz if it came in along the lane")
    print("\n".join(lines))
    if len(pts) < MIN_POINTS:
        print(f"# only {len(pts)} frames; Sungrazer wants at least {MIN_POINTS}", file=sys.stderr)


if __name__ == "__main__":
    main()
