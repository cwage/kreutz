"""Turn an archive run into a shortlist of candidate unreported comets.

    docker compose run --rm kreutz kreutz.shortlist /data/runs/archive/n8/*.jsonl

Tracks fired by the search (as in evaluate.py) are scored for persistence
(persist.py: do they keep going in the frames after?), and anything near
a reported comet's path is set aside as known. Reports only cover the
stretch the reporter measured, and a comet stays visible for hours after,
so the comparison extrapolates each report's straight-line motion with a
tolerance that widens with time (Kreutz comets speed up as they near the
Sun). On the July 2025 benchmark every persistent "unexplained" track
turned out to be a confirmed comet seen after its report ended.

Writes data/shortlist/<first day>_<last day>.json, sorted most
persistent first; look.py renders a vetting strip for any of them.
"""

import argparse
import json
from datetime import timedelta

import numpy as np

from . import DATA
from .evaluate import FRAME, load, report_points, tracks
from .persist import persistence

THRESHOLD = 6.0
LINK = 3
# Persistence: >= STRONG was only ever real comets on the benchmark; the
# MAYBE..STRONG band holds faint short-lived comets (0075 scored 1.3) and a
# few noise tracks, worth a look by eye.
STRONG = 3.0
MAYBE = 1.5
# Report extrapolation: allowed miss = KNOWN_PX + KNOWN_GROW * frames away.
KNOWN_PX = 10
KNOWN_GROW = 1.5
KNOWN_HOURS = 12


def report_tracks(days):
    reps = [json.loads(l) for p in (DATA / "reports").glob("*.jsonl") for l in open(p)]
    out = {}
    for d in days:
        for r, pts, posted in report_points(reps, d):
            if r["kind"] in ("Potential Comet", "Additional Observations",
                             "Possible Fragment/Companion", "Known Comet Now Entering") \
                    and len(pts) >= 2:
                out[r["id"]] = (r, pts)
    return list(out.values())


def known_as(track, reps):
    """The report whose extrapolated path this track follows, or None."""
    for r, pts in reps:
        (t0, x0, y0), (t1, x1, y1) = pts[0], pts[-1]
        n = (t1 - t0) / FRAME
        if n <= 0:
            continue
        vx, vy = (x1 - x0) / n, (y1 - y0) / n
        for c in track:
            k = (c["t"] - t1) / FRAME
            if not -n - 2 <= k <= KNOWN_HOURS * 4:
                continue
            gap = max(0.0, k, -n - k)  # frames outside the reported span
            px, py = x1 + vx * k, y1 + vy * k
            if np.hypot(c["x"] - px, c["y"] - py) <= KNOWN_PX + KNOWN_GROW * gap:
                return r
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--link", type=int, default=LINK)
    args = ap.parse_args()

    from .evaluate import _persist  # shares Sequences per day

    by_time = load(args.runs)
    days = sorted({t.date() for t in by_time})
    reps = report_tracks(days)
    trs = tracks(by_time, args.threshold, args.link)
    out = []
    for tr in trs:
        fire = tr[args.link - 1]
        k = known_as(tr, reps)
        rec = {"time": fire["time"], "x": fire["x"], "y": fire["y"],
               "vx": fire["vx"], "vy": fire["vy"], "windows": len(tr),
               "scores": [c["score"] for c in tr]}
        if k:
            rec["known"] = f"{k['reporter']} {k['id']}"
            out.append(rec)
            continue
        rec["persistence"] = _persist(tr, args.link)
        out.append(rec)
    cands = [r for r in out if "known" not in r and np.isfinite(r["persistence"])
             and r["persistence"] >= MAYBE]
    cands.sort(key=lambda r: -r["persistence"])
    dest = DATA / "shortlist" / f"{days[0]}_{days[-1]}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({"candidates": cands, "all": out}, indent=1, default=str))
    known = [r for r in out if "known" in r]
    print(f"{len(trs)} tracks fired; {len(known)} on reported comets; "
          f"{sum(r['persistence'] >= STRONG for r in cands)} strong and "
          f"{sum(r['persistence'] < STRONG for r in cands)} maybe candidates -> {dest}")
    for r in cands:
        print(f"  {'STRONG' if r['persistence'] >= STRONG else 'maybe '} {r['time'][:16]} "
              f"({r['x']:.0f},{r['y']:.0f}) v=({r['vx']:+.1f},{r['vy']:+.1f}) "
              f"windows {r['windows']} persistence {r['persistence']:+.1f}")


if __name__ == "__main__":
    main()
