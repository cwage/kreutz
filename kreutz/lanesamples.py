"""Lane samples for lane.py from archive runs: the windows of fired tracks
that match the instrument's own Kreutz reports by position.

    KREUTZ_INSTRUMENT=ccor2 docker compose run --rm kreutz kreutz.lanesamples /data/runs-ccor2/n8/*.jsonl

Prints the samples as JSON, [x, y, vx, vy, score, time] per window as in
lane-samples.json, for saving as the instrument's lane-samples file (see
instrument.py). Only windows at or above --min-score are kept: the flow
field is a distance-weighted mean of its samples, and a faint track's
velocity is the noisier.

With --from-reports the samples come from the reports themselves instead
of runs: each measured point of a Kreutz report with two or more, with the
report's mean velocity (first point to last), and no score. Coarser than
search tracks (a report is a few points over an hour), but it needs no
frames, so it's how a new instrument gets its first lane; the lane search
those samples enable then finds the comets whose tracks replace them.

    KREUTZ_INSTRUMENT=ccor2 docker compose run --rm kreutz kreutz.lanesamples --from-reports > kreutz/lane-samples-ccor2.json
"""

import argparse
import json
import sys

from .evaluate import FRAME, load, tracks
from .frames import INSTRUMENT
from .lane import field_or_none
from .shortlist import FRAME_ROWS, _known_by_position, all_reports, report_tracks


def samples(runs, threshold, link, min_score):
    by_time = load(runs)
    if not by_time:
        return []
    days = sorted({t.date() for t in by_time})
    reps = [(r, pts) for r, pts in report_tracks(days, all_reports()) if r.get("group") == "Kreutz"]
    field = field_or_none((FRAME_ROWS, 2048))
    out, matched = [], set()
    for tr in tracks(by_time, threshold, link):
        r = _known_by_position(tr, reps, field)
        if r is None:
            continue
        matched.add(r["id"])
        out += [[c["x"], c["y"], c["vx"], c["vy"], c["score"], c["time"]]
                for c in tr if c["score"] >= min_score]
    print(f"{INSTRUMENT.label}: {len(out)} samples from {len(matched)} of {len(reps)} Kreutz reports "
          f"on {days[0]}..{days[-1]}", file=sys.stderr)
    return sorted(out, key=lambda s: s[5])


def from_reports():
    """[x, y, vx, vy, 0, time] for every point of the instrument's Kreutz
    reports with two or more points, in time order."""
    from datetime import date
    out, n = [], 0
    for r, pts in report_tracks(sorted({date.fromisoformat(r["date"]) for r in all_reports() if r.get("date")}),
                                all_reports()):
        if r.get("group") != "Kreutz":
            continue
        (t0, x0, y0), (t1, x1, y1) = pts[0], pts[-1]
        k = (t1 - t0) / FRAME
        if k <= 0:
            continue
        n += 1
        vx, vy = (x1 - x0) / k, (y1 - y0) / k
        out += [[x, y, round(vx, 2), round(vy, 2), 0, t.isoformat(timespec="seconds")] for t, x, y in pts]
    print(f"{INSTRUMENT.label}: {len(out)} points from {n} Kreutz reports", file=sys.stderr)
    return sorted(out, key=lambda s: s[5])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--from-reports", action="store_true", help="from the reports' points, not runs")
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--link", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=10.0)
    args = ap.parse_args()
    if args.from_reports:
        print(json.dumps(from_reports()))
    elif args.runs:
        print(json.dumps(samples(args.runs, args.threshold, args.link, args.min_score)))
    else:
        ap.error("give runs, or --from-reports")


if __name__ == "__main__":
    main()
