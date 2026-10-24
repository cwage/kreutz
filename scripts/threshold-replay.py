"""Replay archive runs through the live watcher's alert rules at several
thresholds: how many tracks would alert, how many of them are reported
comets, and how many pages the rest would cost.

    KREUTZ_INSTRUMENT=ccor2 PYTHONPATH=. python3 scripts/threshold-replay.py data/runs-ccor2/n8/*.jsonl

Mirrors live.py's alert path as far as the run files allow: candidates
slower than live.LIVE_SPEED_MIN are dropped before linking, the CFAR rule
and LINK windows as in evaluate.tracks, then the stack-line veto
(vet.line_stack) on what fires, and the page gate (live.pageworthy) on
what's left. Not replayed: the star, stationary, off-flow, ghost and
single-frame line vetoes, which need the follow-up's frame-by-frame
measurement; they only remove more, so the false counts here are an
upper bound. Reported comets are the instrument's Kreutz reports with two
or more points, matched by position (evaluate.matches or
shortlist._known_by_position).
"""

import sys
import warnings
from datetime import datetime, timedelta

import numpy as np

warnings.simplefilter("ignore")

from kreutz.evaluate import FRAME, load, report_points, tracks, matches  # noqa: E402
from kreutz.frames import BG_FRAMES, BG_SKIP, STAR_FRAMES, Sequence, frame_paths  # noqa: E402
from kreutz.instrument import CURRENT as INSTRUMENT  # noqa: E402
from kreutz.lane import field_or_none  # noqa: E402
from kreutz.live import LIVE_SPEED_MIN, pageworthy  # noqa: E402
from kreutz.shortlist import FRAME_ROWS, _known_by_position, all_reports  # noqa: E402
from kreutz.vet import line_stack  # noqa: E402

THRESHOLDS = [5.0, 5.5, 6.0]
DETAIL = None  # --detail=T: list each reported comet with when it fired at threshold T
LINK = 3


def main(paths):
    by_time = load(paths)
    by_time = {t: [c for c in cs if np.hypot(c["vx"], c["vy"]) >= LIVE_SPEED_MIN] for t, cs in by_time.items()}
    days = sorted({t.date() for t in by_time})
    reps = all_reports()
    rp = {}
    for d in days:
        for r, pts, posted in report_points(reps, d):
            if r.get("group") == "Kreutz" and len(pts) >= 2 and r.get("kind") != "Retracting Report":
                rp[r["id"]] = (r, pts, posted)
    lo, hi = min(by_time), max(by_time)
    rp = {k: v for k, v in rp.items() if any(lo <= t <= hi for t, _, _ in v[1])}
    field = field_or_none((FRAME_ROWS, 2048))
    sun = INSTRUMENT.sun_xy
    seqs = {}

    def seq_for(t):
        d = t.date()
        if d not in seqs:
            seqs.clear()
            start = datetime.combine(d, datetime.min.time())
            warm = FRAME * (BG_FRAMES + BG_SKIP + STAR_FRAMES + 10)
            seqs[d] = Sequence(frame_paths(start - warm, start + timedelta(days=1, hours=1)))
        return seqs[d]

    print(f"{INSTRUMENT.label}: {len(by_time)} windows over {len(days)} days ({days[0]}..{days[-1]}), "
          f"{len(rp)} Kreutz reports with points in the span")
    caught_by = {}
    for thr in THRESHOLDS:
        fired = sorted(tracks(by_time, thr, LINK), key=lambda tr: tr[LINK - 1]["t"])
        caught, veto_real, false_pages, false_quiet, vetoed = set(), [], 0, 0, 0
        first = {}
        for tr in fired:
            fire = tr[LINK - 1]
            hits = [rid for rid, (r, pts, _) in rp.items() if matches(tr, pts) or _known_by_position(tr, [(r, pts)], field)]
            seq = seq_for(fire["t"])
            try:
                i = seq.times.index(fire["t"])
                why = line_stack(seq, i, fire["x"], fire["y"], fire["vx"], fire["vy"])
            except (ValueError, IndexError):
                why = None
            if why:
                vetoed += 1
                if hits:
                    veto_real.append((fire["t"], hits[0], why))
                continue
            if hits:
                caught |= set(hits)
                for rid in hits:
                    first.setdefault(rid, fire["t"])
            elif pageworthy(fire, sun):
                false_pages += 1
            else:
                false_quiet += 1
        caught_by[thr] = caught
        n = len(days)
        print(f"threshold {thr}: {len(fired)} tracks fire; {vetoed} stack-line vetoed; "
              f"reported comets caught {len(caught)}/{len(rp)}; "
              f"unexplained {false_pages} paged ({false_pages / n:.1f}/day) + {false_quiet} quiet")
        for t, rid, why in veto_real:
            print(f"   vetoed a reported comet: {t:%m-%d %H:%M} {rid} ({why})")
        if thr == DETAIL:
            for rid, (r, pts, posted) in sorted(rp.items(), key=lambda kv: min(p[0] for p in kv[1][1])):
                t0 = min(p[0] for p in pts)
                fired_at = f"fired {first[rid]:%m-%d %H:%M}" if rid in first else "missed"
                post = f"posted {posted:%m-%d %H:%M}" if posted else "posted ?"
                print(f"   {rid:28s} {r.get('reporter', '')[:18]:18s} first point {t0:%m-%d %H:%M}, {post}: {fired_at}")
    base = caught_by[max(THRESHOLDS)]
    for thr in THRESHOLDS:
        extra = caught_by[thr] - base
        if extra:
            print(f"only at {thr} or below: " + ", ".join(sorted(extra)))


if __name__ == "__main__":
    args = sys.argv[1:]
    while args and args[0].startswith("--"):
        opt, val = args.pop(0).split("=", 1)
        if opt == "--thresholds":
            THRESHOLDS = [float(x) for x in val.split(",")]
        elif opt == "--detail":
            DETAIL = float(val)
    main(args)
