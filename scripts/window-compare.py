"""What a second orbit search with longer windows adds to the live one:
earlier or extra catches of reported Kreutz comets, and the pages it costs.

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor1 -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro --entrypoint python kreutz scripts/window-compare.py --base orbit8 --extra orbit16 --from 2026-10-01 --to 2026-11-02 --base-thr 6.0 --extra-thr 6.5,7.0,7.5

Runs are data/runs<tag>/<kind>/*.jsonl (kreutz.run --orbit --n N, or
scripts/orbit-archive.sh). Each kind is replayed as threshold-replay.py
does (live.LIVE_SPEED_MIN, evaluate.tracks linking LINK windows, the
stack-line veto); the base at its threshold stands for the live search,
the extra at each threshold is added to it. Tracks are matched to the
instrument's reports with two or more points (any group, less retracted
ones) by evaluate.matches or shortlist._known_by_position; Kreutz reports
are the comets counted (once per comet with the reports amending them),
a match to another group's report is a known comet, not a false page.
Prints, per comet, its first point, when it was posted, and when each
search first fired; then the extra's tracks that match no report and
fired where the base had nothing within SAME_PX and SAME_H, the pages it
would add (live.pageworthy), with time and place to look at.
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np

warnings.simplefilter("ignore")

from kreutz import DATA  # noqa: E402
from kreutz.evaluate import FRAME, amended, load, matches, report_points, retracted, tracks  # noqa: E402
from kreutz.frames import BG_FRAMES, BG_SKIP, STAR_FRAMES, Sequence, frame_paths  # noqa: E402
from kreutz.instrument import CURRENT as INSTRUMENT  # noqa: E402
from kreutz.lane import field_or_none  # noqa: E402
from kreutz.live import LIVE_SPEED_MIN, pageworthy  # noqa: E402
from kreutz.shortlist import FRAME_ROWS, _known_by_position, all_reports  # noqa: E402
from kreutz.vet import line_stack  # noqa: E402

LINK = 3
SAME_PX, SAME_H = 40.0, 6.0  # an extra track this near a base track (any), carried along its motion, is the same object


def runs(kind, start, end):
    d = DATA / f"runs{INSTRUMENT.tag}" / kind
    paths = []
    for p in sorted(d.glob("*.jsonl")):
        a, b = (datetime.strptime(s, "%Y%m%dT%H%M") for s in p.stem.split("_"))
        if a >= start and b <= end and b - a >= timedelta(hours=23):  # whole days only
            paths.append(p)
    by_time = load(paths)
    return {t: [c for c in cs if np.hypot(c["vx"], c["vy"]) >= LIVE_SPEED_MIN] for t, cs in by_time.items()
            if start <= t < end}


class Seqs:
    """One day's frames at a time, for the stack-line veto."""
    def __init__(self):
        self.d, self.seq = None, None

    def __call__(self, t):
        if t.date() != self.d:
            s = datetime.combine(t.date(), datetime.min.time())
            warm = FRAME * (BG_FRAMES + BG_SKIP + STAR_FRAMES + 10)
            self.d, self.seq = t.date(), Sequence(frame_paths(s - warm, s + timedelta(days=1, hours=1)))
        return self.seq


def same(f, g):
    """Is extra track f (where it fired) the object base track g fired on,
    g carried along its motion to f's time?"""
    dt = (f["t"] - g["t"]) / FRAME
    if abs(dt) * FRAME > timedelta(hours=SAME_H):
        return False
    return np.hypot(f["x"] - (g["x"] + g["vx"] * dt), f["y"] - (g["y"] + g["vy"] * dt)) <= SAME_PX


def fired(by_time, thr, reps, field, seqs):
    """[(fire candidate, [report ids it matches])] for the tracks that fire
    at thr and survive the stack-line veto, in time order."""
    out = []
    for tr in sorted(tracks(by_time, thr, LINK), key=lambda tr: tr[LINK - 1]["t"]):
        f = tr[LINK - 1]
        try:
            seq = seqs(f["t"])
            if line_stack(seq, seq.times.index(f["t"]), f["x"], f["y"], f["vx"], f["vy"]):
                continue
        except (ValueError, IndexError):
            pass
        hits = [rid for rid, (r, pts) in reps.items() if matches(tr, pts) or _known_by_position(tr, [(r, pts)], field)]
        out.append((f, hits))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="orbit8")
    ap.add_argument("--extra", default="orbit16")
    ap.add_argument("--from", dest="start", type=datetime.fromisoformat, required=True)
    ap.add_argument("--to", dest="end", type=datetime.fromisoformat, required=True)
    ap.add_argument("--base-thr", type=float, default=6.0)
    ap.add_argument("--extra-thr", default="6.5,7.0,7.5")
    args = ap.parse_args()
    base, extra = runs(args.base, args.start, args.end), runs(args.extra, args.start, args.end)
    days = sorted({t.date() for t in base} | {t.date() for t in extra})
    allr = all_reports()
    gone = retracted(allr)
    reps, kreutz = {}, {}
    for d in days:
        for r, pts, posted in report_points(allr, d):
            if len(pts) >= 2 and r["id"] not in gone and any(args.start <= t < args.end for t, _, _ in pts):
                reps[r["id"]] = (r, pts)
                if r.get("group") == "Kreutz":
                    kreutz[r["id"]] = (r, pts, posted)
    root = amended({k: (r, pts) for k, (r, pts, _) in kreutz.items()})
    comet = lambda rid: root.get(rid, rid)  # noqa: E731
    field = field_or_none((FRAME_ROWS, INSTRUMENT.shape[1]))
    sun = INSTRUMENT.sun_xy
    seqs = Seqs()
    print(f"{INSTRUMENT.label} {args.start:%m-%d}..{args.end:%m-%d}: {args.base} {len(base)} windows with candidates, "
          f"{args.extra} {len(extra)}; {len({comet(k) for k in kreutz})} Kreutz comets reported, "
          f"{len(reps) - len(kreutz)} other reports")

    def first(fs):
        out = {}
        for f, hits in fs:
            for rid in hits:
                if rid in kreutz:
                    c = comet(rid)
                    out[c] = min(out.get(c, f["t"]), f["t"])
        return out

    def unexplained(fs):
        return [f for f, hits in fs if not hits]

    b = fired(base, args.base_thr, reps, field, seqs)
    bfirst, bfalse = first(b), unexplained(b)
    bpages = [f for f in bfalse if pageworthy(f, sun)]
    print(f"base {args.base} at {args.base_thr}: {len(bfirst)} comets caught, {len(bpages)} unexplained pages "
          f"({len(bpages) / len(days):.2f}/day)")
    for thr in [float(x) for x in args.extra_thr.split(",")]:
        e = fired(extra, thr, reps, field, seqs)
        efirst = first(e)
        new = [f for f in unexplained(e) if not any(same(f, g) for g, _ in b)]
        added = [f for f in new if pageworthy(f, sun)]
        union = {c: min(t for t in (bfirst.get(c), efirst.get(c)) if t) for c in set(bfirst) | set(efirst)}
        gained = {c: (bfirst[c] - union[c]) for c in union if c in bfirst and union[c] < bfirst[c]}
        print(f"\n+ {args.extra} at {thr}: union catches {len(union)} (base {len(bfirst)}); "
              f"{len(gained)} earlier, by {', '.join(f'{g.total_seconds() / 3600:.1f}h' for g in sorted(gained.values(), reverse=True)) or '-'}; "
              f"adds {len(added)} pages ({len(added) / len(days):.2f}/day) + {len(new) - len(added)} quiet")
        for c in sorted(set(union) | {comet(k) for k in kreutz}, key=lambda c: min(p[0] for p in kreutz[c][1])):
            r, pts, posted = kreutz[c]
            t0 = min(p[0] for p in pts)
            bt, et = bfirst.get(c), efirst.get(c)
            mark = " <- extra first" if et and (not bt or et < bt) else ""
            print(f"   {c:26s} {r.get('reporter', '')[:16]:16s} first point {t0:%m-%d %H:%M}, "
                  f"posted {posted:%m-%d %H:%M}" if posted else f"   {c:26s} first point {t0:%m-%d %H:%M}", end="")
            print(f": base {bt:%m-%d %H:%M}" if bt else ": base -", end="")
            print(f", extra {et:%m-%d %H:%M}{mark}" if et else ", extra -")
        for f in added:
            print(f"   added page {f['t']:%m-%d %H:%M} at ({f['x']:.0f},{f['y']:.0f}) v ({f['vx']:+.1f},{f['vy']:+.1f}) "
                  f"score {f['score']:.1f}")


if __name__ == "__main__":
    sys.exit(main())
