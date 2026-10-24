"""Run the search over every window in a span of archived frames.

    docker compose run --rm kreutz kreutz.run 2025-07-17 [2025-07-17T12:00] [--n 4]

Searches windows whose last frame is in [start, end) (end defaults to a
day after start; split a day into chunks to run them in parallel), and
writes every candidate above a loose threshold to
data/runs/n<N>/<start>_<end>.jsonl, one per line with the window's last
frame time, so detection thresholds and linking rules can be tuned
afterwards without searching again. The frames need a few hours of
history before the first searchable window, so the run starts from the
previous day's evening frames when they're on disk.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta

from . import DATA
from .frames import BG_FRAMES, BG_SKIP, CADENCE, INSTRUMENT, STAR_FRAMES, Sequence, frame_paths
from .search import search

LOOSE = 4.5
ORBIT_LOOSE = 3.5  # orbitsearch: its noise peaks reach ~4.6 per window
LANE_LOOSE = 4.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("start", type=lambda s: datetime.fromisoformat(s))
    ap.add_argument("end", type=lambda s: datetime.fromisoformat(s), nargs="?")
    ap.add_argument("--n", type=int, default=4, help="frames per window")
    ap.add_argument("--lane", action="store_true",
                    help="lane-constrained search (lane.py) instead of the full search")
    ap.add_argument("--orbit", action="store_true",
                    help="orbit-constrained search (orbitsearch.py) instead of the full search")
    args = ap.parse_args()
    end = args.end or args.start + timedelta(days=1)
    warm = CADENCE * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.n)
    seq = Sequence(frame_paths(args.start - warm, end))
    first = next(i for i, t in enumerate(seq.times) if t >= args.start)
    first = max(first, BG_FRAMES + BG_SKIP + args.n)
    kind = f"lane{args.n}" if args.lane else f"orbit{args.n}" if args.orbit else f"n{args.n}"
    out = DATA / f"runs{INSTRUMENT.tag}" / kind / f"{args.start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.jsonl"
    field = None
    if args.lane:
        from .lane import build_field
        field = build_field(seq.shape, when=args.start)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for last in range(first, len(seq.paths)):
            if seq.times[last] >= end:
                break
            t0 = time.time()
            seq.forget(last - args.n - BG_FRAMES - BG_SKIP - STAR_FRAMES - 4)
            try:
                if args.lane:
                    from .lane import lane_search
                    cands = lane_search(seq, last, args.n, LANE_LOOSE, field)
                elif args.orbit:
                    from .orbitsearch import orbit_search
                    cands = orbit_search(seq, last, args.n, ORBIT_LOOSE)
                else:
                    cands = search(seq, last, args.n, LOOSE)
            except ValueError as e:  # a gap in the data, too few stars, ...
                print(f"{seq.times[last]:%Y-%m-%d %H:%M}: skipped ({e})", file=sys.stderr)
                continue
            for c in cands:
                rec = asdict(c)
                rec["time"] = seq.times[last].isoformat()
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{seq.times[last]:%Y-%m-%d %H:%M}: {len(cands)} candidates "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
