"""Replay the live watcher (kreutz.live's main loop, unchanged) over a past
stretch as it saw it: a simulated clock, only the frames that had landed
on disk by each moment (their file times), no downloads, notifications,
renders or report lookups, and its log, state and alerts in a scratch
directory. --restart-at stops it and starts it again from its saved state
at those times, as a container restart would. For a live decision that a
clean re-search doesn't reproduce (Ujfalussy's C2 comet of 2026-11-13: a
replay of the windows chained it by 08:48, the watcher never did).

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=c2 -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro --entrypoint python kreutz scripts/live-replay.py --from 2026-11-13T08:24:30 --to 2026-11-13T09:30 --restart-at 2026-11-13T08:29:40 --args "--n 4 --threshold 5.5"

The watcher's own output goes to stderr with the frames' times; its
files go under --out (default data/replay/<instrument>-<from>), which is
emptied first, with the instrument's known.jsonl copied in so the replay
remembers what the watcher remembered (as of now, not of then).
--chains-near X Y prints, after each window, the chains whose newest
candidate is within --near-r px (60) of (X, Y).
"""
import argparse
import shlex
import shutil
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.simplefilter("ignore")

from kreutz import DATA, live, page  # noqa: E402
from kreutz.instrument import CURRENT as INSTRUMENT  # noqa: E402


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="start", type=datetime.fromisoformat, required=True, help="UTC")
    ap.add_argument("--to", dest="end", type=datetime.fromisoformat, required=True, help="UTC")
    ap.add_argument("--restart-at", type=datetime.fromisoformat, nargs="*", default=[], help="UTC")
    ap.add_argument("--args", default="", help="the watcher's arguments, e.g. \"--n 4 --threshold 5.5\"")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--chains-near", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--near-r", type=float, default=60.0, help="px, for --chains-near (default 60)")
    args = ap.parse_args()
    out = args.out or DATA / "replay" / f"{INSTRUMENT.key}-{args.start:%Y%m%dT%H%M}"
    if out.exists():
        shutil.rmtree(out)
    (out / "alerts").mkdir(parents=True)
    if (live.LIVE / "known.jsonl").exists():
        shutil.copy(live.LIVE / "known.jsonl", out / "known.jsonl")

    clock = {"now": args.start}

    class SimDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            t = clock["now"]
            return t.replace(tzinfo=timezone.utc).astimezone(tz) if tz else t

    def landed(p):
        return datetime.fromtimestamp(Path(p).stat().st_mtime, timezone.utc).replace(tzinfo=None)

    real_frame_paths = live.frame_paths
    live.frame_paths = lambda a, b: [p for p in real_frame_paths(a, b) if landed(p) <= clock["now"]]
    live.datetime = SimDatetime
    live.sync = lambda since: 0
    live.latest_reports = lambda: []
    page.write = lambda: None
    live.LIVE = out
    state = out / "state.json"
    real_save, real_load = live.save_state, live.load_state
    live.save_state = lambda done, chains, pending, span, path=None: real_save(done, chains, pending, span, state)
    live.load_state = lambda now, span, path=None: real_load(now, span, state)

    if args.chains_near:
        import numpy as np
        real_write_status = live.write_status
        x0, y0 = args.chains_near

        def write_status(chains, pending, t, *a, **k):
            for ch in chains:
                c = ch[-1]
                if c.get("time", "")[:16] == f"{t:%Y-%m-%dT%H:%M}" and np.hypot(c["x"] - x0, c["y"] - y0) <= args.near_r:
                    print(f"   chain near ({x0:.0f},{y0:.0f}) at {t:%H:%M}: {len(ch)} windows, "
                          + " ".join(f"{d['time'][11:16]}({d['x']:.0f},{d['y']:.0f} v{d['vx']:+.1f},{d['vy']:+.1f} s{d['score']:.1f})"
                                     for d in ch), file=sys.stderr, flush=True)
            return real_write_status(chains, pending, t, *a, **k)
        live.write_status = write_status

    stops = sorted(args.restart_at) + [args.end]

    def sleep(s):
        clock["now"] += timedelta(seconds=s)
        if clock["now"] >= stops[0]:
            raise Stop

    live.time.sleep = sleep
    sys.argv = ["kreutz.live"] + shlex.split(args.args)
    while stops:
        print(f"== watcher starts at {clock['now']:%Y-%m-%d %H:%M:%S} UTC (runs to {stops[0]:%H:%M:%S})",
              file=sys.stderr, flush=True)
        try:
            live.main()
        except Stop:
            pass
        clock["now"] = stops.pop(0)
    print(f"== replay done; log, state and alerts in {out}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
