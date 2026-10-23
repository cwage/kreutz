"""Pull the instrument's frames from its real-time feed to disk.

    KREUTZ_INSTRUMENT=ccor2 docker compose run --rm -d kreutz kreutz.sync --loop

Downloads every frame the feed lists (it keeps about 30 days) that isn't
already under the instrument's frames directory (data/ccor2/l1a/...),
newest-first so a fresh start is useful for searching before it finishes
backfilling. The live watcher does this itself for the hours it needs;
this is for an instrument the watcher can't run on yet (one without a
Kreutz lane) and for frames too old for the watcher but still on the
feed, which the NCEI archive doesn't carry for CCOR-2 at all. With
--loop it keeps polling, so the rolling feed never ages out of reach.
The feed is SWPC's for CCOR and NASA's lastimage directory for LASCO
(lasco.py; it keeps about 10 days, and the archive lags by about three
weeks, so a LASCO watcher with a long gap behind it wants this running).
"""

import argparse
import re
import sys
import time
from datetime import datetime, timezone

from .frames import INSTRUMENT, L1B
from .live import FEED, POLL_S, _get


def listing(since=None):
    """[(time, url, destination)] for the frames on the feed, newest first;
    only those taken at `since` or later, if given (for LASCO that also
    skips listing the older days)."""
    if INSTRUMENT.source == "lasco":
        from . import lasco
        days = [d for d in lasco.day_dirs(FEED) if since is None or d >= since.date()]
        out = [f for d in days for f in lasco.day_frames(FEED, d)]
    else:
        pat = r'href="(' + INSTRUMENT.glob(r'\d{8}T\d{6}_V\d+').replace('.', r'\.') + ')"'
        out = []
        for n in set(re.findall(pat, _get(FEED).decode())):
            t = datetime.strptime(INSTRUMENT.name_re.search(n).group(1), "%Y%m%dT%H%M%S")
            out.append((t, FEED + n, L1B / f"{t:%Y/%m/%d}" / n))
    return sorted((f for f in out if since is None or f[0] >= since), reverse=True)


def pull(frames, get=_get):
    """Download the (time, url, destination) frames not on disk yet; how many were new."""
    new = 0
    for _, url, dest in frames:
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(get(url))
        tmp.rename(dest)
        new += 1
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loop", action="store_true", help=f"keep polling every {POLL_S} s")
    args = ap.parse_args()
    while True:
        try:
            names = listing()
            new = pull(names)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            print(f"{now:%Y-%m-%d %H:%M} {INSTRUMENT.label}: {len(names)} on the feed, {new} downloaded",
                  file=sys.stderr)
        except Exception as e:  # the feed hiccups; try again next poll
            print(f"feed error: {e}", file=sys.stderr)
        if not args.loop:
            break
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
