"""Download archived coronagraph frames from NOAA NCEI.

    docker compose run --rm kreutz kreutz.fetch 2025-07-17 [2025-07-19]

Full-resolution (2048x1920) calibrated frames, one every 15 minutes,
~9 MB each (~880 MB/day), into data/ccor1/l1b/YYYY/MM/DD/. Files already
present are skipped, so an interrupted day just resumes. KREUTZ_INSTRUMENT
picks the instrument (see instrument.py); CCOR-2 isn't in the archive yet.
"""

import argparse
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from .instrument import CURRENT as INSTRUMENT

ARCHIVE = "https://archive.data.noaa.gov/satellite-spaceweather"
PREFIX = INSTRUMENT.archive_prefix
WORKERS = 4


def _get(url, tries=5):
    req = urllib.request.Request(url, headers={"User-Agent": "kreutz"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except OSError:  # URLError and timeouts; NCEI's proxy has blips
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def day_keys(d):
    """Object keys for one day's L1b frames (S3 ListObjectsV2 via NCEI's proxy)."""
    keys, token = [], None
    while True:
        url = f"{ARCHIVE}?list-type=2&prefix={PREFIX}/{d:%Y/%m/%d}/"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token)
        xml = _get(url).decode()
        keys += re.findall(r"<Key>([^<]+\.fits)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m[1]


def _download(key):
    dest = INSTRUMENT.frames_dir / key.split(PREFIX + "/", 1)[1]
    if dest.exists():
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(_get(f"{ARCHIVE}/{key}"))
    tmp.rename(dest)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("start", type=date.fromisoformat)
    ap.add_argument("end", type=date.fromisoformat, nargs="?")
    args = ap.parse_args()
    if PREFIX is None:
        sys.exit(f"{INSTRUMENT.label} isn't in the NCEI archive; kreutz.live syncs its feed")
    d, end = args.start, args.end or args.start
    while d <= end:
        keys = day_keys(d)
        with ThreadPoolExecutor(WORKERS) as ex:
            new = sum(ex.map(_download, keys))
        print(f"{d}: {len(keys)} frames, {new} downloaded", file=sys.stderr)
        d += timedelta(days=1)


if __name__ == "__main__":
    main()
