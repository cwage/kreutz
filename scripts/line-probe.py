"""Score alerts for a straight line through the track in its own stack
(vet.through_line): the CCOR-2 stray-light streak that lies along the
Kreutz lane, slides along itself, and gets tracked as a point.

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor2 -w /srv -e PYTHONPATH=/srv --entrypoint python kreutz scripts/line-probe.py [ALERT_NAME ...]

With no names, every alert of the instrument. Prints, per alert, the
best line score (the weaker side's mean significance along a line
through the track's position) and its angle off the motion.
"""

import json
import sys
import warnings
from datetime import datetime

warnings.simplefilter("ignore")

from kreutz import vet  # noqa: E402
from kreutz.frames import BG_FRAMES, BG_SKIP, CADENCE, STAR_FRAMES, Sequence, frame_paths  # noqa: E402
from kreutz.live import LIVE  # noqa: E402


def main():
    names = sys.argv[1:] or sorted(p.stem for p in (LIVE / "alerts").glob("2026*.json")
                                   if "." not in p.stem)
    for n in names:
        d = json.load(open(LIVE / "alerts" / f"{n}.json"))
        if isinstance(d, list):  # the earliest alerts were the bare track
            d = {"track": d}
        c = d["track"][-1]
        t = datetime.fromisoformat(c["time"])
        warm = CADENCE * (BG_FRAMES + BG_SKIP + STAR_FRAMES + vet.STACK_N + 2)
        seq = Sequence(frame_paths(t - warm, t + CADENCE / 2))
        last = min(range(len(seq.times)), key=lambda i: abs(seq.times[i] - t))
        try:
            score, ang = vet.through_line(seq, last, c["x"], c["y"], c["vx"], c["vy"])
        except Exception as e:
            print(f"{n}: {e}")
            continue
        k = (d.get("known") or {}).get("reporter", "")
        print(f"{n:28s} line {score:5.2f} at {ang:3.0f} deg  paged={d.get('paged')} {k}", flush=True)


if __name__ == "__main__":
    main()
