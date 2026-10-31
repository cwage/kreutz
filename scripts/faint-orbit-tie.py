"""Does one Kreutz family orbit go through a report's points and a later
track (one comet), and where does the report's own orbit put the comet
at later times? For checking a "matched by position" tie
(notes/faint-comet-misses.md).

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor2 -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro --entrypoint python kreutz scripts/faint-orbit-tie.py report_jr_20261024182048 --track 2026-10-25T05:15,1579,1590 2026-10-25T05:30,1572,1585 2026-10-25T05:45,1565,1579 [--track ...] [--predict 2026-10-24T22:00 2026-10-25T05:30 ...]

Prints the report's own family-orbit rms, the joint rms of each --track
group (one --track flag per group) with the report's points
(orbitsearch.joint_rms; under SAME_RMS = 400" means one comet), and the
report-orbit's predicted (x, y) at each --predict time.
"""

import argparse
import sys
import warnings
from datetime import datetime

warnings.simplefilter("ignore")

from kreutz import orbit  # noqa: E402
from kreutz.evaluate import report_points  # noqa: E402
from kreutz.orbitsearch import SAME_RMS, _where, joint_rms, orbit_of  # noqa: E402
from kreutz.frames import Sequence, frame_paths, CADENCE  # noqa: E402
from kreutz.shortlist import all_reports  # noqa: E402


def pt(s):
    t, x, y = s.split(",")
    return datetime.fromisoformat(t), float(x), float(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report")
    ap.add_argument("--track", nargs="+", action="append", default=[], help="time,x,y points of one track")
    ap.add_argument("--predict", nargs="*", default=[])
    args = ap.parse_args()
    r = {r["id"]: r for r in all_reports()}[args.report]
    (r, pts, _), = report_points([r], datetime.strptime(r["posted_utc"], "%Y-%m-%dT%H:%M:%SZ").date())
    fit = orbit_of(pts)
    if fit is None:  # fewer than three of the report's points have a frame on disk
        sys.exit(f"{args.report}: {len(pts)} points, but the family orbit needs three or more with "
                 f"their frames on disk (kreutz.fetch or kreutz.sync the day first)")
    e, rms = fit
    print(f"{args.report}: family orbit through its {len(pts)} points rms {rms:.0f}\" "
          f"(node {e.node:.1f} peri {e.peri:.1f} T {e.T:%m-%d %H:%M})")
    for group in args.track:
        trk = [pt(s) for s in group]
        own = orbit_of(trk)
        j = joint_rms(own[0], trk, pts) if own else float("inf")
        print(f"  track {trk[0][0]:%m-%d %H:%M}..{trk[-1][0]:%H:%M} at ({trk[-1][1]:.0f},{trk[-1][2]:.0f}): "
              f"own orbit rms {own[1] if own else float('nan'):.0f}\", joint with the report {j:.0f}\" "
              f"-> {'one comet' if j < SAME_RMS else 'not one orbit'} (SAME_RMS {SAME_RMS:.0f})")
    for s in args.predict:
        t = datetime.fromisoformat(s)
        seq = Sequence(frame_paths(t - CADENCE / 2, t + CADENCE / 2))
        from kreutz.orbitsearch import _wcs
        w, H, _ = _wcs(seq.paths[0])
        lon, lat = orbit.helioprojective(e, [t], _where())
        col, row = w.wcs_world2pix(lon, lat, 0)
        print(f"  report orbit at {t:%m-%d %H:%M}: ({col[0]:.1f}, {H - 1 - row[0]:.1f})")


if __name__ == "__main__":
    main()
