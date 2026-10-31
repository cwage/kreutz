"""Render a reported comet along the report's own points, for comets too
faint for kreutz.render to follow by detection.

    docker compose run --rm -T -w /srv -e PYTHONPATH=/srv --entrypoint python kreutz scripts/render-report.py report_nu_20261014123802 [--back 16] [--ahead 24] [--name NAME]

Fits a straight line in time to the report's (x, y) points and lays the
track along it from --back frames before the first point to --ahead after
the last, then writes the same products as kreutz.render (track.json,
cutouts.fits, preview.gif) into data/render*/<name>, which style.py turns
into animations. Every frame is "not seen": the positions are the
report's motion, not measurements.
"""

import argparse
import json
import warnings
from datetime import datetime

import numpy as np

warnings.simplefilter("ignore")

from kreutz import DATA  # noqa: E402
from kreutz.evaluate import report_points  # noqa: E402
from kreutz.frames import BG_FRAMES, BG_SKIP, CADENCE, INSTRUMENT, STAR_FRAMES, Sequence, frame_paths  # noqa: E402
from kreutz.render import cutouts, preview, write_fits  # noqa: E402
from kreutz.shortlist import all_reports  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report")
    ap.add_argument("--back", type=int, default=16, help="frames before the first point (default 4 h)")
    ap.add_argument("--ahead", type=int, default=24, help="frames after the last point (default 6 h)")
    ap.add_argument("--name")
    args = ap.parse_args()
    r = {r["id"]: r for r in all_reports()}[args.report]
    posted = datetime.strptime(r["posted_utc"], "%Y-%m-%dT%H:%M:%SZ")
    (r, pts, _), = report_points([r], posted.date())
    t0, t1 = pts[0][0], pts[-1][0]
    warm = CADENCE * (BG_FRAMES + BG_SKIP + STAR_FRAMES + 2)
    seq = Sequence(frame_paths(t0 - CADENCE * args.back - warm, t1 + CADENCE * (args.ahead + 1)))
    H, W = seq.shape
    tp = np.array([(t - t0) / CADENCE for t, _, _ in pts])
    fx = np.polyfit(tp, [p[1] for p in pts], 1) if len(pts) > 1 else None
    fy = np.polyfit(tp, [p[2] for p in pts], 1) if len(pts) > 1 else None
    if fx is None:
        raise SystemExit(f"{args.report}: one point, no motion to follow")
    track = {}
    for i, t in enumerate(seq.times):
        if i < BG_FRAMES + BG_SKIP + 2:
            continue
        k = (t - t0) / CADENCE
        if not -args.back <= k <= (t1 - t0) / CADENCE + args.ahead:
            continue
        x, y = float(np.polyval(fx, k)), float(np.polyval(fy, k))
        if 0 <= x < W and 0 <= y < H:
            track[i] = (x, y, None)
    if not track:
        raise SystemExit(f"{args.report}: no frames on disk along its path")
    idx = sorted(track)
    out = DATA / f"render{INSTRUMENT.tag}" / (args.name or args.report)
    out.mkdir(parents=True, exist_ok=True)
    (out / "track.json").write_text(json.dumps(
        [{"time": seq.times[i].isoformat(), "x": track[i][0], "y": track[i][1], "sigma": None} for i in idx],
        indent=1))
    cubes, origin = cutouts(seq, track)
    write_fits(out / "cutouts.fits", seq, track, cubes, origin)
    preview(cubes[1], [seq.times[i] for i in idx], out / "preview.gif")
    print(f"{out}: {len(idx)} frames {seq.times[idx[0]]:%Y-%m-%d %H:%M} to {seq.times[idx[-1]]:%H:%M} "
          f"along the report's motion ({fx[0]:+.1f}, {fy[0]:+.1f}) px/frame; "
          f"field {cubes[2].shape[1]}x{cubes[2].shape[2]}")


if __name__ == "__main__":
    main()
