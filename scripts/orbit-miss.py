"""Why the orbit search missed a reported comet: is its orbit outside the
family searched, or is it too faint along the orbit it does have?

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor2 -w /srv -e PYTHONPATH=/srv --entrypoint python kreutz scripts/orbit-miss.py report_eb_20261013095017 ...

Fits an orbit (perihelion time, node, argument of perihelion) to the
report's points, says how far it is from the family (orbitsearch.NODES,
PERI_A + PERI_B * node, PERI_OFFSETS), and stacks the significance maps
along the fitted orbit in 8-frame windows from 3 h before the first point
to the last, as orbit_search does, giving the best window's score and its
per-frame values. A comet the family covers and that stacks above the
threshold there was lost in linking or the alert rules, not the search.
"""

import sys
import warnings
from datetime import datetime, timedelta

import numpy as np

warnings.simplefilter("ignore")

from kreutz import orbit, orbitsearch  # noqa: E402
from kreutz.evaluate import report_points  # noqa: E402
from kreutz.frames import BG_FRAMES, BG_SKIP, CADENCE, STAR_FRAMES, Sequence, frame_paths  # noqa: E402
from kreutz.instrument import CURRENT as INSTRUMENT  # noqa: E402
from kreutz.search import CLIP_SIGMA  # noqa: E402
from kreutz.shortlist import all_reports  # noqa: E402

N = 8
BEFORE = timedelta(hours=3)


def sky(paths, times, pts):
    """Helioprojective (deg) of report points, from the nearest frame's WCS."""
    tx, ty, ts = [], [], []
    for t, x, y in pts:
        j = min(range(len(times)), key=lambda i: abs(times[i] - t))
        if abs(times[j] - t) > CADENCE / 2:
            continue
        w, H, _ = orbitsearch._wcs(paths[j])
        lon, lat = w.wcs_pix2world([[x, H - 1 - y]], 0)[0]
        tx.append(lon)
        ty.append(lat)
        ts.append(t)
    return ts, tx, ty


def main(ids):
    reps = {r["id"]: r for r in all_reports()}
    where = "l1" if INSTRUMENT.key == "ccor2" else "earth"
    for rid in ids:
        r = reps[rid]
        (r, pts, posted), = report_points([r], datetime.strptime(r["posted_utc"], "%Y-%m-%dT%H:%M:%SZ").date())
        t0, t1 = pts[0][0], pts[-1][0]
        warm = CADENCE * (BG_FRAMES + BG_SKIP + STAR_FRAMES + N)
        seq = Sequence(frame_paths(t0 - BEFORE - warm, t1 + CADENCE))
        ts, tx, ty = sky(seq.paths, seq.times, pts)
        if len(ts) < 2:
            print(f"{rid}: points not on frames on disk")
            continue
        start = orbit.Elements(T=ts[-1] + timedelta(hours=6), i=orbitsearch.INCLINATION)
        e0, _ = orbit.fit(ts, tx, ty, start, where, free=("T",))
        free = ("T", "node", "peri") if len(ts) >= 3 else ("T", "node")
        e, rms = orbit.fit(ts, tx, ty, e0, where, free=free)
        off = e.peri - (orbitsearch.PERI_A + orbitsearch.PERI_B * e.node)
        inside = orbitsearch.NODES.min() - 0.5 <= e.node <= orbitsearch.NODES.max() + 0.5 \
            and abs(off) <= max(orbitsearch.PERI_OFFSETS) + 1
        best = (-np.inf, None, None)
        H, W = seq.shape
        for last in range(len(seq.times)):
            if not (t0 - BEFORE <= seq.times[last] <= t1) or last < BG_FRAMES + BG_SKIP + N:
                continue
            idx = [i for i in range(last, -1, -1) if (seq.times[last] - seq.times[i]) < CADENCE * (N - 0.5)]
            vals = []
            for i in reversed(idx):
                w, _, _ = orbitsearch._wcs(seq.paths[i])
                mx, my = orbit.helioprojective(e, [seq.times[i]], where)
                col, row = w.wcs_world2pix(mx, my, 0)
                ir, ic = int(round(row[0])), int(round(col[0]))
                s = 0.0
                if 0 <= ir < H and 0 <= ic < W:
                    s = float(np.clip(np.nan_to_num(seq.significance(i)[ir, ic]), -CLIP_SIGMA, CLIP_SIGMA))
                vals.append(s)
            score = sum(vals) / np.sqrt(len(vals))
            if score > best[0]:
                best = (score, seq.times[last], vals)
        score, t, vals = best
        print(f"{rid} {r.get('reporter', '')[:16]:16s} {len(ts)} pts: node {e.node:5.1f} peri {e.peri:5.1f} "
              f"(off family {off:+.1f}) rms {rms:4.0f}\" {'in family' if inside else 'OUTSIDE family'}; "
              f"best window along it {score:.1f} at {t:%m-%d %H:%M} {[round(v, 1) for v in vals]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
