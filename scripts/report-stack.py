"""Is a reported comet in our significance maps at all? Stack N frames along
the report's own straight-line motion, at every offset within +-R px of its
points, and print the stack at the points, the best within 3 px, and the
best anywhere in the box (a co-moving source nearby, e.g. a reporter's offset).

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor2 -w /srv -e PYTHONPATH=/srv --entrypoint python kreutz scripts/report-stack.py report_jr_20261024182048 ...

Noise in a 12-frame stack peaks around 3-3.8 within the 81x81 box; a comet
the live search catches stacks to 6+.
"""
import sys, warnings
from datetime import datetime
import numpy as np
warnings.simplefilter("ignore")
from kreutz.evaluate import report_points
from kreutz.frames import BG_FRAMES, BG_SKIP, CADENCE, STAR_FRAMES, Sequence, frame_paths
from kreutz.shortlist import all_reports
from kreutz.search import CLIP_SIGMA
R, N = 40, 12
reps = {r["id"]: r for r in all_reports()}
for rid in sys.argv[1:]:
    r = reps[rid]
    (r, pts, _), = report_points([r], datetime.strptime(r["posted_utc"], "%Y-%m-%dT%H:%M:%SZ").date())
    t = np.array([(p[0] - pts[0][0]) / CADENCE for p in pts])
    vx, x0 = np.polyfit(t, [p[1] for p in pts], 1); vy, y0 = np.polyfit(t, [p[2] for p in pts], 1)
    tm = pts[-1][0]
    seq = Sequence(frame_paths(tm - CADENCE * (N + BG_FRAMES + BG_SKIP + STAR_FRAMES + 2), tm + CADENCE / 2))
    H, W = seq.shape
    idx = list(range(len(seq.times) - N, len(seq.times)))
    acc = np.zeros((2 * R + 1, 2 * R + 1))
    for i in idx:
        k = (seq.times[i] - pts[0][0]) / CADENCE
        x, y = x0 + vx * k, y0 + vy * k
        r0, c0 = int(round(H - 1 - y)), int(round(x))
        s = np.clip(np.nan_to_num(seq.significance(i)), -CLIP_SIGMA, CLIP_SIGMA)
        acc += s[r0 - R:r0 + R + 1, c0 - R:c0 + R + 1]
    acc /= np.sqrt(N)
    flat = np.sort(acc.ravel())[::-1]
    rr, cc = np.unravel_index(np.argmax(acc), acc.shape)
    print(f"{rid} v=({vx:+.1f},{vy:+.1f}) {N} frames: at the points {acc[R, R]:.1f}, max within 3 px "
          f"{acc[R-3:R+4, R-3:R+4].max():.1f}; best {flat[0]:.1f} at offset ({cc - R:+d},{-(rr - R):+d}) px; 2nd-10th best {flat[1]:.1f}..{flat[9]:.1f}")
