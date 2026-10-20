"""Strip of crops along a hypothesized track, for vetting by eye.

    docker compose run --rm kreutz kreutz.look 2025-07-17T07:15 1909 714 -5 -10.5 [--back 7 --ahead 2]

Top row: raw frames (stretched); bottom row: significance maps. Each crop
is centered on where the track puts the object in that frame, so a real
mover sits still in the middle while stars slide across. Written to
data/look/<time>_<x>_<y>.png.
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

from . import DATA
from .frames import BG_FRAMES, BG_SKIP, INSTRUMENT, STAR_FRAMES, Sequence, frame_paths

HALF = 40


def strip(seq, i_ref, x, y, vx, vy, back, ahead):
    cols = []
    for i in range(i_ref - back, i_ref + ahead + 1):
        if not 0 <= i < len(seq.paths):
            continue
        k = i - i_ref
        r, c = seq.report_to_rc(x + vx * k, y + vy * k)
        r, c = int(round(r)), int(round(c))
        tiles = []
        for img, hi in ((seq.raw(i), 99.5), (seq.significance(i), 99.7)):
            crop = img[max(r - HALF, 0):r + HALF + 1, max(c - HALF, 0):c + HALF + 1]
            if crop.shape != (2 * HALF + 1, 2 * HALF + 1):
                crop = np.full((2 * HALF + 1, 2 * HALF + 1), np.nan)
            v = np.nanpercentile(crop, [1, hi]) if np.isfinite(crop).any() else (0, 1)
            t = np.clip((np.nan_to_num(crop, nan=v[0]) - v[0]) / (v[1] - v[0] + 1e-30) * 255, 0, 255)
            t = np.flipud(t.astype(np.uint8))
            t[HALF, HALF - 9:HALF - 5] = 255
            t[HALF, HALF + 6:HALF + 10] = 255
            tiles.append(t)
        cols.append(np.vstack([tiles[0], np.full((3, tiles[0].shape[1]), 128, np.uint8), tiles[1]]))
    sep = np.full((cols[0].shape[0], 3), 255, np.uint8)
    return np.hstack(sum([[c, sep] for c in cols], [])[:-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("time", type=datetime.fromisoformat)
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("vx", type=float)
    ap.add_argument("vy", type=float)
    ap.add_argument("--back", type=int, default=7)
    ap.add_argument("--ahead", type=int, default=2)
    args = ap.parse_args()
    warm = timedelta(minutes=15) * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.back + 2)
    seq = Sequence(frame_paths(args.time - warm, args.time + timedelta(minutes=15) * (args.ahead + 1)))
    i_ref = seq.times.index(args.time.replace(second=INSTRUMENT.second))
    img = strip(seq, i_ref, args.x, args.y, args.vx, args.vy, args.back, args.ahead)
    out = DATA / f"look{INSTRUMENT.tag}" / f"{args.time:%Y%m%dT%H%M}_{args.x:.0f}_{args.y:.0f}.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(np.kron(img, np.ones((2, 2), np.uint8))).save(out)
    print(out)


if __name__ == "__main__":
    main()
