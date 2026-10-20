"""Contact sheets of vetting strips for a list of candidates.

    docker compose run --rm kreutz kreutz.sheet /data/shortlist/to-vet.json [--per 6]

One look.py strip per candidate (7 frames before to 5 after, crops
following the candidate's motion; raw on top, significance below), several
per sheet, written to data/look/sheet-NN.png. Candidates are grouped by
day so each day's frames are loaded once.
"""

import argparse
import json
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw

from . import DATA
from .frames import BG_FRAMES, BG_SKIP, INSTRUMENT, STAR_FRAMES, Sequence, frame_paths
from .look import strip

BACK, AHEAD = 7, 5


def make_sheets(cands, out, per=6):
    """Contact sheets for candidate records (time, x, y, vx, vy, and
    optionally persistence/known/source), written to out/sheet-NN.png in
    the order given. Returns the paths."""
    cands = [dict(c) for c in cands]
    for n, c in enumerate(cands):
        c["n"] = n
        c["t"] = datetime.fromisoformat(c["time"][:19])
    rows = []
    seq, seq_day = None, None
    for c in sorted(cands, key=lambda c: c["t"]):
        if seq_day != c["t"].date():
            day = datetime.combine(c["t"].date(), datetime.min.time())
            warm = timedelta(minutes=15) * (BG_FRAMES + BG_SKIP + STAR_FRAMES + BACK + 2)
            seq = Sequence(frame_paths(day - warm, day + timedelta(days=1, hours=3)))
            seq_day = c["t"].date()
        try:
            i = seq.times.index(c["t"])
            img = strip(seq, i, c["x"], c["y"], c["vx"], c["vy"], BACK, AHEAD)
        except (ValueError, IndexError) as e:
            print(f"#{c['n']} {c['time'][:16]}: skipped ({e})")
            continue
        tag = (f"pers {c['persistence']:+.1f}" if "persistence" in c and c["persistence"] is not None
               else c.get("known", ""))
        label = Image.new("L", (img.shape[1], 14), 0)
        ImageDraw.Draw(label).text(
            (2, 1), f"#{c['n']}  {c['time'][:16]}  ({c['x']:.0f},{c['y']:.0f})  "
                    f"v=({c['vx']:+.1f},{c['vy']:+.1f})  {c.get('source', '')}  {tag}", fill=255)
        rows.append((c["n"], np.vstack([np.asarray(label), img])))
    rows.sort()
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(rows), per):
        group = [r for _, r in rows[s:s + per]]
        w = max(g.shape[1] for g in group)
        group = [np.pad(g, ((0, 6), (0, w - g.shape[1])), constant_values=255) for g in group]
        path = out / f"sheet-{s // per:02d}.png"
        Image.fromarray(np.vstack(group)).save(path)
        print(path, f"#{rows[s][0]}-#{rows[min(s + per, len(rows)) - 1][0]}")
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidates")
    ap.add_argument("--per", type=int, default=6)
    args = ap.parse_args()
    make_sheets(json.load(open(args.candidates)), DATA / f"look{INSTRUMENT.tag}", args.per)


if __name__ == "__main__":
    main()
