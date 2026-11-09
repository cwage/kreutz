"""A close-up animation of something bright and extended crossing the
field (the earthlit Moon through CCOR-1, 2026-11-08 09:30-13:15 UTC),
from the raw level 1b frames at native resolution: each frame minus the
pixelwise minimum of the previous day (the static corona and glare
cancel), lightly smoothed, cropped to a box around the crossing,
stretched like scripts/field-headroom.py (sky linear to gray 150, the
rest of the gray scale for what's above that), and enlarged with Lanczos
so the disk is a few hundred pixels across in the video.

    docker compose run --rm -T -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro --entrypoint python kreutz scripts/moon-crossing.py --from 2026-11-08T09:00 --to 2026-11-08T13:45 --box 150 950 1560 2048 --out /data/render/moon-20261108

-> <out>/NNN.png and <out>.mp4 (ffmpeg on the host if the container
lacks it). --box is rows r0 r1 and columns c0 c1 of the crop in the
frame as stored (row 0 at the top of the array), --zoom the enlargement
(default 3), --smooth the gaussian sigma in px (default 0.8), --fps the
frame rate (default 4).
"""
import argparse
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

warnings.simplefilter("ignore")

from kreutz.frames import frame_paths, frame_time, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="start", type=datetime.fromisoformat, required=True)
    ap.add_argument("--to", dest="end", type=datetime.fromisoformat, required=True)
    ap.add_argument("--box", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--smooth", type=float, default=0.8)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--top", type=int, default=150, help="gray level where the sky stretch ends (default 150)")
    ap.add_argument("--background-hours", type=float, default=24.0)
    args = ap.parse_args()
    r0, r1, c0, c1 = args.box
    bg_paths = frame_paths(args.start - timedelta(hours=args.background_hours), args.start - timedelta(seconds=1))
    paths = frame_paths(args.start, args.end)
    if not bg_paths or not paths:
        raise SystemExit(f"frames: {len(bg_paths)} for the background, {len(paths)} to show; need both")
    bg = None
    for p in bg_paths:
        d = load(p)[0][r0:r1, c0:c1].astype(np.float32)
        bg = d if bg is None else np.fmin(bg, d)
    print(f"background: the minimum of {len(bg_paths)} frames", flush=True)
    subs = []
    for p in paths:
        d = load(p)[0][r0:r1, c0:c1].astype(np.float32) - bg
        bad = ~np.isfinite(d)
        s = gaussian_filter(np.nan_to_num(d), args.smooth) if args.smooth else np.nan_to_num(d)
        s[bad] = np.nan
        subs.append(s)
    cube = np.stack(subs)
    v = cube[np.isfinite(cube)]
    med = np.median(v)
    sig = np.median(np.abs(v - med)) / 0.6745
    lo, hi = med - 2 * sig, med + 10 * sig
    over = v[v > hi]
    top = float(np.percentile(over, 99.5)) if over.size else hi
    print(f"sky {lo:.3g}..{hi:.3g} -> 0..{args.top}; headroom to {top:.3g} -> 255 ({top / hi:.0f}x)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.png"):
        old.unlink()
    for k, (p, a) in enumerate(zip(paths, cube)):
        img = np.clip((a - lo) / (hi - lo), 0, 1) * args.top
        m = a > hi
        img[m] = args.top + (255 - args.top) * np.clip((a[m] - hi) / (top - hi), 0, 1)
        img8 = np.where(np.isfinite(a), img, 0).astype(np.uint8)[::-1]  # north up
        im = Image.fromarray(img8)
        im = im.resize((im.width * args.zoom, im.height * args.zoom), Image.LANCZOS).convert("RGB")
        ImageDraw.Draw(im).text((6, 4), f"{frame_time(p):%Y-%m-%d %H:%M} UTC", fill=(255, 255, 255))
        im.save(args.out / f"{k:03d}.png")
    mp4 = args.out.with_suffix(".mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(args.out / "%03d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", "18",
           "-movflags", "+faststart", str(mp4)]
    try:
        subprocess.run(cmd, check=True)
        print(f"{len(paths)} frames -> {args.out}/ and {mp4}")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"{len(paths)} frames in {args.out}/; ffmpeg failed or is missing here ({e}); run on the host:\n  "
              + " ".join(cmd))


if __name__ == "__main__":
    sys.exit(main())
