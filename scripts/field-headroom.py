"""The field animation with headroom: the sky stretched as the soft preset
does (linear, -2 to +10 sigma of the subtracted field), and everything
above that, which the preset pins to white, given the top of the gray
scale on its own linear stretch, so a bright extended thing (the earthlit
Moon crossing CCOR-1 on 2026-11-08) shows as a disk with shading instead
of a white hole. The comet and the stars keep the preset's look.

    docker compose run --rm -T -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro --entrypoint python kreutz scripts/field-headroom.py /data/render/20261107T1830_1900_1597 [--top 220] [--smooth 1.5]

-> <render>/style-soft/field-headroom/NNN.png and field-headroom.mp4
(H.264, even dimensions, 7 fps). --top is the gray level where the
preset's white used to be (the rest is the headroom); --smooth widens
the preset's smoothing for the frames, which helps a dim disk read.
"""
import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

warnings.simplefilter("ignore")

from kreutz import style  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("render", type=Path)
    ap.add_argument("--preset", default="soft")
    ap.add_argument("--top", type=int, default=150, help="gray level for the preset's old white (default 150)")
    ap.add_argument("--top-value", type=float, default=None,
                    help="field value that maps to white; default: the 90th percentile of what the preset pinned")
    ap.add_argument("--smooth", type=float, default=None, help="gaussian sigma in px instead of the preset's")
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--zoom", type=int, default=1, help="enlarge the frames this many times (Lanczos) before marking")
    args = ap.parse_args()
    p = dict(style.PRESETS[args.preset])
    if args.smooth is not None:
        p["smooth"] = args.smooth
    with fits.open(args.render / "cutouts.fits") as h:
        hdr = h[0].header
        field = h["FIELD_SUB"].data.astype(np.float32)
        tr = h["TRACK"].data
        times, xs, ys = tr["TIME"], tr["X"], tr["Y"]
    r0, c0 = hdr["ROW0"], hdr["COL0"]
    fp = style.process(field, p, (hdr["SUNROW"] - r0, hdr["SUNCOL"] - c0))
    lo, hi = style.limits(fp, p)
    over_all = fp[np.isfinite(fp) & (fp > hi)]
    top = args.top_value or (float(np.percentile(over_all, 90)) if over_all.size else hi)
    print(f"sky stretch {lo:.3g}..{hi:.3g} -> 0..{args.top}; headroom {hi:.3g}..{top:.3g} -> {args.top}..255 "
          f"({top / hi:.0f}x the old white)")
    out = args.render / f"style-{args.preset}" / ("field-headroom" if args.zoom == 1 else f"field-headroom-x{args.zoom}")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    for k in range(len(times)):
        a = fp[k]
        img = np.clip((a - lo) / (hi - lo), 0, 1) * args.top
        over = a > hi
        if top > hi:
            img[over] = args.top + (255 - args.top) * np.clip((a[over] - hi) / (top - hi), 0, 1)
        img8 = np.where(np.isfinite(a), img, 255).astype(np.uint8)[::-1]  # display north up like the preset
        im = Image.fromarray(img8)
        if args.zoom > 1:
            im = im.resize((im.width * args.zoom, im.height * args.zoom), Image.LANCZOS)
        im = im.convert("RGB")
        if p["mark"]:
            rr = a.shape[0] - 1 - ((hdr["NFULLROW"] - 1 - ys[k]) - r0)
            im = style.crosshair(im, (xs[k] - c0) * args.zoom, rr * args.zoom, args.zoom)
        style.stamp(im, f"{times[k][:10]} {times[k][11:16]} UTC", 255)
        im.save(out / f"{k:03d}.png")
    mp4 = out.with_suffix(".mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(out / "%03d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", "20",
           "-movflags", "+faststart", str(mp4)]
    try:
        subprocess.run(cmd, check=True)
        print(f"{len(times)} frames -> {out}/ and {mp4}")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"frames in {out}/; ffmpeg failed or is missing here ({e}); run on the host:\n  " + " ".join(cmd))


if __name__ == "__main__":
    sys.exit(main())
