"""Is a faint reported comet in the plain difference images, and which step
of the significance-map processing loses it? (notes/faint-comet-misses.md)

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor2 -e OMP_NUM_THREADS=3 -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro -v $PWD/data/probe:/out --entrypoint python kreutz scripts/faint-probe.py report_jr_20261024182048 [--extra 0515,1579,1590 ...] [--deg 2] [--from 1800 --to 0545] [--n 8] [--out /out]

The track is a polynomial in time (--deg, default 1) fitted to the report's
points plus any --extra points (HHMM,x,y in report coordinates, on the day
after the report's first point when HHMM is earlier than it), or with
--orbit a Kreutz orbit (T, node, peri free, from orbitsearch.orbit_of)
through them, projected through each frame's WCS; --only-extra drops the
report's own points (to follow a later alert's track, say). For every
frame from --from to --to (default: the report's points, 8 frames before
the first to 4 after the last) it cuts a region around the track out of
the raw frames (dark-filled as the pipeline does), subtracts the median
of the previous 16 (skipping 1), removes particle hits (CCOR-2), and
builds these per-frame maps, each divided by a pipeline-style local noise
(41 px box of |f|, valid pixels only) and clipped at +-5:

    pipe      the cached live significance map, as the search sees it
    dog12     DoG 1.2/4.0 after the star model (the pipeline, without its
              star mask, row/column flatten and blotch mask)
    dog12ns   DoG 1.2/4.0 with no star removal
    dog2      DoG 2.0/6.0 after the star model
    dog3      DoG 3.0/10  after the star model
    dog4      DoG 4.0/14  after the star model
    bin2      2x2-binned difference, DoG 1.2/4.0 at the binned scale
              (~2.4/8 px), after the star model
    plain     no star removal: Gaussian 1.5 px minus a 41 px median (a
              smoothed plain difference with the slow glow taken out)

Each map is shift-and-stacked along the track in sliding --n frame windows
(sum / sqrt(n), integer shifts as search.py does). For every window and
map it prints the stack at the track (max within 2 px) and the stack's
empirical noise at the same motion: the std and the max of the stack over
+-100 px offsets outside 12 px of the track, so "z_emp" = value / std and
"rank" = value / noise max compare variants with different smoothing
honestly (correlated noise, stars, residue all count). For the window
ending at the last report point it also scans velocity +-1 px/frame in
0.5 steps and compares the best value within 3 px of the track with the
same max taken over 7x7 cells across the box (the look-elsewhere at equal
trials). --end HHMM picks another window for that scan and the pictures.
It also prints that window's per-frame values at the track (max within
1 px), where a star-masked frame of the live map shows as exactly 0.0.

Outputs are for looking at and comparing, not a detector: the variants
skip the live map's flatten and masks, and the noise box holds stars and
residue, so "rank" is a local look-elsewhere, not a false-alarm rate.

PNGs into --out/<--name or report>/ (--pics HHMM ... picks the frames): frames-*.png (per-frame crops +-40 px, 3x, at the
report times: plain difference smoothed 1.5 px, 2x2 binned, star-removed,
and the live map) and stack-*.png (stacks along the track of each map),
centers marked with ticks. Nothing in data/ is written except what
Sequence.significance caches for frames it hasn't seen.
"""

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter, uniform_filter

warnings.simplefilter("ignore")

from kreutz.evaluate import report_points  # noqa: E402
from kreutz.frames import (BG_FRAMES, BG_SKIP, CADENCE, INSTRUMENT, NOISE_BOX, STAR_FRAMES,  # noqa: E402
                           Sequence, cosmic_rays, frame_paths)
from kreutz.search import CLIP_SIGMA  # noqa: E402
from kreutz.shortlist import all_reports  # noqa: E402

HALF = 100      # stack box half-size
PIC_HALF = 40   # picture half-size
MARGIN = 190    # region padding around the track: HALF + filter support + noise box
NOISE_R = 12    # px from the track excluded from the noise estimate
VARIANTS = ["pipe", "dog12", "dog12ns", "dog2", "dog3", "dog4", "bin2", "plain"]


def local_sigma(f, valid, box=NOISE_BOX):
    frac = uniform_filter(valid.astype(np.float32), box)
    with np.errstate(divide="ignore", invalid="ignore"):
        return uniform_filter(np.abs(f) * valid, box) / frac / np.sqrt(2 / np.pi)


def norm(f, valid, box=NOISE_BOX):
    with np.errstate(divide="ignore", invalid="ignore"):
        s = f / local_sigma(f, valid, box)
    s[~valid | ~np.isfinite(s)] = 0
    return s


def dog(d, a, b):
    return gaussian_filter(d, a) - gaussian_filter(d, b)


def bin2(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def unbin2(a, shape):
    out = np.repeat(np.repeat(a, 2, 0), 2, 1)
    full = np.zeros(shape, np.float32)
    full[:out.shape[0], :out.shape[1]] = out
    return full


def plain_hp(d):
    """Gaussian 1.5 px smoothing minus a ~41 px running median (taken on a
    4x4-binned copy, 11 bins wide, for speed): the plain difference with
    only the slow glow removed."""
    g = gaussian_filter(d, 1.5)
    h, w = g.shape[0] // 4 * 4, g.shape[1] // 4 * 4
    m = median_filter(g[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3)), 11)
    full = np.pad(np.repeat(np.repeat(m, 4, 0), 4, 1), ((0, g.shape[0] - h), (0, g.shape[1] - w)), mode="edge")
    return g - full


def parse_hhmm(hhmm, ref):
    t = ref.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]), second=0, microsecond=0)
    if t < ref - timedelta(hours=2):
        t += timedelta(days=1)
    return t


def gray(img, lo=-2.0, hi=4.0, scale=2):
    """8-bit picture: robust-noise stretch lo..hi about the median."""
    v = img[np.isfinite(img)]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 or 1.0
    g = np.clip((np.nan_to_num(img, nan=med) - med) / mad, lo, hi)
    g = ((g - lo) / (hi - lo) * 255).astype(np.uint8)
    im = Image.fromarray(g).resize((g.shape[1] * scale, g.shape[0] * scale), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    cy, cx = im.size[1] // 2, im.size[0] // 2
    gap, ln = 20, 14
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        d.line([(cx + dx * gap, cy + dy * gap), (cx + dx * (gap + ln), cy + dy * (gap + ln))],
               fill=(255, 200, 60), width=1)
    return im


def montage(tiles, labels, cols, path, scale_note=""):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 14)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for k, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = (k % cols) * w, (k // cols) * (h + 14)
        sheet.paste(t, (x, y + 14))
        d.text((x + 3, y + 1), lab, fill=(220, 220, 220))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report")
    ap.add_argument("--extra", nargs="*", default=[], help="HHMM,x,y track points beyond the report's")
    ap.add_argument("--deg", type=int, default=1)
    ap.add_argument("--only-extra", action="store_true", help="fit the track to the --extra points alone")
    ap.add_argument("--orbit", action="store_true",
                    help="track along a Kreutz orbit (T, node, peri free) through the points instead of a polynomial")
    ap.add_argument("--from", dest="t_from", help="HHMM of the first window end frame")
    ap.add_argument("--to", dest="t_to", help="HHMM of the last frame")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pics", nargs="*", help="HHMM of frames to picture (default: the report's points)")
    ap.add_argument("--end", help="HHMM of the last frame of the scanned/pictured window (default: the last report point)")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--name")
    args = ap.parse_args()

    r = {r["id"]: r for r in all_reports()}[args.report]
    posted = datetime.strptime(r["posted_utc"], "%Y-%m-%dT%H:%M:%SZ")
    (r, pts, _), = report_points([r], posted.date())
    t0 = pts[0][0]
    allpts = ([] if args.only_extra else list(pts)) + [(parse_hhmm(e.split(",")[0], t0), float(e.split(",")[1]), float(e.split(",")[2]))
                          for e in args.extra]
    tk = np.array([(t - t0) / CADENCE for t, _, _ in allpts])
    fx = np.polyfit(tk, [p[1] for p in allpts], args.deg)
    fy = np.polyfit(tk, [p[2] for p in allpts], args.deg)
    res = [(float(np.polyval(fx, k) - p[1]), float(np.polyval(fy, k) - p[2])) for k, p in zip(tk, allpts)]
    print(f"{args.report}: track deg {args.deg} through {len(allpts)} points; residuals (x,y) px "
          + " ".join(f"({a:+.1f},{b:+.1f})" for a, b in res))

    first = parse_hhmm(args.t_from, t0) if args.t_from else t0 + (args.n - 5) * -CADENCE
    last = parse_hhmm(args.t_to, t0) if args.t_to else pts[-1][0] + 4 * CADENCE
    span_first = first - CADENCE * (args.n - 1)
    seq = Sequence(frame_paths(span_first - CADENCE * (BG_FRAMES + BG_SKIP + STAR_FRAMES + 2),
                               last + CADENCE / 2))
    H, W = seq.shape
    times = seq.times

    if args.orbit:
        from kreutz import orbit
        from kreutz.orbitsearch import _wcs, _where, orbit_of, sky_points
        e0, _ = orbit_of(allpts)
        ts, tx, ty = sky_points(allpts)
        el, rms = orbit.fit(ts, tx, ty, e0, _where(), free=("T", "node", "peri"))
        print(f"orbit through the points: rms {rms:.0f}\" node {el.node:.1f} peri {el.peri:.1f} T {el.T:%m-%d %H:%M}")

    def track_rc(i):
        if args.orbit:
            w, _, _ = _wcs(seq.paths[i])
            lon, lat = orbit.helioprojective(el, [times[i]], _where())
            col, row = w.wcs_world2pix(lon, lat, 0)
            return float(row[0]), float(col[0]), float(col[0]), float(H - 1 - row[0])
        k = (times[i] - t0) / CADENCE
        x, y = float(np.polyval(fx, k)), float(np.polyval(fy, k))
        return H - 1 - y, x, x, y

    ana = [i for i, t in enumerate(times) if span_first - CADENCE / 2 <= t <= last + CADENCE / 2]
    rcs = np.array([track_rc(i)[:2] for i in ana])
    R0 = int(max(rcs[:, 0].min() - MARGIN, 0)); R1 = int(min(rcs[:, 0].max() + MARGIN, H))
    C0 = int(max(rcs[:, 1].min() - MARGIN, 0)); C1 = int(min(rcs[:, 1].max() + MARGIN, W))
    print(f"frames {times[ana[0]]:%m-%d %H:%M}..{times[ana[-1]]:%m-%d %H:%M} ({len(ana)}), region rows {R0}:{R1} cols {C0}:{C1}")

    # Raw crops (dark-filled by seq.raw) of every frame any analysed frame needs.
    need = sorted({j for i in ana for j in range(i - BG_SKIP - BG_FRAMES, i + 1) if j >= 0})
    crops, pipe = {}, {}
    for i in ana:  # live maps first, while seq still holds what they need
        pipe[i] = np.array(seq.significance(i)[R0:R1, C0:C1], np.float32)
    for j in need:
        crops[j] = np.array(seq.raw(j)[R0:R1, C0:C1], np.float32)
        if j >= min(ana) - STAR_FRAMES - BG_SKIP - 2:
            seq._sources(j)
        seq._raw.pop(j, None)
    rows = np.arange(R0, R1, dtype=np.float32)[:, None]
    cols = np.arange(C0, C1, dtype=np.float32)[None, :]

    maps = {v: {} for v in VARIANTS}
    plain_imgs, star_imgs = {}, {}
    for i in ana:
        nb = [j for j in range(i - BG_SKIP - BG_FRAMES, i - BG_SKIP) if j >= 0]
        bg = np.nanmedian(np.stack([crops[j] for j in nb]), axis=0)
        d = crops[i] - bg
        bad = ~np.isfinite(d)
        d[bad] = 0
        if INSTRUMENT.cosmic_rays:
            d[cosmic_rays(d)] = 0
        layers = []
        for back in range(BG_SKIP + 1, BG_SKIP + 1 + STAR_FRAMES):
            j = i - back
            dr, dc = seq._displacement(j, i)(rows, cols)
            rj = np.nan_to_num(crops[j] - bg)
            layers.append(map_coordinates(rj, [rows - dr - R0, cols - dc - C0], order=1, cval=0.0))
        ds = d - np.median(np.stack(layers), axis=0)
        valid = ~bad
        out = {
            "pipe": np.nan_to_num(pipe[i]),
            "dog12": norm(dog(ds, 1.2, 4.0), valid),
            "dog12ns": norm(dog(d, 1.2, 4.0), valid),
            "dog2": norm(dog(ds, 2.0, 6.0), valid),
            "dog3": norm(dog(ds, 3.0, 10.0), valid),
            "dog4": norm(dog(ds, 4.0, 14.0), valid),
            "bin2": unbin2(norm(dog(bin2(ds), 1.2, 4.0), bin2(valid.astype(np.float32)) > 0.99, 21), d.shape),
            "plain": norm(plain_hp(d), valid),
        }
        for v in VARIANTS:
            maps[v][i] = np.clip(out[v], -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)
        plain_imgs[i] = gaussian_filter(np.where(valid, d, np.nan), 1.5)
        star_imgs[i] = gaussian_filter(np.where(valid, ds, np.nan), 1.5)
        for j in [j for j in crops if j < i - BG_SKIP - BG_FRAMES - STAR_FRAMES - 2]:
            del crops[j]

    def cut(img, i, dv=(0.0, 0.0), half=HALF, fill=0.0):
        """The +-half crop of a region-sized map centred on the track in frame i;
        dv (report px/frame) tilts the track about the report's last point."""
        r, c, _, _ = track_rc(i)
        k = (times[i] - pts[-1][0]) / CADENCE
        r, c = r - dv[1] * k, c + dv[0] * k  # report y is down, rows are up
        r, c = int(round(r)) - R0, int(round(c)) - C0
        out = np.full((2 * half + 1, 2 * half + 1), fill, np.float32)  # padded past the frame's edge
        a0, a1 = max(r - half, 0), min(r + half + 1, img.shape[0])
        b0, b1 = max(c - half, 0), min(c + half + 1, img.shape[1])
        out[a0 - (r - half):a1 - (r - half), b0 - (c - half):b1 - (c - half)] = img[a0:a1, b0:b1]
        return out

    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    far = np.hypot(yy, xx) > NOISE_R
    near2 = np.hypot(yy, xx) <= 2

    def stack(v, idx, dv=(0.0, 0.0)):
        return sum(cut(maps[v][i], i, dv) for i in idx) / np.sqrt(len(idx))

    out_dir = Path(args.out) / (args.name or args.report)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    print(f"\nsliding {args.n}-frame windows along the track; per variant: value at track (max r<=2) / "
          f"empirical z (value/std off-track) / ratio to the box's off-track max")
    print(f"{'window end':11s} " + " ".join(f"{v:>16s}" for v in VARIANTS))
    for e in range(args.n - 1, len(ana)):
        idx = [i for i in ana[e - args.n + 1:e + 1] if (times[ana[e]] - times[i]) < CADENCE * (args.n - 0.5)]
        if len(idx) < args.n - 1:
            continue
        row = {"end": times[ana[e]].isoformat(), "n": len(idx)}
        cells = []
        for v in VARIANTS:
            s = stack(v, idx)
            val = float(s[near2].max())
            sd = float(s[far].std())
            mx = float(s[far].max())
            row[v] = dict(value=round(val, 2), z_emp=round(val / sd, 2), noise_max=round(mx, 2), noise_sd=round(sd, 2))
            cells.append(f"{val:5.1f}/{val / sd:4.1f}/{val / mx:4.2f}")
        table.append(row)
        print(f"{times[ana[e]]:%m-%d %H:%M} " + " ".join(f"{c:>16s}" for c in cells), flush=True)

    # Velocity scan and look-elsewhere at the window ending at the last report point.
    t_end = parse_hhmm(args.end, t0) if args.end else pts[-1][0]
    e = min(range(len(ana)), key=lambda k: abs(times[ana[k]] - t_end))
    idx = ana[max(e - args.n + 1, 0):e + 1]
    print(f"\nvelocity scan +-1 px/frame, window ending {times[ana[e]]:%m-%d %H:%M} ({len(idx)} frames): "
          f"best within 3 px of the track vs the max of the same over 7x7 cells off-track")
    near3 = np.hypot(yy, xx) <= 3
    scan = {}
    for v in VARIANTS:
        best, cell_max = -np.inf, None
        for dvx in np.arange(-1, 1.01, 0.5):
            for dvy in np.arange(-1, 1.01, 0.5):
                s = stack(v, idx, (dvx, dvy))
                b = float(s[near3].max())
                if b > best:
                    best, bv = b, (float(dvx), float(dvy))
                cm = np.array([[s[a:a + 7, c:c + 7].max() for c in range(0, 2 * HALF - 6, 7)]
                               for a in range(0, 2 * HALF - 6, 7)])
                cc = np.array([[np.hypot(a + 3 - HALF, c + 3 - HALF) > NOISE_R + 3
                                for c in range(0, 2 * HALF - 6, 7)] for a in range(0, 2 * HALF - 6, 7)])
                cm = np.where(cc, cm, -np.inf)
                cell_max = cm if cell_max is None else np.maximum(cell_max, cm)
        cv = cell_max[np.isfinite(cell_max)]
        frac = float(np.mean(cv >= best))
        scan[v] = dict(best=round(best, 2), dv=bv, cells_max=round(float(cv.max()), 2),
                       cells_p99=round(float(np.percentile(cv, 99)), 2), frac_cells_above=round(frac, 3))
        print(f"  {v:8s} best {best:5.1f} at dv {bv}; off-track cells: max {cv.max():5.1f}, "
              f"99th pct {np.percentile(cv, 99):5.1f}, fraction >= best {frac:.3f} of {cv.size}")

    print("\nper-frame value at the track (max r<=1), window frames:")
    yy1, xx1 = np.mgrid[-1:2, -1:2]
    for v in VARIANTS:
        vals = [float(cut(maps[v][i], i, half=1).max()) for i in idx]
        print(f"  {v:8s} " + " ".join(f"{x:5.1f}" for x in vals))

    # Pictures: +-PIC_HALF px around the track, 3x (2x2-binned: 6x).
    pic_times = [parse_hhmm(h, t0) for h in args.pics] if args.pics else [p[0] for p in pts]
    pic_idx = [min(ana, key=lambda i: abs(times[i] - t)) for t in pic_times]
    idx = ana[max(e - args.n + 1, 0):e + 1]
    for name, src, b in (("plain", plain_imgs, False), ("plain-bin2", plain_imgs, True),
                         ("starsub", star_imgs, False), ("pipe", maps["pipe"], False)):
        tiles, labels = [], []
        planes = [cut(src[i], i, half=PIC_HALF, fill=np.nan) for i in pic_idx]
        planes.append(np.nanmean([cut(src[i], i, half=PIC_HALF, fill=np.nan) for i in idx], axis=0))
        for k, c in enumerate(planes):
            if b:
                tiles.append(gray(bin2(np.nan_to_num(c)), scale=6))
            else:
                tiles.append(gray(c, scale=3))
            labels.append(f"{name} {times[pic_idx[k]]:%m-%d %H:%M}" if k < len(pic_idx)
                          else f"stack {len(idx)} to {times[idx[-1]]:%H:%M}")
        montage(tiles, labels, len(tiles), out_dir / f"frames-{name}.png")
    tiles, labels = [], []
    for v in VARIANTS:
        s = stack(v, idx)[HALF - PIC_HALF:HALF + PIC_HALF + 1, HALF - PIC_HALF:HALF + PIC_HALF + 1]
        tiles.append(gray(s, lo=-3, hi=5, scale=3))
        labels.append(f"{v} stack {len(idx)} to {times[idx[-1]]:%H:%M}")
    montage(tiles, labels, 4, out_dir / "stack-variants.png")
    (out_dir / "probe.json").write_text(json.dumps(dict(report=args.report, extra=args.extra, deg=args.deg,
                                                        windows=table, scan=scan), indent=1))
    print(f"\npictures and probe.json in {out_dir}")


if __name__ == "__main__":
    main()
