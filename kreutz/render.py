"""Follow one detection through the frames and write the data of its passage.

    docker compose run --rm kreutz kreutz.render 2026-09-16T01:45 1953 1126 -9 -1.5 [--name sep16]
    docker compose run --rm kreutz kreutz.render --vet /data/weekly/2026-09-27_2026-10-04/to-vet.json 12

Starting from one detection (a time, position and velocity in report
coordinates, as the search and shortlist record them), the object is
followed through the significance maps forward and backward until it is
lost (--lost consecutive misses), the span runs out (--back/--ahead
frames) or the data has a gap longer than MAX_GAP frames, refitting its
velocity as it goes so acceleration over hours doesn't lose it, and
stepping by elapsed time so a short gap in the data doesn't either. Then into data/render/<name>/:

    track.json      per frame: time, position (report coords), the
                    significance peak if it was seen, else interpolated
    cutouts.fits    the data, float32, FITS array orientation (row 0 is
                    the bottom of the displayed image; report y = NAXIS2-1-row):
        CROP_RAW    (frames, 2h+1, 2h+1) raw frame crops centered on the
                    object in every frame, so it is aligned and stars move
        CROP_SUB    the same crops with the detector's median background
                    (the previous 16 frames) subtracted
        FIELD_RAW   (frames, rows, cols) the region of the full frame the
                    path crosses, unaligned; header ROW0/COL0 give its origin
        FIELD_SUB   the same, background-subtracted
        STACK       (2h+1, 2h+1) mean of CROP_SUB: a shift-and-stack along
                    the object's motion, object sharp, stars trailed
        TRACK       table: TIME, X, Y, SIGMA, SEEN, FRAME (index into the cubes)
    preview.gif     a quick look: CROP_SUB, linear stretch, nothing else,
                    to check the right thing was followed

Nothing is filtered, masked, clipped or stretched in the FITS products
(NaN where the detector has no data). How to make pictures from them is a
separate step and a matter of taste: style.py has reference presets.
"""

import argparse
import json
from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw

from . import DATA
from .frames import BG_FRAMES, BG_SKIP, CADENCE, INSTRUMENT, STAR_FRAMES, Sequence, frame_paths

FOLLOW_R = 4        # px around the prediction to look for the peak
GAP_R = 2           # extra px of search radius per frame of data gap
MAX_GAP = 8         # frames; beyond this the background model is stale and
                    # a widened search box finds noise, so the track ends
PEAK_SIGMA = 4.0    # a frame counts when the significance peak reaches this
FIT_POINTS = 6      # recent accepted points the velocity is refit from
CROP_HALF = 60
FIELD_PAD = 120     # px around the path's bounding box
PREVIEW_SCALE = 3
PREVIEW_MS = 150


def follow(seq, i0, x, y, vx, vy, back, ahead, lost):
    """{frame index: (x, y, significance or None)} along the object, with
    positions interpolated where it wasn't seen."""
    H = seq.shape[0]

    def frames_since(i, j):
        return (seq.times[i] - seq.times[j]) / CADENCE

    def seen(i, px, py, radius):
        s = seq.significance(i)
        r0, c0 = int(round(H - 1 - py)), int(round(px))
        win = s[r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1]
        if win.shape != (2 * radius + 1,) * 2 or not np.isfinite(win).any():
            return None
        rr, cc = np.unravel_index(np.nanargmax(win), win.shape)
        if win[rr, cc] < PEAK_SIGMA:
            return None
        return float(c0 - radius + cc), float(H - 1 - (r0 - radius + rr)), float(win[rr, cc])

    hits = {}
    for direction, limit in ((1, ahead), (-1, back)):
        px, py, cvx, cvy = x, y, vx, vy
        i_prev = i0
        pts = []  # (frames since i0, x, y) accepted in this direction
        misses = 0
        for k in range(0, limit + 1):
            i = i0 + direction * k
            if not 0 <= i < len(seq.paths):
                break
            dt = frames_since(i, i_prev)  # signed; negative going back
            if abs(dt) > MAX_GAP:
                break
            qx, qy = px + cvx * dt, py + cvy * dt
            radius = FOLLOW_R + int(round(GAP_R * max(abs(dt) - 1, 0)))
            h = seen(i, qx, qy, radius)
            if h is None:
                misses += 1
                if misses > lost and k > 0:
                    break
                continue
            misses = 0
            hits[i] = h
            px, py, i_prev = h[0], h[1], i
            pts.append((frames_since(i, i0), h[0], h[1]))
            if len(pts) >= 3:
                t = np.array([p[0] for p in pts[-FIT_POINTS:]])
                cvx = float(np.polyfit(t, [p[1] for p in pts[-FIT_POINTS:]], 1)[0])
                cvy = float(np.polyfit(t, [p[2] for p in pts[-FIT_POINTS:]], 1)[0])
    if not hits:
        raise ValueError("object not seen in any frame")
    idx = sorted(hits)
    track = {}
    for i in range(idx[0], idx[-1] + 1):
        if i in hits:
            track[i] = hits[i]
        else:
            a = max(j for j in idx if j < i)
            b = min(j for j in idx if j > i)
            f = frames_since(i, a) / frames_since(b, a)
            track[i] = (hits[a][0] + f * (hits[b][0] - hits[a][0]),
                        hits[a][1] + f * (hits[b][1] - hits[a][1]), None)
    return track


def crop_at(img, x, y, H, half):
    r, c = int(round(H - 1 - y)), int(round(x))
    out = np.full((2 * half + 1, 2 * half + 1), np.nan, np.float32)
    r0, r1 = max(r - half, 0), min(r + half + 1, img.shape[0])
    c0, c1 = max(c - half, 0), min(c + half + 1, img.shape[1])
    out[r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = img[r0:r1, c0:c1]
    return out


def cutouts(seq, track):
    """The cubes and the field origin."""
    H, W = seq.shape
    idx = sorted(track)
    xs = np.array([track[i][0] for i in idx])
    ys = np.array([track[i][1] for i in idx])
    r0 = int(max(min(H - 1 - ys) - FIELD_PAD, 0)); r1 = int(min(max(H - 1 - ys) + FIELD_PAD, H))
    c0 = int(max(min(xs) - FIELD_PAD, 0)); c1 = int(min(max(xs) + FIELD_PAD, W))
    crop_raw, crop_sub, field_raw, field_sub = [], [], [], []
    for i in idx:
        seq.forget(i - BG_FRAMES - BG_SKIP - 2)
        raw = seq.raw(i)
        sub = raw - seq._background(i)
        x, y = track[i][0], track[i][1]
        crop_raw.append(crop_at(raw, x, y, H, CROP_HALF))
        crop_sub.append(crop_at(sub, x, y, H, CROP_HALF))
        field_raw.append(raw[r0:r1, c0:c1].astype(np.float32))
        field_sub.append(sub[r0:r1, c0:c1].astype(np.float32))
    return (np.stack(crop_raw), np.stack(crop_sub), np.stack(field_raw), np.stack(field_sub)), (r0, c0)


def write_fits(path, seq, track, cubes, origin):
    idx = sorted(track)
    crop_raw, crop_sub, field_raw, field_sub = cubes
    sr, sc = seq.sun_rc()
    primary = fits.PrimaryHDU()
    h = primary.header
    h["ORIGIN"] = "kreutz.render"
    h["INSTRUME"] = INSTRUMENT.label
    h["NFRAMES"] = len(idx)
    h["TSTART"] = seq.times[idx[0]].isoformat()
    h["TEND"] = seq.times[idx[-1]].isoformat()
    h["CROPHALF"] = (CROP_HALF, "crops are 2*CROPHALF+1 square, object at center")
    h["ROW0"] = (origin[0], "FIELD_* origin row in the full frame")
    h["COL0"] = (origin[1], "FIELD_* origin col in the full frame")
    h["NFULLROW"] = (seq.shape[0], "full frame rows; report y = NFULLROW-1-row")
    h["NFULLCOL"] = (seq.shape[1], "full frame cols")
    h["SUNROW"] = (float(sr), "Sun center row in the full frame")
    h["SUNCOL"] = (float(sc), "Sun center col in the full frame")
    h["BGFRAMES"] = (BG_FRAMES, "median background from this many previous frames")
    h["BGSKIP"] = (BG_SKIP, "skipping this many most recent")
    h["COMMENT"] = "Array orientation as stored in the L1b files: report y = NAXIS2-1-row."
    hdus = [primary]
    for name, cube in (("CROP_RAW", crop_raw), ("CROP_SUB", crop_sub),
                       ("FIELD_RAW", field_raw), ("FIELD_SUB", field_sub)):
        hdus.append(fits.ImageHDU(cube, name=name))
    hdus.append(fits.ImageHDU(np.nanmean(crop_sub, axis=0).astype(np.float32), name="STACK"))
    cols = [
        fits.Column("TIME", "A19", array=np.array([seq.times[i].isoformat()[:19] for i in idx])),
        fits.Column("X", "E", array=np.array([track[i][0] for i in idx])),
        fits.Column("Y", "E", array=np.array([track[i][1] for i in idx])),
        fits.Column("SIGMA", "E", array=np.array([track[i][2] if track[i][2] is not None else np.nan
                                                  for i in idx])),
        fits.Column("SEEN", "L", array=np.array([track[i][2] is not None for i in idx])),
        fits.Column("FRAME", "J", array=np.arange(len(idx))),
    ]
    hdus.append(fits.BinTableHDU.from_columns(cols, name="TRACK"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def preview_limits(crop_sub):
    """One linear scale for all frames, taken from a typical frame: the
    median over frames of each frame's 1st and 99.7th percentiles. A few
    eclipse-flash frames, saturated white, would otherwise set the top of
    the scale and crush every normal frame to black; this way they clip
    instead."""
    per = np.array([np.nanpercentile(plane, [1, 99.7]) for plane in crop_sub
                    if np.isfinite(plane).any()])
    lo, hi = np.median(per, axis=0)
    return lo, hi


def preview(crop_sub, times, path):
    """Linear stretch of the subtracted crops, one scale for all frames."""
    lo, hi = preview_limits(crop_sub)
    frames = []
    for plane, t in zip(crop_sub, times):
        a = np.clip((np.nan_to_num(plane, nan=lo) - lo) / (hi - lo + 1e-30) * 255, 0, 255)
        a = np.flipud(a.astype(np.uint8))
        im = Image.fromarray(np.kron(a, np.ones((PREVIEW_SCALE, PREVIEW_SCALE), np.uint8)))
        ImageDraw.Draw(im).text((4, 2), f"{t:%Y-%m-%d %H:%M} UTC", fill=255)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=PREVIEW_MS, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("time", nargs="?")
    ap.add_argument("x", nargs="?", type=float)
    ap.add_argument("y", nargs="?", type=float)
    ap.add_argument("vx", nargs="?", type=float)
    ap.add_argument("vy", nargs="?", type=float)
    ap.add_argument("--vet", nargs=2, metavar=("JSON", "N"),
                    help="take time/x/y/vx/vy from entry N of a to-vet or shortlist file, "
                         "or from a live alert file (N ignored)")
    ap.add_argument("--back", type=int, default=48, help="frames to follow back (default 12 h)")
    ap.add_argument("--ahead", type=int, default=48)
    ap.add_argument("--lost", type=int, default=4, help="consecutive misses before giving up")
    ap.add_argument("--name")
    args = ap.parse_args()
    if args.vet:
        c = json.load(open(args.vet[0]))
        if isinstance(c, dict) and "track" in c:  # a live alert: its track's last window
            c = c["track"][-1]
        else:
            c = c["candidates"][int(args.vet[1])] if isinstance(c, dict) else c[int(args.vet[1])]
        t, x, y, vx, vy = datetime.fromisoformat(c["time"][:19]), c["x"], c["y"], c["vx"], c["vy"]
    else:
        t, x, y, vx, vy = datetime.fromisoformat(args.time), args.x, args.y, args.vx, args.vy
    t = t.replace(second=INSTRUMENT.second)
    warm = timedelta(minutes=15) * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.back + 2)
    seq = Sequence(frame_paths(t - warm, t + timedelta(minutes=15) * (args.ahead + 1)))
    i0 = seq.times.index(t)
    track = follow(seq, i0, x, y, vx, vy, args.back, args.ahead, args.lost)
    idx = sorted(track)
    out = DATA / f"render{INSTRUMENT.tag}" / (args.name or f"{t:%Y%m%dT%H%M}_{x:.0f}_{y:.0f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "track.json").write_text(json.dumps(
        [{"time": seq.times[i].isoformat(), "x": track[i][0], "y": track[i][1],
          "sigma": track[i][2]} for i in idx], indent=1))
    cubes, origin = cutouts(seq, track)
    write_fits(out / "cutouts.fits", seq, track, cubes, origin)
    preview(cubes[1], [seq.times[i] for i in idx], out / "preview.gif")
    seen = sum(track[i][2] is not None for i in idx)
    print(f"{out}: {len(idx)} frames {seq.times[idx[0]]:%Y-%m-%d %H:%M} to "
          f"{seq.times[idx[-1]]:%H:%M}, seen in {seen}; field {cubes[2].shape[1]}x{cubes[2].shape[2]}")


if __name__ == "__main__":
    main()
