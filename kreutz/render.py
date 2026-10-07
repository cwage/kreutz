"""Presentable images of a comet's passage: crop and field animations, a still.

    docker compose run --rm kreutz kreutz.render 2026-09-30T19:45 1729 1131 -10 -2 [--name sep30]
    docker compose run --rm kreutz kreutz.render --vet /data/weekly/2026-09-27_2026-10-04/to-vet.json 12

Starting from one detection (a time, position and velocity in report
coordinates, as the search and shortlist record them), the object is
followed through the significance maps forward and backward until it is
lost (--lost consecutive misses) or the span runs out (--back/--ahead
frames), re-fitting its velocity as it goes so acceleration over hours
doesn't lose it. Then, from background-subtracted frames, into
data/render/<name>/:

    crop.gif    a crop that moves with the comet, so it sits still while
                stars slide past; fixed stretch so brightening is real
    field.gif   the region of the frame the path crosses, native
                resolution, comet ringed
    crop/, field/   the same frames as numbered PNGs
    still.png   the frames stacked along the comet's motion: comet sharp,
                stars trailed
    track.json  the followed positions

Everything is full quality (8-bit grayscale, native pixels); make smaller
versions from the PNGs as needed, e.g. a half-size MP4:

    ffmpeg -framerate 8 -i field/%03d.png -vf scale=iw/2:-2 -pix_fmt yuv420p field.mp4

Frames are stretched with an asinh so the bright head and the faint tail
both show.
"""

import argparse
import json
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from . import DATA
from .frames import BG_FRAMES, BG_SKIP, CADENCE, STAR_FRAMES, Sequence, frame_paths

FOLLOW_R = 4        # px around the prediction to look for the peak
GAP_R = 2           # extra px of search radius per frame of data gap
PEAK_SIGMA = 4.0    # a frame counts when the significance peak reaches this
FIT_POINTS = 6      # recent accepted points the velocity is refit from
CROP_HALF = 60
CROP_SCALE = 3
FIELD_PAD = 120     # px around the path's bounding box
DURATION = 150      # ms per frame
HIGHPASS_SIGMA = 12
OCCULTER_R = 340    # px from the Sun inside which the glare off the occulter edge is blanked


def follow(seq, i0, x, y, vx, vy, back, ahead, lost):
    """{frame index: (x, y, significance or None)} along the object, with
    positions interpolated where it wasn't seen. Steps by elapsed time, not
    frame count, so a gap in the data doesn't throw the prediction off."""
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
            qx, qy = px + cvx * dt, py + cvy * dt
            # Across a gap the velocity is less certain: look wider.
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


def subtracted(seq, i):
    """Frame i minus its median background, star model left in (stars are
    the point of the animation), smoothed a touch; NaN off the detector."""
    d = np.nan_to_num(seq.raw(i) - seq._background(i))
    # The median background leaves slow coronal change and the detector's
    # column pattern behind; take out everything wider than a tail.
    d = gaussian_filter(d, 0.7) - gaussian_filter(d, HIGHPASS_SIGMA)
    d[~np.isfinite(seq.raw(i))] = np.nan
    sr, sc = seq.sun_rc()
    rr, cc = np.ogrid[:d.shape[0], :d.shape[1]]
    d[(rr - sr) ** 2 + (cc - sc) ** 2 < OCCULTER_R ** 2] = np.nan
    return d


def stretch(img, lo, hi):
    """asinh stretch between lo and hi to 8-bit, NaN black, flipped to
    display orientation (north up)."""
    a = np.arcsinh(np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo), 0, None) * 10) / np.arcsinh(10)
    return np.flipud(np.clip(a * 255, 0, 255).astype(np.uint8))


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((4, 2), text, fill=255)
    return im


def crop_at(img, x, y, H, half):
    r, c = int(round(H - 1 - y)), int(round(x))
    out = np.full((2 * half + 1, 2 * half + 1), np.nan, np.float32)
    r0, r1 = max(r - half, 0), min(r + half + 1, img.shape[0])
    c0, c1 = max(c - half, 0), min(c + half + 1, img.shape[1])
    out[r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = img[r0:r1, c0:c1]
    return out


def save_gif(frames, path):
    """8-bit grayscale frames as a GIF (256 grays, so nothing is lost)."""
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=DURATION, loop=0)


def render(seq, track, out):
    out.mkdir(parents=True, exist_ok=True)
    H, W = seq.shape
    idx = sorted(track)
    xs = np.array([track[i][0] for i in idx])
    ys = np.array([track[i][1] for i in idx])
    # Field region: the path's bounding box, padded, in array coords.
    r0 = int(max(min(H - 1 - ys) - FIELD_PAD, 0)); r1 = int(min(max(H - 1 - ys) + FIELD_PAD, H))
    c0 = int(max(min(xs) - FIELD_PAD, 0)); c1 = int(min(max(xs) + FIELD_PAD, W))

    subs, crops = [], []
    for i in idx:
        seq.forget(i - BG_FRAMES - BG_SKIP - 2)
        d = subtracted(seq, i)
        subs.append(d[r0:r1, c0:c1].copy())
        crops.append(crop_at(d, track[i][0], track[i][1], H, CROP_HALF))
    # One stretch for the whole animation, from the crops (where the comet is).
    allc = np.concatenate([c[np.isfinite(c)] for c in crops])
    lo, hi = np.percentile(allc, [30, 99.9])
    flo, fhi = lo, hi  # the comet is the brightest thing that matters

    crop_frames, field_frames = [], []
    for n, i in enumerate(idx):
        t = seq.times[i]
        tag = f"{t:%Y-%m-%d %H:%M} UTC"
        c = stretch(crops[n], lo, hi)
        c = np.kron(c, np.ones((CROP_SCALE, CROP_SCALE), np.uint8))
        crop_frames.append(label(c, tag))
        f = stretch(subs[n], flo, fhi)
        im = Image.fromarray(f).convert("L")
        # ring the comet: array -> downsampled, then flipped rows
        rr = H - 1 - track[i][1] - r0
        cc = track[i][0] - c0
        rr = f.shape[0] - 1 - rr
        ImageDraw.Draw(im).ellipse([cc - 12, rr - 12, cc + 12, rr + 12], outline=160)
        ImageDraw.Draw(im).text((4, 2), tag, fill=255)
        field_frames.append(im)
    save_gif(crop_frames, out / "crop.gif")
    save_gif(field_frames, out / "field.gif")
    for kind, frames in (("crop", crop_frames), ("field", field_frames)):
        (out / kind).mkdir(exist_ok=True)
        for n, im in enumerate(frames):
            im.save(out / kind / f"{n:03d}.png")

    # Still: stack the crops (already aligned on the comet).
    stack = np.nanmean(np.stack(crops), axis=0)
    lo_s, hi_s = np.nanpercentile(stack, [30, 99.95])
    s = np.kron(stretch(stack, lo_s, hi_s), np.ones((CROP_SCALE, CROP_SCALE), np.uint8))
    label(s, f"{seq.times[idx[0]]:%Y-%m-%d %H:%M} to {seq.times[idx[-1]]:%H:%M} UTC, "
             f"{len(idx)} frames").save(out / "still.png")

    (out / "track.json").write_text(json.dumps(
        [{"time": seq.times[i].isoformat(), "x": track[i][0], "y": track[i][1],
          "sigma": track[i][2]} for i in idx], indent=1))
    seen = sum(track[i][2] is not None for i in idx)
    print(f"{out}: {len(idx)} frames {seq.times[idx[0]]:%Y-%m-%d %H:%M} to "
          f"{seq.times[idx[-1]]:%H:%M}, seen in {seen}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("time", nargs="?")
    ap.add_argument("x", nargs="?", type=float)
    ap.add_argument("y", nargs="?", type=float)
    ap.add_argument("vx", nargs="?", type=float)
    ap.add_argument("vy", nargs="?", type=float)
    ap.add_argument("--vet", nargs=2, metavar=("JSON", "N"),
                    help="take time/x/y/vx/vy from entry N of a to-vet or shortlist file")
    ap.add_argument("--back", type=int, default=48, help="frames to follow back (default 12 h)")
    ap.add_argument("--ahead", type=int, default=48)
    ap.add_argument("--lost", type=int, default=4, help="consecutive misses before giving up")
    ap.add_argument("--name")
    args = ap.parse_args()
    if args.vet:
        c = json.load(open(args.vet[0]))
        c = c["candidates"][int(args.vet[1])] if isinstance(c, dict) else c[int(args.vet[1])]
        t, x, y, vx, vy = datetime.fromisoformat(c["time"][:19]), c["x"], c["y"], c["vx"], c["vy"]
    else:
        t, x, y, vx, vy = datetime.fromisoformat(args.time), args.x, args.y, args.vx, args.vy
    t = t.replace(second=25)
    warm = timedelta(minutes=15) * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.back + 2)
    seq = Sequence(frame_paths(t - warm, t + timedelta(minutes=15) * (args.ahead + 1)))
    i0 = seq.times.index(t)
    track = follow(seq, i0, x, y, vx, vy, args.back, args.ahead, args.lost)
    name = args.name or f"{t:%Y%m%dT%H%M}_{x:.0f}_{y:.0f}"
    render(seq, track, DATA / "render" / name)


if __name__ == "__main__":
    main()
