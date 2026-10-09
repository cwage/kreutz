"""Pictures from a render's cutouts.fits, in a named style.

    docker compose run --rm kreutz kreutz.style /data/render/sep16 [--preset soft] [--all]

Writes data/render/<name>/style-<preset>/: crop.gif (the object held still,
stars sliding past), field.gif (the region it crosses, object ringed if
the preset says so), crop/ and field/ as numbered PNG frames, still.png
(the STACK plane). 8-bit grayscale. The PNG frames and still are full
resolution; field.gif is made to be looked at rather than kept: at most
GIF_MAX px on its long side (--gif-max, 0 for full size) and GIF_LEVELS
gray levels, since a whole crossing's field is most of the frame and a
GIF of it at full size runs to tens of megabytes (the noise dithers the
coarser levels away). For anything better than a GIF, convert the frames:

    ffmpeg -framerate 8 -i field/%03d.png -vf scale=iw/2:-2 -pix_fmt yuv420p field.mp4

The presets are reference points, not the project's opinion. Each is a
small dictionary; copy one and change it.
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

PRESETS = {
    # The subtracted data as it is: the slow coronal glow stays and the
    # stretch is scaled to the field, so noise and glow sit in mid-grays
    # and the comet is a bright thing in a luminous field.
    "soft": dict(highpass=None, smooth=0.7, sun_mask=None, stretch="linear",
                 ref="field", units="sigma", lo=-2, hi=10, ring=False, scale=3, tint=None),
    # The same, toned like an old plate.
    "tintype": dict(highpass=None, smooth=0.7, sun_mask=None, stretch="linear",
                    ref="field", units="sigma", lo=-2, hi=10, ring=False, scale=3,
                    tint=((28, 24, 18), (232, 214, 178))),
    # Everything wider than a tail removed, glare near the Sun blanked,
    # asinh stretch scaled to the comet crop: stars and comet on black.
    "highpass": dict(highpass=12, smooth=0.7, sun_mask=340, stretch="asinh",
                     ref="crop", units="percentile", lo=30, hi=99.9, ring=True, scale=3, tint=None),
}
# ref: what the stretch is scaled to, the comet crop or the field.
# units: lo/hi as percentiles of it, or as multiples of its robust noise
# (median absolute deviation) about its median, which ignores a few
# bright pixels (the glare at the occulter edge) that percentiles don't.
DURATION = 150


def process(cube, p, sun=None):
    """Apply a preset's filtering to a (frames, rows, cols) cube; NaN kept."""
    out = np.empty_like(cube)
    for k, plane in enumerate(cube):
        bad = ~np.isfinite(plane)
        d = np.nan_to_num(plane)
        if p["smooth"]:
            d = gaussian_filter(d, p["smooth"])
        if p["highpass"]:
            d = d - gaussian_filter(np.nan_to_num(plane), p["highpass"])
        d[bad] = np.nan
        if p["sun_mask"] and sun is not None:
            sr, sc = sun
            rr, cc = np.ogrid[:d.shape[0], :d.shape[1]]
            d[(rr - sr) ** 2 + (cc - sc) ** 2 < p["sun_mask"] ** 2] = np.nan
        out[k] = d
    return out


def to8(img, lo, hi, stretch):
    a = np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo + 1e-30), 0, None)
    if stretch == "asinh":
        a = np.arcsinh(a * 10) / np.arcsinh(10)
    return np.flipud(np.clip(a * 255, 0, 255).astype(np.uint8))


def limits(data, p):
    d = data[np.isfinite(data)]
    if p["units"] == "sigma":
        med = np.median(d)
        sig = np.median(np.abs(d - med)) / 0.6745
        return med + p["lo"] * sig, med + p["hi"] * sig
    return np.percentile(d, [p["lo"], p["hi"]])


def toned(img8, tint):
    """8-bit gray -> RGB between two colors (dark, light)."""
    if tint is None:
        return Image.fromarray(img8)
    (r0, g0, b0), (r1, g1, b1) = tint
    f = img8.astype(np.float32) / 255
    rgb = np.stack([r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0)], -1)
    return Image.fromarray(rgb.astype(np.uint8))


GIF_MAX = 800  # field.gif's long side, px
GIF_LEVELS = 64  # gray levels in field.gif; LZW likes fewer, the eye can't tell on a noisy field


def coarser(img8, levels):
    """img8 with its 256 gray levels reduced to `levels`, evenly spaced,
    the ends kept so stars stay as bright as they were."""
    step = 256 // levels
    v = img8.astype(np.int32) // step * step
    return (v * 255 // (256 - step)).astype(np.uint8)


def shrunk(img8, max_px):
    """img8 scaled so its long side is at most max_px (0: as is), and the
    scale factor applied."""
    h, w = img8.shape
    if not max_px or max(h, w) <= max_px:
        return img8, 1.0
    k = max_px / max(h, w)
    return np.array(Image.fromarray(img8).resize((round(w * k), round(h * k)), Image.LANCZOS)), k


def save_gif(frames, path):
    if frames[0].mode == "RGB":
        frames = [f.quantize(256) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=DURATION, loop=0)


def render(src, preset, gif_max=GIF_MAX):
    p = PRESETS[preset]
    with fits.open(src / "cutouts.fits") as h:
        hdr = h[0].header
        crop = h["CROP_SUB"].data.astype(np.float32)
        field = h["FIELD_SUB"].data.astype(np.float32)
        stack = h["STACK"].data.astype(np.float32)
        tr = h["TRACK"].data
        times, xs, ys = tr["TIME"], tr["X"], tr["Y"]
    r0, c0 = hdr["ROW0"], hdr["COL0"]
    sun_field = (hdr["SUNROW"] - r0, hdr["SUNCOL"] - c0)
    crop_p = process(crop, p)  # the crop moves; the Sun mask applies to the field only
    field_p = process(field, p, sun_field)
    stack_p = process(stack[None], p)[0]
    # One stretch for the whole animation, scaled to the crop (the object
    # sets the scale) or the field (the sky does).
    lo, hi = limits(crop_p if p["ref"] == "crop" else field_p, p)
    label = 255 if p["tint"] is None else p["tint"][1]
    out = src / f"style-{preset}"
    for sub in ("crop", "field"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*.png"):  # a re-render may have fewer frames
            old.unlink()
    crop_frames, field_frames, field_gif = [], [], []
    for k in range(len(times)):
        tag = f"{times[k][:10]} {times[k][11:16]} UTC"
        c = np.kron(to8(crop_p[k], lo, hi, p["stretch"]), np.ones((p["scale"], p["scale"]), np.uint8))
        im = toned(c, p["tint"])
        ImageDraw.Draw(im).text((4, 2), tag, fill=label)
        crop_frames.append(im)
        f8 = to8(field_p[k], lo, hi, p["stretch"])

        def field_image(f8, scale=1.0):
            im = toned(f8, p["tint"])
            d = ImageDraw.Draw(im)
            if p["ring"]:
                # report (x, y) -> field array (row, col) -> flipped display row
                rr = (field_p.shape[1] - 1 - ((hdr["NFULLROW"] - 1 - ys[k]) - r0)) * scale
                cc = (xs[k] - c0) * scale
                rad = 12 * scale
                d.ellipse([cc - rad, rr - rad, cc + rad, rr + rad], outline=label)
            d.text((4, 2), tag, fill=label)
            return im

        field_frames.append(field_image(f8))
        # the GIF frame: shrunk first, then fewer levels, so the resampling
        # doesn't put the levels back
        small, scale = shrunk(f8, gif_max)
        field_gif.append(field_image(coarser(small, GIF_LEVELS), scale))
    for k, (a, b) in enumerate(zip(crop_frames, field_frames)):
        a.save(out / "crop" / f"{k:03d}.png")
        b.save(out / "field" / f"{k:03d}.png")
    save_gif(crop_frames, out / "crop.gif")
    save_gif(field_gif, out / "field.gif")
    slo, shi = limits(stack_p, p)
    s = np.kron(to8(stack_p, slo, shi, p["stretch"]), np.ones((p["scale"], p["scale"]), np.uint8))
    im = toned(s, p["tint"])
    ImageDraw.Draw(im).text((4, 2), f"{times[0][:16].replace('T', ' ')} to {times[-1][11:16]} UTC, "
                                    f"{len(times)} frames", fill=label)
    im.save(out / "still.png")
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("render", type=Path, help="a data/render/<name> directory")
    ap.add_argument("--preset", default="soft", choices=sorted(PRESETS))
    ap.add_argument("--all", action="store_true", help="every preset")
    ap.add_argument("--gif-max", type=int, default=GIF_MAX,
                    help=f"shrink field.gif to this many px on its long side (default {GIF_MAX}; 0: full size)")
    args = ap.parse_args()
    for preset in (sorted(PRESETS) if args.all else [args.preset]):
        render(args.render, preset, args.gif_max)


if __name__ == "__main__":
    main()
