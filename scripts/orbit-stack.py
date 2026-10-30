"""Is a faint track real? Stack the significance maps along the Kreutz orbit
through it, on this instrument or the other CCOR, and compare with the same
stack at offsets around it.

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor2 -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro --entrypoint python kreutz scripts/orbit-stack.py --source ccor1 --track 2026-10-29T21:45,1518,1276 2026-10-29T22:00,1510,1271 2026-10-29T22:15,1502,1267 --from 2026-10-29T18:00 --to 2026-10-29T22:30

The track's points are in --source's report coordinates (default: this
instrument; a LASCO track only on its own instrument, through each frame's
Sun centre and plate scale). orbitsearch's family orbit (node scan, T free) is fitted to
them; its position in each of this instrument's frames from --from to --to
comes from the orbit, seen from this instrument's place (Earth or L1),
through that frame's WCS (CCOR), or for LASCO (SOHO, at L1) through the
frame's Sun centre and plate scale, the frames being turned solar north
up about the Sun (lasco.load). Prints, per frame, the significance there (best
within 2 px) and the running sum / sqrt(n); then the full stack against
NULL offset copies of the path (--null-r px away, random directions): its
rank and z-score among them. A comet stacks up with more frames; noise
doesn't, and sits inside the offsets' spread.
"""

import argparse
import warnings
from datetime import datetime

import numpy as np

warnings.simplefilter("ignore")

from kreutz import orbit  # noqa: E402
from kreutz.frames import Sequence, frame_paths, frame_time  # noqa: E402
from kreutz.instrument import CURRENT as INSTRUMENT, INSTRUMENTS  # noqa: E402
from astropy.io import fits  # noqa: E402
from kreutz.orbitsearch import NODES, PERI_A, PERI_B, INCLINATION, _where, _wcs, _frame_near, sky_points  # noqa: E402

NULL = 200
R = 2  # px: best significance within this of the predicted position


def pt(s):
    t, x, y = s.split(",")
    return datetime.fromisoformat(t), float(x), float(y)


def where(inst):
    return "l1" if inst.source == "lasco" else _where(inst)


def lasco_sky(pts):
    """sky_points for this LASCO instrument's own points: through each
    nearest frame's Sun centre and plate scale (project, inverted)."""
    ts, tx, ty = [], [], []
    for t, x, y in pts:
        near = frame_paths(t - INSTRUMENT.cadence, t + INSTRUMENT.cadence)
        if not near:
            continue
        h = fits.getheader(min(near, key=lambda p: abs(frame_time(p) - t)), 0)
        ts.append(t)
        tx.append((x - (h["CRPIX1"] - 1)) * h["CDELT1"] / 3600)
        ty.append((h["NAXIS2"] - 1 - y - (h["CRPIX2"] - 1)) * h["CDELT2"] / 3600)
    return ts, tx, ty


def fit(pts, inst):
    """The family orbit through [(t, x, y)] in inst's coordinates, seen from inst's place."""
    from datetime import timedelta
    if inst.source == "lasco":
        if inst is not INSTRUMENT:
            raise SystemExit("a LASCO track only as this instrument's own (--source left out)")
        ts, tx, ty = lasco_sky(pts)
    else:
        ts, tx, ty = sky_points(pts, inst)
    best = None
    for node in NODES:
        for ahead in (6, 24):
            e0 = orbit.Elements(T=ts[-1] + timedelta(hours=ahead), node=float(node),
                                peri=float(PERI_A + PERI_B * node), i=INCLINATION)
            e, rms = orbit.fit(ts, tx, ty, e0, where(inst), free=("T",))
            if best is None or rms < best[1]:
                best = (e, rms)
    return best


def project(e, t, path):
    """The orbit's (x, y) in report coordinates in this instrument's frame
    at t (`path`), or None."""
    if INSTRUMENT.source == "lasco":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = fits.getheader(path, 0)
        lon, lat = orbit.helioprojective(e, [t], "l1")
        col = h["CRPIX1"] - 1 + float(lon[0]) * 3600 / h["CDELT1"]
        row = h["CRPIX2"] - 1 + float(lat[0]) * 3600 / h["CDELT2"]
        return col, h["NAXIS2"] - 1 - row
    path = _frame_near(INSTRUMENT, t)
    if path is None:
        return None
    w, Hw, _ = _wcs(path)
    lon, lat = orbit.helioprojective(e, [t], _where())
    col, row = w.wcs_world2pix([[float(lon[0]), float(lat[0])]], 0)[0]
    return float(col), float(Hw - 1 - row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--track", nargs="+", type=pt, required=True)
    ap.add_argument("--source", default=INSTRUMENT.key)
    ap.add_argument("--from", dest="start", type=datetime.fromisoformat, required=True)
    ap.add_argument("--to", dest="end", type=datetime.fromisoformat, required=True)
    ap.add_argument("--null-r", type=float, nargs=2, default=(20, 120), metavar=("MIN", "MAX"))
    ap.add_argument("--shift", type=float, nargs=2, default=(0, 0), metavar=("DX", "DY"),
                    help="add to every projected position (report px): LASCO's level 0.5 Sun centre is off; "
                         "C2's projection ran 12 px north of our measured track of the 2026-10-23 comet")
    ap.add_argument("--r", type=int, default=R, help=f"px around the position searched for the peak (default {R})")
    ap.add_argument("--scan", type=int, default=0, metavar="PX",
                    help="also stack shifted copies of the path, every 2 px out to PX either way, and give the best; "
                         "the offsets get the same best-of-the-grid, so the z stays fair. For a path extrapolated "
                         "hours past its track, where the orbit's error grows")
    args = ap.parse_args()
    src = INSTRUMENTS[args.source]
    e, rms = fit(args.track, src)
    print(f"orbit through the {src.label} track: rms {rms:.0f}\", T {e.T:%m-%d %H:%M}, node {e.node:.1f}")
    seq = Sequence(frame_paths(args.start, args.end))
    H, W = seq.shape
    rng = np.random.default_rng(1)
    ang = rng.uniform(0, 2 * np.pi, NULL)
    dist = rng.uniform(*args.null_r, NULL)
    offs = np.stack([dist * np.cos(ang), dist * np.sin(ang)], axis=1)
    total, null, n = 0.0, np.zeros(NULL), 0
    g = np.arange(-args.scan, args.scan + 1, 2) if args.scan else np.zeros(1)
    gx, gy = [a.ravel() for a in np.meshgrid(g, g)]
    grid, gnull = np.zeros(gx.size), np.zeros((NULL, gx.size))
    for i, t in enumerate(seq.times):
        if t < args.start:
            continue
        xy = project(e, t, seq.paths[i])
        if xy is None:
            continue
        x, y = xy[0] + args.shift[0], xy[1] + args.shift[1]
        try:
            sig = seq.significance(i)
        except Exception:
            continue

        def at(px, py):
            r0, c0 = int(round(H - 1 - py)), int(round(px))
            if not (args.r <= r0 < H - args.r and args.r <= c0 < W - args.r):
                return np.nan
            return float(np.nanmax(sig[r0 - args.r:r0 + args.r + 1, c0 - args.r:c0 + args.r + 1]))
        v = at(x, y)
        if not np.isfinite(v):
            print(f"  {t:%m-%d %H:%M} ({x:.0f},{y:.0f}) off the map")
            continue
        n += 1
        total += v
        null += np.nan_to_num([at(x + dx, y + dy) for dx, dy in offs])
        print(f"  {t:%m-%d %H:%M} ({x:.0f},{y:.0f}) {v:5.1f}   stack {total / np.sqrt(n):5.1f} over {n}")
        if args.scan:
            grid += np.nan_to_num([at(x + dx, y + dy) for dx, dy in zip(gx, gy)])
            gnull += np.nan_to_num([[at(x + ox + dx, y + oy + dy) for dx, dy in zip(gx, gy)] for ox, oy in offs])
    if n:
        s, nl = total / np.sqrt(n), null / np.sqrt(n)
        print(f"{INSTRUMENT.label}: stack {s:.1f} over {n} frames; offsets {nl.mean():.1f} +- {nl.std():.1f}, "
              f"z {(s - nl.mean()) / nl.std():.1f}, beaten by {int((nl >= s).sum())} of {NULL}")
    if n and args.scan:
        k = int(np.argmax(grid))
        s, nl = grid[k] / np.sqrt(n), gnull.max(axis=1) / np.sqrt(n)
        print(f"best shift ({gx[k]:+.0f},{gy[k]:+.0f}) of +-{args.scan}: stack {s:.1f}; offsets' best {nl.mean():.1f} "
              f"+- {nl.std():.1f}, z {(s - nl.mean()) / nl.std():.1f}, beaten by {int((nl >= s).sum())} of {NULL}")
        m = (grid / np.sqrt(n)).reshape(g.size, g.size)
        print("stack by shift (rows dy, columns dx):")
        print("      " + " ".join(f"{d:+4.0f}" for d in g))
        for j, d in enumerate(g):
            print(f"  {d:+4.0f}" + " ".join(f"{v:4.1f}" for v in m[j]))


if __name__ == "__main__":
    main()
