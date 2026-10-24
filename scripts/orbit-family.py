"""Fit Kreutz orbits to rendered comet tracks: where orbitsearch.py's
family (node range, PERI_A + PERI_B * node, INCLINATION) comes from.

    docker compose run --rm -T -w /srv -e PYTHONPATH=/srv --entrypoint python kreutz scripts/orbit-family.py /data/render/20261023T1045_1933_1347 /data/render-ccor2/20261023T0100_1798_1701 ...

Each directory is a render (render.py) with a track.json of measured
positions; a render-ccor2 directory is fitted from L1, the rest from
Earth. Prints the fitted node and argument of perihelion with the
perihelion time free, then with the inclination free too, and the rms
residual. A comet whose rms stays in the hundreds of arcsec doesn't fit a
Kreutz orbit near the family at all (Ruan's of 2026-10-21 11:00).
"""

import glob
import json
import sys
import warnings
from datetime import datetime, timedelta

from astropy.io import fits
from astropy.wcs import WCS

warnings.simplefilter("ignore")

from kreutz import orbit  # noqa: E402


def main(dirs):
    for d in dirs:
        inst = "ccor2" if "render-ccor2" in d else "ccor1"
        pts = [p for p in json.load(open(f"{d}/track.json")) if p["sigma"] is not None]
        times, tx, ty = [], [], []
        for p in pts:
            tt = datetime.fromisoformat(p["time"])
            pat = (f"/data/ccor1/l1b/{tt:%Y/%m/%d}/CCOR1_1B_{tt:%Y%m%dT%H%M}*_NC.fits" if inst == "ccor1"
                   else f"/data/ccor2/l1a/{tt:%Y/%m/%d}/CCOR2_1A_{tt:%Y%m%dT%H%M}*.fits")
            f = sorted(glob.glob(pat))
            if not f:
                continue
            h = fits.getheader(f[0], 1)
            lon, lat = WCS(h).wcs_pix2world([[p["x"], h["NAXIS2"] - 1 - p["y"]]], 0)[0]
            times.append(tt)
            tx.append(lon)
            ty.append(lat)
        if len(times) < 8:
            print(f"{d}: too few frames on disk ({len(times)})")
            continue
        where = "l1" if inst == "ccor2" else "earth"
        e0, _ = orbit.fit(times, tx, ty, orbit.Elements(T=times[-1] + timedelta(hours=6)), where, free=("T",))
        e1, r1 = orbit.fit(times, tx, ty, e0, where, free=("T", "node", "peri"))
        e2, r2 = orbit.fit(times, tx, ty, e1, where, free=("T", "node", "peri", "i"))
        span = (times[-1] - times[0]).total_seconds() / 3600
        print(f"{d.rstrip('/').rsplit('/', 1)[-1]} {inst}: {len(times)} pts over {span:.1f} h | "
              f"node {e1.node:6.2f} peri {e1.peri:6.2f} rms {r1:4.0f}\" | with i: node {e2.node:6.2f} "
              f"peri {e2.peri:6.2f} i {e2.i:6.2f} rms {r2:4.0f}\" | T {e2.T:%m-%d %H:%M}")


if __name__ == "__main__":
    main(sys.argv[1:])
