"""SOHO LASCO C2 and C3 frames: listing, download, and loading as the
pipeline expects them (instrument.py, source "lasco").

NASA keeps level 0.5 frames in two places with the same layout,
<root>/YYMMDD/{c2,c3}/: the last ~10 days, a few minutes behind
(instrument.feed), and the archive from 2000 on, about three weeks behind
(instrument.archive_prefix). Files are named by sequence number
(23903880.fts); each day directory's img_hdr.txt lists them with their
time, size, filter and polarizer. Only the standard frames are taken
(full 1024x1024, C2 orange, C3 clear, no polarizer), which is the ~12
minute synoptic sequence: the occasional colour, binned or polarized
frames would each look like a jump in the background. They're stored as
C2_05_<observation time>_<sequence number>.fts so frames.frame_paths
finds them by time.

load() returns what frames.Sequence works on: counts above the bias per
second of exposure (both vary frame to frame in level 0.5), rotated to
solar north up about the Sun, with NaN off the sky. The rotation is
SOHO's roll (CROTA1), which flips by 180 degrees every few months and
wanders +-7 degrees in between; turning every frame north up keeps the
search sector, the lane and Sungrazer report coordinates the same all
year.
"""

import re
import urllib.error
import urllib.request
import warnings
from datetime import datetime

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

from .instrument import CURRENT as INSTRUMENT

# The synoptic sequence's (filter, polarizer).
STANDARD = {"C2": ("Orange", "Clear"), "C3": ("Clear", "Clear")}
# Off the sky: the occulter, the field stop's shadow (C3), the vignetted
# corners (C2), missing telemetry. Measured 2026-10-22, as fractions of
# the frame's median: outside C3's field stop ~1%, inside C2's occulter
# 5%; C3's corona stays above 50% out to the field stop, C2's fades with
# vignetting to 16% at 660 px from the Sun, so C2 loses its corners.
DARK_FRAC = 0.1

_ROW = re.compile(r"^(\d+\.fts)\s+(\d{4}/\d\d/\d\d)\s+(\d\d:\d\d:\d\d)\s+(\S+)\s+\S+\s+(\d+)\s+(\d+)"
                  r"\s+\d+\s+\d+\s+(\S+)\s+(\S+)")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kreutz"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def parse_index(text, telescope=INSTRUMENT.telescope):
    """[(time, file name)] of the standard frames in an img_hdr.txt."""
    out = []
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        name, day, hms, tel, cols, rows, filt, polar = m.groups()
        if tel != telescope or (cols, rows) != ("1024", "1024") or (filt, polar) != STANDARD[telescope]:
            continue
        out.append((datetime.strptime(f"{day} {hms}", "%Y/%m/%d %H:%M:%S"), name))
    return out


def day_dirs(root):
    """The YYMMDD day directories under root, as dates, oldest first."""
    return sorted(datetime.strptime(d, "%y%m%d").date()
                  for d in set(re.findall(r'href="(\d{6})/"', _get(root).decode())))


def day_frames(root, d):
    """[(time, url, destination)] for one day's standard frames under root
    (a missing day is an empty list)."""
    base = f"{root}{d:%y%m%d}/{INSTRUMENT.telescope.lower()}/"
    try:
        text = _get(base + "img_hdr.txt").decode(errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    return [(t, base + name, destination(t, name)) for t, name in parse_index(text)]


def destination(t, name):
    stem = name.rsplit(".", 1)[0]
    return INSTRUMENT.frames_dir / f"{t:%Y/%m/%d}" / f"{INSTRUMENT.telescope}_{INSTRUMENT.level}_{t:%Y%m%dT%H%M%S}_{stem}.fts"


def north_up(img, header):
    """img rotated by the roll so solar north is up, about the Sun (CRPIX;
    the level 0.5 headers put the Sun there, CRVAL 0). FITS convention:
    sky offset = R(CROTA) . pixel offset, with rows counting up."""
    rho = np.deg2rad(header["CROTA1"])
    r0, c0 = header["CRPIX2"] - 1, header["CRPIX1"] - 1
    rows, cols = np.indices(img.shape, dtype=np.float32)
    dx, dy = cols - c0, rows - r0
    c, s = np.cos(rho), np.sin(rho)
    src_col = c0 + c * dx + s * dy
    src_row = r0 - s * dx + c * dy
    return map_coordinates(img, [src_row, src_col], order=1, cval=np.nan).astype(np.float32)


# The Sun's apparent motion along the ecliptic (low-precision solar
# coordinates, good to ~0.01 deg), and the solar equator's ascending
# node on the ecliptic and its inclination, for the star drift below.
_EPOCH = datetime(2000, 1, 1, 12)
SOLAR_INCLINATION = 7.25


def star_drift(t, header):
    """The star field's motion in a north-up frame at time t, (rows, cols)
    per day, rows counting up: the Sun moves east along the ecliptic at
    ~1 deg/day (faster in January, at perihelion), so the stars move west,
    to the right, tilted by the angle between the ecliptic and solar
    north. SOHO's halo orbit about L1 adds ~1%. On C3 2026-09-15 05:18
    this gives (-0.007, +0.522) px/frame; the stars measured (-0.02, +0.51)."""
    d = (t - _EPOCH).total_seconds() / 86400
    g = np.radians(357.528 + 0.9856003 * d)
    lam = 280.460 + 0.9856474 * d + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    rate = 0.9856474 + np.radians(0.9856003) * (1.915 * np.cos(g) + 0.040 * np.cos(2 * g))  # deg/day
    node = 73.6667 + 0.0139583 * (d / 365.25 + 150)  # 150: 1850 to 2000
    tilt = np.arctan(np.tan(np.radians(SOLAR_INCLINATION)) * np.cos(np.radians(lam - node)))
    px = rate * 3600 / header["CDELT1"]
    return float(px * np.sin(tilt)), float(px * np.cos(tilt))


def load(path):
    """(image, header) as described in the module docstring; the header's
    roll is set to 0 to match."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path) as h:
            raw = h[0].data.astype(np.float32)
            header = h[0].header.copy()
    img = (raw - header["OFFSET"]) / header["EXPTIME"]
    img[raw <= 0] = np.nan  # missing telemetry blocks
    img = north_up(img, header)
    with np.errstate(invalid="ignore"):
        img[img < DARK_FRAC * np.nanmedian(img)] = np.nan
    header["CROTA1"] = header["CROTA2"] = 0.0
    return img, header
