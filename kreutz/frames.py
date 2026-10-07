"""CCOR-1 L1b frames as background-subtracted significance maps.

Coordinates: arrays are (row, col) as stored in the FITS file. Sungrazer
reports measure (x, y) with the origin at the upper left of the image as
displayed (solar north up), so x = col and y = (rows - 1) - row; verified
by stacking crops along confirmed comets' reported positions, where only
this convention puts the comet at the center.
"""

import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import (binary_dilation, gaussian_filter, map_coordinates,
                           maximum_filter, uniform_filter)
from scipy.spatial import cKDTree

from . import DATA

L1B = DATA / "ccor1" / "l1b"
CADENCE = timedelta(minutes=15)

# Background = median of the frames before this one (only the past: a
# real-time detector doesn't have the future), skipping the most recent:
# a comet moving a few px per frame would otherwise sit in its own
# background and subtract itself out. 16 frames is four hours.
BG_FRAMES = 16
BG_SKIP = 1
# Stars drift across the field together (the Sun moves ~1 deg/day against
# them). They're modeled from this many recent frames, each shifted along
# the drift so the stars line up, then median-combined: a comet moving any
# other way doesn't survive the median, the stars do, and the model is
# subtracted. Without it bright stars stack into tracks at a whole range
# of velocities near the drift.
STAR_FRAMES = 8
# Stars used for alignment: point sources at least this significant, and
# outside this radius from Sun center (inside it, the corona's structure
# makes too many false points).
STAR_Z = 10
STAR_MIN_R = 350
STAR_MASK_R = 4

# Matched filter for a point source (the PSF is ~2 px across), minus a
# wider blur: a difference of Gaussians keeps points and cancels anything
# diffuse (streamers, CME fronts, gradients at the detector edges), which
# otherwise stacks into "tracks" at any slow velocity.
PSF_SIGMA = 1.2
SURROUND_SIGMA = 4.0
# Distance from the occulter/vignetted region (NaN in L1b) within which
# nothing is trusted: the point filter reaches ~3 px, so a little more.
# It was 24 before the noise estimate learned to ignore missing pixels,
# and that hid comet 0080 (2025-07-24), 6-9 px from the field's edge.
EDGE_MARGIN = 5
BORDER = 12
# Local noise is estimated over this box (the noise rises steeply toward
# the occulter, so one global sigma would be wrong almost everywhere).
NOISE_BOX = 41

_NAME = re.compile(r"CCOR1_1B_(\d{8}T\d{6})_V\d+_NC\.fits$")


def frame_time(path):
    return datetime.strptime(_NAME.search(str(path)).group(1), "%Y%m%dT%H%M%S")


def frame_paths(start, end):
    """L1b files with observation time in [start, end], in time order, on
    the standard 15-minute grid. CCOR-1 sometimes runs at 5-minute
    cadence for a while; the extra frames are dropped so every step
    between frames means the same motion."""
    paths = []
    d = start.date()
    while d <= end.date():
        # _NC is the standard product; special campaigns (e.g. 5-minute
        # cadence, 2025-07-18 03:45-06:15) come as _0C and stand in for it.
        paths += (L1B / f"{d:%Y/%m/%d}").glob("CCOR1_1B_*_NC.fits")
        d += timedelta(days=1)
    paths = [p for p in paths if start <= frame_time(p) <= end
             and frame_time(p).minute % 15 == 0]
    return sorted(paths, key=frame_time)


def load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the files carry a BLANK on float data
        with fits.open(path) as h:
            return h[1].data.astype(np.float32), h[1].header


def _cache_key():
    """Settings that shape a significance map; maps made with other
    settings live in another directory and are never served."""
    key = (f"bg{BG_FRAMES}s{BG_SKIP}-st{STAR_FRAMES}z{STAR_Z}r{STAR_MIN_R}m{STAR_MASK_R}"
           f"-psf{PSF_SIGMA}-{SURROUND_SIGMA}-nb{NOISE_BOX}-e{EDGE_MARGIN}b{BORDER}")
    return key.replace(".", "p")


SIG_CACHE = DATA / "sig"


def _cache_path(raw_path):
    """Where frame `raw_path`'s significance map is cached (float16). The
    map depends on the frames before it too, but only through the same
    archive, so the frame's name is a sufficient key."""
    t = frame_time(raw_path)
    return SIG_CACHE / _cache_key() / f"{t:%Y/%m/%d}" / (Path(raw_path).stem + ".npy")


class Sequence:
    """A run of consecutive frames with a lazily built significance map each."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.times = [frame_time(p) for p in self.paths]
        self._raw = {}
        self._sig = {}
        self._src = {}
        self._disp = {}
        self.header = load(self.paths[0])[1]
        self.shape = (self.header["NAXIS2"], self.header["NAXIS1"])

    def raw(self, i):
        if i not in self._raw:
            self._raw[i] = load(self.paths[i])[0]
        return self._raw[i]

    def significance(self, i):
        """Frame i minus its temporal-median background, PSF-filtered and
        divided by the local noise: roughly N(0,1) where there's nothing,
        and a point source shows as a positive peak. NaN off the detector
        (occulter, corners)."""
        if i in self._sig:
            return self._sig[i]
        cached = _cache_path(self.paths[i])
        if cached.exists():
            self._sig[i] = np.load(cached).astype(np.float32)
            return self._sig[i]
        # Raw frames older than the background window won't be needed again.
        for j in [j for j in self._raw if j < i - BG_SKIP - BG_FRAMES]:
            del self._raw[j]
        bg = self._background(i)
        d = self.raw(i) - bg
        bad = ~np.isfinite(d)
        d[bad] = 0
        d -= self._stars(i, bg)
        f = gaussian_filter(d, PSF_SIGMA) - gaussian_filter(d, SURROUND_SIGMA)
        # Robust local sigma: the box-mean of |f| scales to sigma for
        # Gaussian noise (E|x| = sigma * sqrt(2/pi)), and a lone source
        # barely moves a 41x41 mean. Averaged over valid pixels only: near
        # the edge of the usable field the box half-covers zero-filled
        # pixels, and a plain mean would underestimate the noise there and
        # inflate everything next to the edge, which is exactly where
        # comets enter the field.
        valid = (~bad).astype(np.float32)
        frac = uniform_filter(valid, NOISE_BOX)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = uniform_filter(np.abs(f) * valid, NOISE_BOX) / frac / np.sqrt(2 / np.pi)
        bad = binary_dilation(bad, iterations=EDGE_MARGIN)
        bad[:BORDER] = bad[-BORDER:] = True
        bad[:, :BORDER] = bad[:, -BORDER:] = True
        with np.errstate(divide="ignore", invalid="ignore"):
            s = f / sigma
        s[bad | ~np.isfinite(s)] = np.nan
        s[self._star_mask(i)] = np.nan
        self._sig[i] = s.astype(np.float32)
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp.npy")
        np.save(tmp, s.astype(np.float16))
        tmp.rename(cached)
        return self._sig[i]

    def _background(self, i):
        nb = [j for j in range(i - BG_SKIP - BG_FRAMES, i - BG_SKIP) if j >= 0]
        if len(nb) < BG_FRAMES // 2:
            raise ValueError(f"frame {i} has only {len(nb)} frames of history")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # all-NaN under the occulter
            return np.nanmedian(np.stack([self.raw(j) for j in nb]), axis=0)

    def _stars(self, i, bg):
        """The star field as it should appear in frame i: recent frames
        (minus the same static background) warped onto frame i by their
        measured star alignment, and median-combined."""
        H, W = self.shape
        rows = np.arange(H, dtype=np.float32)[:, None]
        cols = np.arange(W, dtype=np.float32)[None, :]
        layers = []
        for back in range(BG_SKIP + 1, BG_SKIP + 1 + STAR_FRAMES):
            j = i - back
            if j < 0:
                break
            dr, dc = self._displacement(j, i)(rows, cols)
            r = np.nan_to_num(self.raw(j) - bg)
            # Where a star sits in frame i, it sat at (p - displacement) in j.
            layers.append(map_coordinates(r, [rows - dr, cols - dc], order=1, cval=0.0))
        return np.median(np.stack(layers), axis=0)

    def _sources(self, i):
        """Bright point sources in frame i, (row, col) sub-pixel: stars,
        mostly. Straight from the raw frame through the point filter, so no
        background history is needed."""
        if i in self._src:
            return self._src[i]
        d = np.nan_to_num(self.raw(i))
        f = gaussian_filter(d, PSF_SIGMA) - gaussian_filter(d, SURROUND_SIGMA)
        sigma = uniform_filter(np.abs(f), NOISE_BOX) / np.sqrt(2 / np.pi)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.nan_to_num(f / sigma)
        H, W = self.shape
        sr, sc = self.sun_rc()
        m = 2 + BORDER
        pk = (z > STAR_Z) & (z == maximum_filter(z, 7))
        pk[:m] = pk[-m:] = False
        pk[:, :m] = pk[:, -m:] = False
        rows, cols = np.nonzero(pk)
        near_sun = np.hypot(rows - sr, cols - sc) < STAR_MIN_R
        rows, cols = rows[~near_sun], cols[~near_sun]
        yy, xx = np.mgrid[-2:3, -2:3]
        out = []
        for r, c in zip(rows, cols):
            w = np.clip(f[r - 2:r + 3, c - 2:c + 3], 0, None)
            if w.sum() > 0:
                out.append((r + (w * yy).sum() / w.sum(), c + (w * xx).sum() / w.sum()))
        self._src[i] = np.array(out).reshape(-1, 2)
        return self._src[i]

    def _displacement(self, j, i):
        """How the star field moved from frame j to frame i: a function
        (rows, cols) -> (drow, dcol), quadratic in position, fitted to
        matched stars.

        The drift isn't uniform or constant: it's ~11% faster 900 px out
        than near the Sun (the optics' plate scale), and it swings by
        ~10% over a day as GOES-19 circles Earth. So it's measured for each
        pair of frames rather than modeled.
        """
        key = (j, i)
        if key in self._disp:
            return self._disp[key]
        a, b = self._sources(j), self._sources(i)
        if len(a) < 20 or len(b) < 20:
            raise ValueError(f"too few stars to align frames {j} and {i}")
        # Coarse: the most common displacement among all nearby pairs.
        reach = 2.5 * abs(i - j) + 4
        pairs = cKDTree(b).query_ball_point(a, reach)
        vec = np.array([b[k] - p for p, ks in zip(a, pairs) for k in ks]).reshape(-1, 2)
        hist, er, ec = np.histogram2d(vec[:, 0], vec[:, 1], bins=int(reach * 4),
                                      range=[[-reach, reach], [-reach, reach]])
        hr, hc = np.unravel_index(np.argmax(hist), hist.shape)
        guess = np.array([(er[hr] + er[hr + 1]) / 2, (ec[hc] + ec[hc + 1]) / 2])
        # Fine: match each star near the guess, fit, clip, refit.
        dist, k = cKDTree(b).query(a + guess, distance_upper_bound=2.0)
        ok = np.isfinite(dist)
        p, d = a[ok], b[k[ok]] - a[ok]
        sr, sc = self.sun_rc()
        def design(r, c):
            u, v = (r - sr) / 1000, (c - sc) / 1000
            return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)
        X = design(p[:, 0], p[:, 1])
        keep = np.ones(len(p), bool)
        for _ in range(3):
            coef = np.linalg.lstsq(X[keep], d[keep], rcond=None)[0]
            res = np.hypot(*(d - X @ coef).T)
            keep = res < max(3 * np.std(res[keep]), 0.3)
        def field(r, c, coef=coef.astype(np.float32)):
            # Expanded rather than through design(): on a full frame the
            # (H, W, 6) design matrix is most of the cost.
            u = (np.asarray(r, np.float32) - np.float32(sr)) / 1000
            v = (np.asarray(c, np.float32) - np.float32(sc)) / 1000
            k = coef
            return tuple(k[0, a] + k[1, a] * u + k[2, a] * v + k[3, a] * u * u
                         + k[4, a] * u * v + k[5, a] * v * v for a in (0, 1))
        field.matched = int(keep.sum())
        field.rms = float(np.std(res[keep]))
        self._disp[key] = field
        return field

    def _star_mask(self, i):
        """Pixels around frame i's confirmed stars: sources that were also
        where the star drift says they'd have been two frames earlier. The
        star model leaves residue at the brightest ones, and a residue
        bright enough to clip in every frame stacks at a whole range of
        velocities."""
        H, W = self.shape
        mask = np.zeros((H, W), bool)
        cur = self._sources(i)
        prev = self._sources(i - 2)
        dr, dc = self._displacement(i - 2, i)(prev[:, 0], prev[:, 1])
        moved = np.stack([prev[:, 0] + dr, prev[:, 1] + dc], axis=1)
        dist, _ = cKDTree(moved).query(cur, distance_upper_bound=1.5)
        stars = cur[np.isfinite(dist)]
        yy, xx = np.mgrid[-STAR_MASK_R:STAR_MASK_R + 1, -STAR_MASK_R:STAR_MASK_R + 1]
        disk = yy ** 2 + xx ** 2 <= STAR_MASK_R ** 2
        for r, c in np.round(stars).astype(int):
            r0, c0 = r - STAR_MASK_R, c - STAR_MASK_R
            if r0 < 0 or c0 < 0 or r0 + disk.shape[0] > H or c0 + disk.shape[1] > W:
                continue
            mask[r0:r0 + disk.shape[0], c0:c0 + disk.shape[1]] |= disk
        return mask

    def star_drift(self, i):
        """Star-field velocity (rows, cols) per frame at frame i, at the
        middle of the search region (right of the Sun)."""
        sr, sc = self.sun_rc()
        dr, dc = self._displacement(i - 1, i)(sr, sc + 700)
        return float(dr), float(dc)

    def forget(self, before):
        """Drop cached frames, maps and alignments older than frame index
        `before` (long runs would otherwise hold every frame in memory)."""
        for d in (self._raw, self._sig, self._src):
            for j in [j for j in d if j < before]:
                del d[j]
        for key in [k for k in self._disp if k[0] < before]:
            del self._disp[key]

    def report_to_rc(self, x, y):
        """Sungrazer report (x, y) -> array (row, col)."""
        return self.shape[0] - 1 - y, x

    def rc_to_report(self, row, col):
        return col, self.shape[0] - 1 - row

    def sun_rc(self):
        """Sun center (row, col), from the WCS reference pixel (1-based)."""
        return self.header["CRPIX2"] - 1, self.header["CRPIX1"] - 1
