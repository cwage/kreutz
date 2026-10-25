"""Orbit-constrained search: shift-and-stack along Kreutz orbits.

search.py tries every velocity at every pixel of the sector, about 760
million trials per window, and the noise among that many trials sets the
threshold. A Kreutz comet's track is fixed by its orbit, and the group's
orbits are nearly one (orbit.py: nine comets of October 2026 fitted to a
pixel or so with the node within 5..17 deg and the argument of perihelion
tied to it, peri = 79.4 + 0.8 node, give or take 1.3 deg). So here the
trials are orbits: a family of node values with the perihelion argument
tied to it, a small offset either side, and every perihelion time that puts
the comet in the frame, a minute apart. For each, the comet's predicted
position in every frame of the window comes from the orbit and the frame's
WCS, and the significance there is summed, as search.py sums along a
straight line. A few hundred thousand trials instead of 760 million; and
the predicted path curves and speeds up with the comet, so long windows
don't smear.

Needs a helioprojective WCS (CCOR-1, CCOR-2). Candidates come out as
search.Candidate, positions and velocities in report coordinates.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import maximum_filter

from . import orbit
from .frames import Sequence
from .instrument import CURRENT as INSTRUMENT
from .search import CLIP_SIGMA, Candidate, window_frames

NODES = np.arange(4.0, 18.01, 1.0)       # deg
PERI_OFFSETS = (-2.0, 0.0, 2.0)          # deg about the node relation
PERI_A, PERI_B = 79.41, 0.798            # peri = A + B * node (orbit fits of 2026-10)
INCLINATION = 143.9
T_STEP = timedelta(minutes=1)
# perihelion times searched: the comet at the window's last frame between
# these before perihelion (CCOR-1's outer edge is ~2 days out, the occulter
# a few hours)
TAU_MAX = timedelta(days=3.0)
TAU_MIN = timedelta(hours=2)
SEP_PX = 12  # candidates closer than this at the last frame are one
# One orbit through a report's points and our track: the same comet fits
# within ~10-170" (Collins' canvas-offset points the worst), two comets of
# the same few days not within ~1900" (2026-10-23..25, CCOR-1 and CCOR-2).
SAME_RMS = 400.0


def family():
    """The orbit templates searched (T is set per trial)."""
    out = []
    for node in NODES:
        for dp in PERI_OFFSETS:
            out.append(orbit.Elements(T=None, node=float(node), peri=float(PERI_A + PERI_B * node + dp),
                                      i=INCLINATION))
    return out


_WCS = {}


def _wcs(path):
    if path not in _WCS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = fits.getheader(path, 1)
            _WCS[path] = (WCS(h), h["NAXIS2"], h["NAXIS1"])
    return _WCS[path]


def _sky(e, dt_days, obs):
    """Helioprojective (deg) of orbit e at times dt_days from perihelion
    (array), seen from heliocentric `obs` (3,)."""
    W = 3 * np.sqrt(orbit.K2 / (2 * e.q ** 3)) * dt_days
    y = np.cbrt(W / 2 + np.sqrt(W ** 2 / 4 + 1))
    s = y - 1 / y
    r = e.q * (1 + s ** 2)
    nu = 2 * np.arctan(s)
    c = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(r)], 1) @ orbit._orbit_to_icrs(e).T
    u = -obs / np.linalg.norm(obs)
    v = c - obs
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    n = orbit.SOLAR_POLE - (u @ orbit.SOLAR_POLE) * u
    n /= np.linalg.norm(n)
    w = np.cross(u, n)
    return np.degrees(np.arctan2(v @ w, v @ u)), np.degrees(np.arcsin(np.clip(v @ n, -1, 1)))


def orbit_search(seq: Sequence, last: int, n: int, threshold: float):
    """Candidates for the window of n frames ending at `last`."""
    idx, back = window_frames(seq, last, n)
    if len(_WCS) > 4 * len(seq.paths):  # a live watcher's frames move on
        keep = set(seq.paths)
        for p in [p for p in _WCS if p not in keep]:
            del _WCS[p]
    H, W = seq.shape
    t_last = seq.times[last]
    where = "l1" if INSTRUMENT.key == "ccor2" else "earth"
    # an incoming comet reaches perihelion after t_last: T = t_last + tau,
    # tau from TAU_MIN to TAU_MAX in steps of T_STEP (days)
    taus = np.arange(TAU_MIN / T_STEP, TAU_MAX / T_STEP + 1) * (T_STEP / timedelta(days=1))
    fams = family()
    obs = orbit.observer([seq.times[i] for i in idx], where)
    score = np.zeros((len(fams), len(taus)), np.float32)
    per = np.zeros((len(fams), len(taus), len(idx)), np.float32)
    pos = {}
    for k, (i, o) in enumerate(zip(idx, obs)):
        s = np.clip(np.nan_to_num(seq.significance(i), nan=0.0), -CLIP_SIGMA, CLIP_SIGMA)
        w, _, _ = _wcs(seq.paths[i])
        dt_frame = (seq.times[i] - t_last) / timedelta(days=1)  # <= 0
        for m, e in enumerate(fams):
            tx, ty = _sky(e, dt_frame - taus, o)
            col, row = w.wcs_world2pix(tx, ty, 0)
            ir, ic = np.round(row).astype(int), np.round(col).astype(int)
            ok = (ir >= 0) & (ir < H) & (ic >= 0) & (ic < W)
            v = np.zeros(len(taus), np.float32)
            v[ok] = s[ir[ok], ic[ok]]
            score[m] += v
            per[m, :, k] = v
            if i == last:
                pos[m] = (col, row, ok)
            elif k == len(idx) - 2:
                pos.setdefault(("prev", m), (col, row, seq.times[i]))
    score /= np.sqrt(len(idx))
    # only trials with the comet inside the frame at the last frame
    for m in range(len(fams)):
        score[m, ~pos[m][2]] = -np.inf
    peaks = (score >= threshold) & (score == maximum_filter(score, size=(3, 31), mode="nearest"))
    cands = []
    for m, j in zip(*np.nonzero(peaks)):
        col, row, _ = pos[m]
        pcol, prow, pt = pos[("prev", m)]
        steps = max((t_last - pt) / (seq.times[last] - seq.times[idx[-2]]) if len(idx) > 1 else 1, 1)
        x, y = seq.rc_to_report(row[j], col[j])
        px, py = seq.rc_to_report(prow[j], pcol[j])
        cands.append(Candidate(frame=last, x=float(np.round(x)), y=float(np.round(y)),
                               vx=float(np.round((x - px) / steps * 2) / 2), vy=float(np.round((y - py) / steps * 2) / 2),
                               score=float(score[m, j]), per_frame=[round(float(a), 1) for a in per[m, j]]))
    # one candidate per object: the best within SEP_PX at the last frame
    cands.sort(key=lambda c: -c.score)
    out = []
    for c in cands:
        if all(np.hypot(c.x - o.x, c.y - o.y) > SEP_PX for o in out):
            out.append(c)
    return out


def _where(inst=None):
    return "l1" if (inst or INSTRUMENT).key == "ccor2" else "earth"


_LISTING = {}


def _frame_near(inst, t):
    """`inst`'s frame within half a cadence of t on disk, or None."""
    best = None
    for d in sorted({(t - inst.cadence / 2).date(), (t + inst.cadence / 2).date()}):
        key = (inst.key, d)
        if key not in _LISTING or d >= datetime.utcnow().date() - timedelta(days=1):  # today's still growing
            _LISTING[key] = [(datetime.strptime(inst.name_re.search(p.name).group(1), "%Y%m%dT%H%M%S"), p)
                             for p in (inst.frames_dir / f"{d:%Y/%m/%d}").glob(inst.glob())
                             if inst.name_re.search(p.name)]
        for ft, p in _LISTING[key]:
            if abs(ft - t) <= inst.cadence / 2 and (best is None or abs(ft - t) < abs(best[0] - t)):
                best = (ft, p)
    return best[1] if best else None


def sky_points(pts, inst=None):
    """[(time, x, y)] in `inst`'s (default ours) report coordinates to
    (times, Tx, Ty) in deg, each through the WCS of that instrument's frame
    within half a cadence of it; points with no such frame on disk are
    dropped."""
    inst = inst or INSTRUMENT
    ts, tx, ty = [], [], []
    for t, x, y in pts:
        path = _frame_near(inst, t)
        if path is None:
            continue
        w, H, _ = _wcs(path)
        lon, lat = w.wcs_pix2world([[x, H - 1 - y]], 0)[0]
        ts.append(t)
        tx.append(lon)
        ty.append(lat)
    return ts, tx, ty


def orbit_of(pts):
    """The family orbit (node on NODES, perihelion argument on the node
    relation, perihelion time free) that best fits [(time, x, y)], and its
    rms in arcsec; None with fewer than 3 points on frames on disk."""
    ts, tx, ty = sky_points(pts)
    if len(ts) < 3:
        return None
    best = None
    for node in NODES:
        for ahead in (6, 24):
            e0 = orbit.Elements(T=ts[-1] + timedelta(hours=ahead), node=float(node),
                                peri=float(PERI_A + PERI_B * node), i=INCLINATION)
            e, rms = orbit.fit(ts, tx, ty, e0, _where(), free=("T",))
            if best is None or rms < best[1]:
                best = (e, rms)
    return best


def joint_rms(e, pts, others, inst=None):
    """The rms (arcsec) of one orbit, started from e with its perihelion
    time free, through our points and `others` (in `inst`'s coordinates,
    default ours), each seen from its own instrument: under SAME_RMS, one
    comet."""
    a, b = sky_points(pts), sky_points(others, inst)
    ts, tx, ty = (x + y for x, y in zip(a, b))
    if len(ts) < 3 or not b[0]:
        return float("inf")
    where = [_where()] * len(a[0]) + [_where(inst)] * len(b[0])
    return orbit.fit(ts, tx, ty, e, where, free=("T",))[1]
