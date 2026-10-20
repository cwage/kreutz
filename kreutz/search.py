"""Shift-and-stack search for faint movers in a run of CCOR-1 frames.

For every trial velocity, the last N significance maps are shifted back
along that motion and summed: a body moving at that velocity lands on the
same pixel in every shifted frame and adds up (signal x N, noise x sqrt N),
while noise and anything moving differently doesn't. Each frame's
significance is clipped first, so a single-frame outlier (cosmic ray, a
star caught in one frame) can't make a track by itself.

Velocities and positions are in Sungrazer report coordinates (x right, y
down from the top, px per 15-min frame) so they compare directly with
reports. The priors come from the 81 Kreutz tracks reported in CCOR-1
images in July-August 2025: all on the Sun's right, 460-1150 px from its
center, moving 0.2-11.3 px/frame within -37..72 deg of straight at it.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from .frames import CADENCE, INSTRUMENT, Sequence

CLIP_SIGMA = 5.0
SPEED_MAX = 12.5
# A window may be missing this many frames (a 30-minute gap is a single
# missing frame, and those are the common kind: 13 of the 14 gaps in the
# July 2025 benchmark week). The frames present are stacked at their true
# time offsets; the score is normalized by how many there are.
MAX_MISSING = 1
# Below this a "track" is something that doesn't move: a pixel that
# flickers for a frame or two (first live alert, 2026-10-05 00:00, was
# one). Of 81 Kreutz tracks reported in July-August 2025 only a couple
# were slower.
SPEED_MIN = 1.0
V_STEP = 0.5
# Where in the frame, relative to the Sun, a search looks (report coords).
# The inner edge: inside ~460 px the corona is striped with fine radial
# rays, and a track pointed at the Sun stacks a ray into a "detection"
# (live alert 2026-10-05 22:15, r=430 px). Every Kreutz comet reported in
# CCOR-1 in July-August 2025 was first seen at 459 px or farther. (Those
# distances were measured about the WCS reference pixel, which is 40 px
# from the Sun; the annulus is about the Sun now, frames.sun_pixel.)
# (Those are CCOR-1's numbers; each instrument carries its own, instrument.py.)
R_MIN, R_MAX = INSTRUMENT.r_min, INSTRUMENT.r_max
ANGLE_CENTER = INSTRUMENT.angle_center  # deg counterclockwise from straight right of the Sun
ANGLE_MAX = INSTRUMENT.angle_max        # deg either side of that
# Stay this far inside the frame: the outermost rows and columns carry
# vignetting gradients that survive background subtraction.
EDGE_PX = 30
# Most frames in a window have to contribute: noise that stacks to a high
# score usually does it with one hot frame and the rest near zero. The
# search keeps anything where at least half the frames clear 1 sigma, so
# the stricter rule (consistent() below) can be tuned on stored runs; a
# rule like "every frame >= 1.5" is fine at 4 frames but throws away
# faint comets at 8, where a 2-sigma comet dips under 1.5 now and then.
FRAME_KEEP = 1.0
FRAME_KEEP_FRAC = 0.5
FRAME_GOOD = 1.0
FRAME_GOOD_FRAC = 0.75


def consistent(per_frame):
    """Enough of a window's frames show the object (see FRAME_GOOD*)."""
    per = np.asarray(per_frame)
    return np.sum(per >= FRAME_GOOD) >= np.ceil(FRAME_GOOD_FRAC * len(per))
# Allowed motion relative to the direction to the Sun.
HEADING_MIN, HEADING_MAX = -60, 95
# Stars are subtracted in the frames (see frames.STAR_FRAMES); whatever
# survives near their drift is still more likely star residue than comet.
STAR_EXCLUDE = 2.0


@dataclass
class Candidate:
    frame: int          # index of the last frame in the window
    x: float            # position in that frame, report coords
    y: float
    vx: float           # px per frame, report coords
    vy: float
    score: float        # stacked significance, ~N(0,1) on pure noise
    per_frame: list     # each frame's significance along the track


def velocity_grid(star_v):
    v = np.arange(-SPEED_MAX, SPEED_MAX + 1e-9, V_STEP)
    vx, vy = np.meshgrid(v, v)
    sp = np.hypot(vx, vy)
    keep = (sp >= SPEED_MIN) & (sp <= SPEED_MAX)
    keep &= np.hypot(vx - star_v[0], vy - star_v[1]) > STAR_EXCLUDE
    return np.stack([vx[keep], vy[keep]], axis=1)


def _roi(seq: Sequence):
    """Bounding box (rows, cols) of the search region, and a mask over it."""
    sr, sc = seq.sun_rc()
    H, W = seq.shape
    rows = np.arange(H)
    cols = np.arange(W)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    x, y = seq.rc_to_report(rr, cc)
    sx, sy = seq.rc_to_report(sr, sc)
    dx, dy = x - sx, -(y - sy)  # y up for the angle
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    off = (ang - ANGLE_CENTER + 180) % 360 - 180
    m = (r >= R_MIN) & (r <= R_MAX) & (np.abs(off) <= ANGLE_MAX)
    m &= (rr >= EDGE_PX) & (rr < H - EDGE_PX) & (cc >= EDGE_PX) & (cc < W - EDGE_PX)
    rs, cs = np.nonzero(m)
    box = (slice(rs.min(), rs.max() + 1), slice(cs.min(), cs.max() + 1))
    return box, m[box]


def window_frames(seq: Sequence, last: int, n: int):
    """Indices of the frames in the n-frame window ending at `last` and
    how many frames before `last` each is, oldest first; raises if more
    than MAX_MISSING are absent."""
    idx, back = [], []
    i = last
    while i >= 0:
        k = round((seq.times[last] - seq.times[i]) / CADENCE)
        if k > n - 1:
            break
        idx.append(i)
        back.append(k)
        i -= 1
    if len(idx) < n - MAX_MISSING:
        span = seq.times[last] - seq.times[max(last - (n - 1), 0)]
        raise ValueError(f"window spans a data gap ({span})")
    return idx[::-1], np.array(back[::-1])


def _offsets(vel, back, pad):
    """(velocity, frame) -> (row, col) slice origin in the padded frames;
    `back` is how many frames before the last each frame is."""
    drow, dcol = -vel[:, 1], vel[:, 0]  # report (vx, vy) -> array (drow, dcol)
    r0 = pad - np.round(drow[:, None] * back[None, :]).astype(int)
    c0 = pad - np.round(dcol[:, None] * back[None, :]).astype(int)
    return r0, c0


def _stack(frames, vel, back, pad, h, w):
    """Best stacked sum over velocities per pixel, and which velocity.
    On the GPU when there is one (~50x faster), numpy otherwise."""
    n = len(frames)
    r0, c0 = _offsets(vel, back, pad)
    if _torch() is not None:
        torch = _torch()
        F = torch.from_numpy(np.stack(frames)).cuda()
        best = torch.full((h, w), float("-inf"), device="cuda")
        best_v = torch.zeros((h, w), dtype=torch.int32, device="cuda")
        for vi in range(len(vel)):
            acc = F[0, r0[vi, 0]:r0[vi, 0] + h, c0[vi, 0]:c0[vi, 0] + w].clone()
            for k in range(1, n):
                acc += F[k, r0[vi, k]:r0[vi, k] + h, c0[vi, k]:c0[vi, k] + w]
            better = acc > best
            best = torch.where(better, acc, best)
            best_v = torch.where(better, torch.full_like(best_v, vi), best_v)
        return best.cpu().numpy(), best_v.cpu().numpy()
    best = np.full((h, w), -np.inf, np.float32)
    best_v = np.zeros((h, w), np.int32)
    for vi in range(len(vel)):
        acc = np.zeros((h, w), np.float32)
        for k in range(n):
            acc += frames[k][r0[vi, k]:r0[vi, k] + h, c0[vi, k]:c0[vi, k] + w]
        better = acc > best
        best[better] = acc[better]
        best_v[better] = vi
    return best, best_v


_TORCH = []


def _torch():
    """torch with a usable CUDA device, or None."""
    if not _TORCH:
        try:
            import torch
            _TORCH.append(torch if torch.cuda.is_available() else None)
        except ImportError:
            _TORCH.append(None)
    return _TORCH[0]


def search(seq: Sequence, last: int, n: int, threshold: float):
    """Candidates in the window of n frames ending at frame `last`."""
    idx, back = window_frames(seq, last, n)
    box, mask = _roi(seq)
    drow, dcol = seq.star_drift(last)
    vel = velocity_grid((dcol, -drow))
    pad = int(np.ceil(SPEED_MAX * (n - 1))) + 1
    frames = []
    for i in idx:
        s = seq.significance(i)[box]
        s = np.clip(np.nan_to_num(s, nan=0.0), -CLIP_SIGMA, CLIP_SIGMA)
        frames.append(np.pad(s, pad))
    h, w = mask.shape
    best, best_v = _stack(frames, vel, back, pad, h, w)
    best /= np.sqrt(len(frames))
    best[~mask] = -np.inf

    peaks = (best >= threshold) & (best == maximum_filter(best, size=9))
    sr, sc = seq.sun_rc()
    sx, sy = seq.rc_to_report(sr, sc)
    out = []
    for r, c in zip(*np.nonzero(peaks)):
        vx, vy = vel[best_v[r, c]]
        x, y = seq.rc_to_report(r + box[0].start, c + box[1].start)
        to_sun = np.degrees(np.arctan2(sy - y, sx - x))
        heading = (np.degrees(np.arctan2(vy, vx)) - to_sun + 180) % 360 - 180
        if not HEADING_MIN <= heading <= HEADING_MAX:
            continue
        per = []
        for fr, b in zip(frames, back):
            rr = pad + r - int(round(-vy * b))
            cc = pad + c - int(round(vx * b))
            per.append(round(float(fr[rr, cc]), 1))
        if np.sum(np.asarray(per) >= FRAME_KEEP) < np.ceil(FRAME_KEEP_FRAC * len(per)):
            continue
        out.append(Candidate(last, float(x), float(y), float(vx), float(vy),
                             round(float(best[r, c]), 2), per))
    return _dedupe(sorted(out, key=lambda c: -c.score))


# Candidates closer than this are one object: a bright source peaks at
# several neighboring pixels and velocities at once.
DEDUPE_PX = 10


def _dedupe(cands):
    kept = []
    for c in cands:
        if all(np.hypot(c.x - k.x, c.y - k.y) > DEDUPE_PX for k in kept):
            kept.append(c)
    return kept
