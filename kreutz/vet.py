"""Second look at a fired track, before anyone is woken for it.

The search's velocity is coarse (0.5 px/frame grid, and a star's
subtraction residue can pull it off by more), so a star slips through now
and then. Here the object is followed in the raw frames themselves, frame
by frame back from the last one, its centroid fitted with a straight line,
and that motion compared with the star drift measured at that spot. A
real comet moves its own way; a star moves exactly with the field.

Seen on the live feed 2026-10-05 01:15: search said (1.0, -2.0) px/frame,
the raw centroids said (1.84, -0.63), the local star drift was
(1.87, -0.61).

The same fit tells two more things the search's coarse velocity can't.
An object that isn't moving at all: the post-eclipse blob of 2026-10-08
07:45 sat within a quarter pixel for six frames while the search's stack
peaks wandered enough to pass as 3 px/frame (is_stationary). And one
moving against the Kreutz lane's flow (lane.off_flow): the stray-light
feature of the eclipse-season mornings crept right at 3 px/frame where
every comet heads left at 8.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from scipy.ndimage import gaussian_filter, label

from .frames import (BG_FRAMES, BG_SKIP, CADENCE, PSF_SIGMA, STAR_FRAMES, SURROUND_SIGMA, Sequence,
                     frame_paths)

FOLLOW_FRAMES = 10
FOLLOW_R = 4        # px around each prediction to look for the peak
PEAK_SIGMA = 4.0    # a frame counts when the peak stands this far above
STAR_MATCH = 0.4    # px/frame from the star drift: moving with the stars
STATIONARY = 1.0    # px/frame: measured over three frames or more and slower than this, it isn't moving


@dataclass
class Motion:
    vx: float
    vy: float
    points: list        # (frame index, x, y, peak sigma) in report coords
    star_vx: float
    star_vy: float

    @property
    def is_star(self):
        return np.hypot(self.vx - self.star_vx, self.vy - self.star_vy) < STAR_MATCH

    @property
    def measured(self):
        """vx, vy are fitted to the raw frames (else they're the search's)."""
        return len(self.points) >= 3

    @property
    def is_stationary(self):
        return self.measured and np.hypot(self.vx, self.vy) < STATIONARY


def follow(seq: Sequence, last: int, x: float, y: float, vx: float, vy: float):
    """Follow the object back through raw frames and fit its motion."""
    H = seq.shape[0]
    pts = []
    px, py = x, y
    for k in range(FOLLOW_FRAMES):
        i = last - k
        if i < 0:
            break
        # frames before `last`, by time (a missing frame makes it jump), in
        # the instrument's steps: not the first two frames' spacing, which
        # on LASCO is often a 24 or 36 minute gap, and not rounded (LASCO
        # runs at 10.6 minutes against its 12 for hours at a time)
        kt = (seq.times[last] - seq.times[i]) / CADENCE
        if kt > FOLLOW_FRAMES + 0.5:
            break
        r0, c0 = int(round(H - 1 - py)), int(round(px))
        half = FOLLOW_R + 8
        win = seq.raw(i)[r0 - half:r0 + half + 1, c0 - half:c0 + half + 1]
        if win.shape != (2 * half + 1, 2 * half + 1):
            break
        win = np.nan_to_num(win)
        f = gaussian_filter(win, PSF_SIGMA) - gaussian_filter(win, SURROUND_SIGMA)
        core = f[8:-8, 8:-8]  # the +-FOLLOW_R box around the prediction
        rr, cc = np.unravel_index(np.argmax(core), core.shape)
        sig = core[rr, cc] / (np.median(np.abs(f)) / 0.6745 + 1e-30)
        fx = c0 - FOLLOW_R + cc
        fy = H - 1 - (r0 - FOLLOW_R + rr)
        if sig >= PEAK_SIGMA:
            pts.append((-kt, float(fx), float(fy), float(sig)))
            px, py = fx, fy
        # step the prediction back to the next frame on disk with the
        # current best velocity
        if len(pts) >= 3:
            t = np.array([p[0] for p in pts])
            vx = np.polyfit(t, [p[1] for p in pts], 1)[0]
            vy = np.polyfit(t, [p[2] for p in pts], 1)[0]
        if i - 1 >= 0:
            dt = (seq.times[i] - seq.times[i - 1]) / CADENCE
            px, py = px - vx * dt, py - vy * dt
    if len(pts) >= 3:
        t = np.array([p[0] for p in pts])
        vx = float(np.polyfit(t, [p[1] for p in pts], 1)[0])
        vy = float(np.polyfit(t, [p[2] for p in pts], 1)[0])
    dr, dc = seq._displacement(last - 1, last)(H - 1 - y, x)
    return Motion(vx, vy, pts, float(dc), float(-dr))


# Two more looks, both at the significance maps along the track, for the
# artifacts that move just enough and in the right direction to pass the
# motion checks.
PROFILE_FRAMES = 12  # frames back from the alert's last, covering its linked windows
PROFILE_R = 3        # px around each prediction to take the peak from


def profile(seq: Sequence, last: int, x, y, vx, vy, n=PROFILE_FRAMES, r=PROFILE_R):
    """{k: (peak sigma, x, y)} for the frames k = 0..n-1 steps before
    `last` (by time, so a missing frame leaves a gap): the peak of the
    significance map within r px of where a straight track through (x, y)
    at (vx, vy) px/frame is k frames earlier, and where that peak is."""
    H, W = seq.shape
    out = {}
    for i in range(last, -1, -1):
        k = round((seq.times[last] - seq.times[i]) / CADENCE)
        if k >= n:
            break
        px, py = x - vx * k, y - vy * k
        r0, c0 = int(round(H - 1 - py)), int(round(px))
        if not (r <= r0 < H - r and r <= c0 < W - r):
            continue
        win = np.nan_to_num(seq.significance(i)[r0 - r:r0 + r + 1, c0 - r:c0 + r + 1])
        dr, dc = np.unravel_index(np.argmax(win), win.shape)
        out[k] = (float(win[dr, dc]), float(c0 - r + dc), float(H - 1 - (r0 - r + dr)))
    return out


# A ghost: an eclipse-season stray-light feature fixed in the instrument's
# frame, which comes back at the same pixels at the same time of day. The
# one at (1665..1651, 607) peaked 8.7 then 7.3 then 5.2 sigma at 02:15,
# 02:45 and 03:15 UT on 2026-10-16 and 8.6, 7.5, 4.9 on the 17th, and
# alerted on the 17th; another at (1228,1565) peaked at 08:00 both days. A
# comet is a day's travel away 24 h later, and nothing sits where it was.
GHOST_LOOKBACK = timedelta(days=1)
GHOST_MIN = 4.0    # a day earlier, the track's peak frame had to hold something this bright
GHOST_CORR = 0.7   # and the two nights' per-frame profiles have to rise and fall together
GHOST_FRAMES = 6   # frames the two profiles must share


def same_as_yesterday(seq: Sequence, last: int, x, y, vx, vy, paths=frame_paths):
    """Why the track looks like yesterday's at the same time of day, or
    None: the per-frame profile along it now and along the same pixels 24 h
    earlier, from the frames on disk, correlate and the earlier one has a
    real peak. None too when there aren't frames enough a day back."""
    t0 = seq.times[last] - GHOST_LOOKBACK
    warm = CADENCE * (BG_FRAMES + BG_SKIP + STAR_FRAMES + 2)
    ps = paths(t0 - warm - CADENCE * PROFILE_FRAMES, t0 + CADENCE / 2)
    if not ps:
        return None
    prev_seq = Sequence(ps)
    j = min(range(len(prev_seq.times)), key=lambda i: abs(prev_seq.times[i] - t0))
    if abs(prev_seq.times[j] - t0) > CADENCE / 2:
        return None
    return ghost_verdict(profile(seq, last, x, y, vx, vy), profile(prev_seq, j, x, y, vx, vy))


def ghost_verdict(now, prev):
    """same_as_yesterday's test on two profiles (profile())."""
    ks = sorted(set(now) & set(prev))
    if len(ks) < GHOST_FRAMES:
        return None
    a, b = np.array([now[k][0] for k in ks]), np.array([prev[k][0] for k in ks])
    if b.max() < GHOST_MIN or a.std() == 0 or b.std() == 0:
        return None
    corr = float(np.corrcoef(a, b)[0, 1])
    if corr < GHOST_CORR:
        return None
    return f"same as 24 h earlier ({b.max():.1f} sigma there then, profiles correlate {corr:.2f})"


# A line: a comet is a compact peak a few px across in the significance map,
# where stray-light streaks and streamer rays run straight across it. The
# 2026-10-17 07:45 alert was a streak crossing the field 05:15-06:45; the
# 00:30 near-miss a radial ray brightening under the track.
SHAPE_R = 25       # px around the peak looked at
LINE_LEN = 15.0    # px: a component through the peak, above half of it, longer than this is a line
LINE_RATIO = 4.0   # or more elongated than this
# and it has to look like that in more than one frame: a comet's peak at 3 or
# 4 sigma sometimes merges with a noise bump into a short bar. The faint
# CCOR-2 comet of 2026-10-24 (Cerati/Ruan) was a point in 11 of 12 frames
# and a 9 px, 4.1:1 bar in the 12:00 one, its brightest when it alerted;
# judged on that frame alone it was vetoed. The five CCOR-1 artifacts
# vetoed 2026-10-21..24 were lines in 2 or 3 of their frames above 3 sigma,
# the two CCOR-2 stripes in 5 or more.
LINE_FRAME_SIGMA = 3.0  # frames along the track with a peak this bright are looked at
LINE_FRAMES = 2         # this many of them a line, it's a line

# The track's own stack (line_stack): the last STACK_N significance maps
# shifted along the track and summed, as the search sums them. A point
# source lands on one pixel and stacks to a blob; a stray-light stripe or
# ray lying along the track stacks to a line, however the frames are
# shifted, and CCOR-2's radial striping moves exactly like a sungrazer
# in heading and speed, so nothing kinematic rejects it. Over every alert
# on record (2026-10-08..22): the 15 real CCOR-1 comets' stacks were at
# most 2.8:1 and 13 px long above STACK_SIGMA; the eight CCOR-2 striping
# alerts of 2026-10-21/22 were 6.7:1 to 30:1 and 24 to 98 px, and six
# CCOR-1 false alerts 4.5:1 to 10.7:1 and 16 to 29 px.
STACK_N = 8
STACK_HALF = 30
STACK_SIGMA = 3.0        # the bright region is the stack above this, per unit stack
STACK_LINE_RATIO = 4.0   # the component through the peak, more elongated than this
STACK_LINE_LEN = 20.0    # and longer than this (4 sigma along its long axis), is a line


def extent(sig, r0, c0):
    """(length px, elongation) of the connected region of sig above half
    of its value at (r0, c0) that contains (r0, c0): the length of a
    uniform bar with the region's spread along its major axis, and the
    ratio of its spreads along the two axes."""
    peak = sig[r0, c0]
    lab, _ = label(np.nan_to_num(sig) >= peak / 2)
    rr, cc = np.nonzero(lab == lab[r0, c0])
    if len(rr) < 3:
        return float(len(rr)), 1.0
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([rr, cc]))))
    major, minor = max(ev[1], 1e-6), max(ev[0], 1 / 12)  # a pixel's own spread is 1/12
    return float(np.sqrt(12 * major)), float(np.sqrt(major / minor))


def stack_shape(seq: Sequence, last: int, x, y, vx, vy, n=STACK_N, half=STACK_HALF):
    """The track's stack through frame `last`: (peak within 3 px of the
    track, axis ratio, length, area) of the component above STACK_SIGMA
    that holds the peak, in a (2*half+1) square around the track's
    position in frame `last`. The stack is divided by sqrt(frames), so it
    reads in sigma like the search's score."""
    H, W = seq.shape
    size = 2 * half + 1
    stack = np.zeros((size, size), np.float32)
    frames = 0
    for i in range(max(0, last - n + 1), last + 1):
        k = round((seq.times[i] - seq.times[last]) / CADENCE)
        r0, c0 = int(round(H - 1 - (y + vy * k))), int(round(x + vx * k))
        if r0 - half < 0 or c0 - half < 0 or r0 + half + 1 > H or c0 + half + 1 > W:
            continue
        stack += np.nan_to_num(seq.significance(i)[r0 - half:r0 + half + 1, c0 - half:c0 + half + 1])
        frames += 1
    if not frames:
        return 0.0, 1.0, 0.0, 0
    stack /= np.sqrt(frames)
    win = stack[half - 3:half + 4, half - 3:half + 4]
    pr, pc = np.unravel_index(np.argmax(win), win.shape)
    pr, pc = pr + half - 3, pc + half - 3
    peak = float(stack[pr, pc])
    labels, _ = label(stack > STACK_SIGMA)
    comp = labels == labels[pr, pc] if labels[pr, pc] else np.zeros_like(labels, bool)
    area = int(comp.sum())
    if area < 3:
        return peak, 1.0, 0.0, area
    ys, xs = np.nonzero(comp)
    ev = np.linalg.eigvalsh(np.cov(np.stack([xs, ys])))
    ratio = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-6)))
    return peak, ratio, float(4 * np.sqrt(max(ev[1], 0.0))), area


def line_stack(seq: Sequence, last: int, x, y, vx, vy):
    """Why the track's own stack is a line rather than a point, or None."""
    _, ratio, length, _ = stack_shape(seq, last, x, y, vx, vy)
    if ratio >= STACK_LINE_RATIO and length >= STACK_LINE_LEN:
        return f"a line in its own stack: {length:.0f} px long, {ratio:.0f}:1"
    return None


def line_like(seq: Sequence, last: int, x, y, vx, vy):
    """Why the track shows a line rather than a point, or None: in
    LINE_FRAMES or more of its frames with a peak of LINE_FRAME_SIGMA, as
    long as its brightest reaches PEAK_SIGMA."""
    prof = profile(seq, last, x, y, vx, vy)
    if not prof or max(p[0] for p in prof.values()) < PEAK_SIGMA:
        return None
    H, W = seq.shape
    lines, looked = [], 0
    for i in range(last, -1, -1):
        k = round((seq.times[last] - seq.times[i]) / CADENCE)
        if k not in prof or prof[k][0] < LINE_FRAME_SIGMA:
            continue
        looked += 1
        _, px, py = prof[k]
        r0, c0 = int(round(H - 1 - py)), int(round(px))
        rs, cs = slice(max(r0 - SHAPE_R, 0), r0 + SHAPE_R + 1), slice(max(c0 - SHAPE_R, 0), c0 + SHAPE_R + 1)
        length, ratio = extent(seq.significance(i)[rs, cs], r0 - rs.start, c0 - cs.start)
        if length > LINE_LEN or ratio > LINE_RATIO:
            lines.append((prof[k][0], length, ratio, seq.times[i]))
    if len(lines) < LINE_FRAMES:
        return None
    _, length, ratio, t = max(lines, key=lambda a: a[0])
    return (f"a line, not a point: {length:.0f} px long, {ratio:.1f}:1, in the {t:%H:%M} frame "
            f"(a line in {len(lines)} of {looked} frames)")
