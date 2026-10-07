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
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .frames import PSF_SIGMA, SURROUND_SIGMA, Sequence

FOLLOW_FRAMES = 10
FOLLOW_R = 4        # px around each prediction to look for the peak
PEAK_SIGMA = 4.0    # a frame counts when the peak stands this far above
STAR_MATCH = 0.4    # px/frame from the star drift: moving with the stars


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


def follow(seq: Sequence, last: int, x: float, y: float, vx: float, vy: float):
    """Follow the object back through raw frames and fit its motion."""
    H = seq.shape[0]
    pts = []
    px, py = x, y
    for k in range(FOLLOW_FRAMES):
        i = last - k
        if i < 0:
            break
        # frames before `last`, by time (a missing frame makes it jump)
        kt = round((seq.times[last] - seq.times[i]) / (seq.times[1] - seq.times[0]))
        if kt > FOLLOW_FRAMES:
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
            dt = (seq.times[i] - seq.times[i - 1]) / (seq.times[1] - seq.times[0])
            px, py = px - vx * dt, py - vy * dt
    if len(pts) >= 3:
        t = np.array([p[0] for p in pts])
        vx = float(np.polyfit(t, [p[1] for p in pts], 1)[0])
        vy = float(np.polyfit(t, [p[2] for p in pts], 1)[0])
    dr, dc = seq._displacement(last - 1, last)(H - 1 - y, x)
    return Motion(vx, vy, pts, float(dc), float(-dr))
