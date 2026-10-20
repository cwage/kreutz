"""Lane-constrained search: shift-and-stack along the Kreutz flow.

All Kreutz comets are pieces of one parent on nearly the same orbit, so in
CCOR-1 they all come in along the same lane at nearly the same speed for a
given spot (Aug-Oct 2026: entering near (1950, 1125), 9 px/frame, curving
in to (1450, 1030) at 12 px/frame). search.py tries ~1900 velocities over
the whole right half of the frame; here the velocity is known from where a
pixel is, up to a small offset, and only the lane is searched. Fewer trials
means fewer noise peaks at a given threshold, so the threshold can go
down, and the predicted path follows the comet as it speeds up, so the
stack can be 16 frames (4 hours) long without smearing.

The flow field is built from observed comet tracks (lane-samples.json, shipped
with the package; lane-samples-ccor2.json for CCOR-2 once it has one: every window of the bright lane tracks an archive
search over Aug-Oct 2026 found): each
pixel's velocity is the distance-weighted mean of the nearest samples'.
The band searched is everywhere within BAND_PX of a sample.
"""

import json
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter
from scipy.spatial import cKDTree

from .frames import INSTRUMENT, Sequence
from .search import CLIP_SIGMA, FRAME_KEEP, FRAME_KEEP_FRAC, Candidate, _dedupe, window_frames

BAND_PX = 40
K_NEAREST = 20
DV = np.arange(-1.5, 1.51, 0.5)  # allowed offset from the flow, px/frame


@dataclass
class Field:
    rows: slice           # bounding box of the band, array coords
    cols: slice
    band: np.ndarray      # bool, (h, w) inside the box
    vrow: np.ndarray      # flow per frame, array coords (row, col), (h, w)
    vcol: np.ndarray
    tree: object          # cKDTree of the samples' (row, col), for in_lane
    svr: np.ndarray       # the samples' flow, array coords
    svc: np.ndarray


def field_or_none(shape):
    """The instrument's lane field, or None while it has no samples yet
    (callers that only use the lane to extrapolate known comets backward
    do without)."""
    return build_field(shape) if INSTRUMENT.lane_samples.exists() else None


def build_field(shape, samples=None):
    """Flow field and search band from lane samples [(x, y, vx, vy, ...)]."""
    if samples is None:
        f = INSTRUMENT.lane_samples
        if not f.exists():
            raise FileNotFoundError(f"no Kreutz lane for {INSTRUMENT.label} yet: {f.name} is missing")
        samples = json.load(open(f))
    S = np.array([s[:4] for s in samples], dtype=float)
    H, W = shape
    # report (x, y, vx, vy) -> array (row, col, vrow, vcol)
    r, c = H - 1 - S[:, 1], S[:, 0]
    vr, vc = -S[:, 3], S[:, 2]
    r0, r1 = int(max(r.min() - BAND_PX, 0)), int(min(r.max() + BAND_PX + 1, H))
    c0, c1 = int(max(c.min() - BAND_PX, 0)), int(min(c.max() + BAND_PX + 1, W))
    hh, ww = r1 - r0, c1 - c0
    near = np.zeros((hh, ww), bool)
    near[np.clip(np.round(r - r0).astype(int), 0, hh - 1),
         np.clip(np.round(c - c0).astype(int), 0, ww - 1)] = True
    band = binary_dilation(near, iterations=BAND_PX)
    gr, gc = np.mgrid[r0:r1, c0:c1]
    tree = cKDTree(np.stack([r, c], 1))
    d, k = tree.query(np.stack([gr.ravel(), gc.ravel()], 1), k=min(K_NEAREST, len(S)))
    w = 1 / (d + 5.0)
    vrow = ((vr[k] * w).sum(1) / w.sum(1)).reshape(hh, ww).astype(np.float32)
    vcol = ((vc[k] * w).sum(1) / w.sum(1)).reshape(hh, ww).astype(np.float32)
    return Field(slice(r0, r1), slice(c0, c1), band, vrow, vcol, tree, vr, vc)


def _flow_at(field, rr, cc):
    """Flow (vrow, vcol) at array positions, nearest grid cell, clamped to
    the box (the lane's flow continues smoothly past its sampled ends)."""
    i = np.clip(np.round(rr - field.rows.start).astype(int), 0, field.vrow.shape[0] - 1)
    j = np.clip(np.round(cc - field.cols.start).astype(int), 0, field.vrow.shape[1] - 1)
    return field.vrow[i, j], field.vcol[i, j]


# For deciding a track is a Kreutz-lane object (not for searching): within
# MATCH_PX of a lane sample and moving within MATCH_DV of the samples'
# flow there. Looser than the search band because the lane shifts through
# the season and the samples cover Aug-Oct 2026. MATCH_PX is also how far
# from the lane a live candidate may be and still page: every real alert
# since 2026-10-08 was within 50 px of a sample and every October report
# on Sungrazer starts within 150 px, while 22 of the 28 false alerts were
# farther out (stray-light ghosts, lines and corner noise).
MATCH_PX = 150
MATCH_DV = 3.0


def lane_px(field, x, y, H):
    """How far a report position is from the nearest lane sample, in px."""
    return float(field.tree.query([H - 1 - y, x], k=1)[0])


def in_lane(field, x, y, vx, vy, H):
    """Is a track (report coords) in the Kreutz lane, moving with its
    flow? Used to decide that a Kreutz report from another instrument at
    the same time is the same object."""
    r, c = H - 1 - y, x
    d, k = field.tree.query([r, c], k=min(K_NEAREST, len(field.svr)))
    if d.min() > MATCH_PX:
        return False
    w = 1 / (d + 5.0)
    fvr, fvc = (field.svr[k] * w).sum() / w.sum(), (field.svc[k] * w).sum() / w.sum()
    return bool(np.hypot(fvc - vx, -fvr - vy) <= MATCH_DV)


# For vetting a measured motion: along the flow at that spot to within
# FLOW_ANGLE. Direction only, since a comet speeds up past the samples'
# flow as it nears the Sun (the 2026-10-08 comet was doing 15 px/frame at
# 1400 px where the flow says 11, which in_lane's velocity match rejects).
# This season's real tracks were within 12 deg of the flow; the stray-light
# features that fired on eclipse-season mornings were 54 deg off and more.
FLOW_ANGLE = 45.0


def flow_at(field, x, y, H):
    """The lane flow (vx, vy), report coords, at a report position: the
    distance-weighted mean of the nearest samples', anywhere in the frame."""
    d, k = field.tree.query([H - 1 - y, x], k=min(K_NEAREST, len(field.svr)))
    w = 1 / (d + 5.0)
    return float((field.svc[k] * w).sum() / w.sum()), float(-(field.svr[k] * w).sum() / w.sum())


def off_flow(field, x, y, vx, vy, H):
    """Why a motion (report coords) isn't the lane's at (x, y), or None."""
    fx, fy = flow_at(field, x, y, H)
    cos = (vx * fx + vy * fy) / (np.hypot(vx, vy) * np.hypot(fx, fy) + 1e-30)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    if ang > FLOW_ANGLE:
        return f"{ang:.0f} deg off the lane flow, ({vx:+.1f},{vy:+.1f}) where it runs ({fx:+.1f},{fy:+.1f})"
    return None


def lane_search(seq: Sequence, last: int, n: int, threshold: float, field: Field):
    """Candidates in the lane for the window of n frames ending at `last`."""
    idx, back = window_frames(seq, last, n)
    H, W = seq.shape
    # frames by how many steps before the last they are; None where missing
    frames = [None] * n
    for i, b in zip(idx, back):
        s = seq.significance(i)
        frames[b] = np.clip(np.nan_to_num(s, nan=0.0), -CLIP_SIGMA, CLIP_SIGMA)
    present = [k for k in range(n) if frames[k] is not None]
    br, bc = np.nonzero(field.band)
    pr0 = (br + field.rows.start).astype(np.float32)
    pc0 = (bc + field.cols.start).astype(np.float32)
    best = np.full(len(br), -np.inf, np.float32)
    best_dv = np.zeros((len(br), 2), np.float32)
    best_per = np.zeros((len(br), len(present)), np.float32)
    for dvr in DV:
        for dvc in DV:
            pr, pc = pr0.copy(), pc0.copy()
            acc = np.zeros(len(br), np.float32)
            per = np.zeros((len(br), len(present)), np.float32)
            col = 0
            for k in range(n):  # k steps before the last frame
                if frames[k] is not None:
                    ir, ic = np.round(pr).astype(int), np.round(pc).astype(int)
                    ok = (ir >= 0) & (ir < H) & (ic >= 0) & (ic < W)
                    v = np.zeros(len(br), np.float32)
                    v[ok] = frames[k][ir[ok], ic[ok]]
                    acc += v
                    per[:, col] = v
                    col += 1
                # one frame back: undo one frame of motion along the flow
                fr, fc = _flow_at(field, pr, pc)
                pr -= fr + dvr
                pc -= fc + dvc
            better = acc > best
            best[better] = acc[better]
            best_dv[better] = (dvr, dvc)
            best_per[better] = per[better]
    best /= np.sqrt(len(present))

    img = np.full(field.band.shape, -np.inf, np.float32)
    img[br, bc] = best
    peaks = (img >= threshold) & (img == maximum_filter(img, size=9))
    idx = {(a, b): i for i, (a, b) in enumerate(zip(br, bc))}
    out = []
    for a, b in zip(*np.nonzero(peaks)):
        i = idx[(a, b)]
        per = best_per[i][::-1]  # oldest frame first, like search.py
        if np.sum(per >= FRAME_KEEP) < np.ceil(FRAME_KEEP_FRAC * len(per)):
            continue
        r, c = a + field.rows.start, b + field.cols.start
        fr, fc = _flow_at(field, np.array([r]), np.array([c]))
        vrow, vcol = float(fr[0] + best_dv[i][0]), float(fc[0] + best_dv[i][1])
        x, y = seq.rc_to_report(r, c)
        out.append(Candidate(last, float(x), float(y), vcol, -vrow,
                             round(float(best[i]), 2), [round(float(v), 1) for v in per]))
    return _dedupe(sorted(out, key=lambda c: -c.score))
