"""Is a track real? Stack the significance maps along the path through it
and compare with the same stack along offset copies of the path: a comet
adds up frame after frame, noise doesn't, and sits inside the offsets'
spread. The watcher's candidate/confirmed stages say whether the follow-up
could measure the object in single frames, which is about its brightness
per frame; this says whether there is anything there at all, and it was
what settled every live candidate by hand (the C3 comet of 2026-11-08:
z 38.9 over 73 frames; the C2 maybe of 11-05: z 5.5).

    check(track, end) -> Stack along the straight line through `track`'s
    points (dicts with "time", "x", "y"), over the frames from the first
    point's stacking window to `end`.

A Stack is cumulative and serialisable (to_dict/from_dict), so the live
follow-up carries one per candidate in its state and adds each new frame
at the position it predicts for it; z_at(t) gives the z the stack had
reached by any frame. scripts/orbit-stack.py does the same by hand, along
a Kreutz orbit or a line; scripts/stack-calibrate.py ran check() over
every alert since 2026-10-23 to place REAL_Z.
"""
import warnings
from datetime import datetime, timedelta

import numpy as np

from .frames import BG_FRAMES, BG_SKIP, CADENCE, Sequence, frame_paths
from .instrument import CURRENT as INSTRUMENT

NULL = 200              # offset copies of the path
NULL_R = (20.0, 120.0)  # px from the path, random directions
R = 2                   # px around the predicted position searched for the peak

# The stack's z at or past this reads "real": a page says REPORT. From
# scripts/stack-calibrate.py over the alerts of 2026-10-23..11-08
# (data/calib): along the alert's own line over its own frames, no false
# CCOR alert that passed the page gates stacked past 6.3 while real ones
# ran 3 to 50 (the bright ones page here, the faint ones by persistence,
# live.HELD_WINDOWS); C2's false alerts stacked to 13.6 (stars and
# non-Kreutz movers its gates let through), its real ones 27 and 51.
REAL_Z_BY = {"ccor1": 8.0, "ccor2": 8.0, "c2": 15.0, "c3": 15.0}
REAL_Z = REAL_Z_BY.get(INSTRUMENT.key, 8.0)


def offsets(n=NULL, null_r=NULL_R, seed=1):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    dist = rng.uniform(*null_r, n)
    return np.stack([dist * np.cos(ang), dist * np.sin(ang)], axis=1)


def peak(sig, x, y, r=R):
    """The largest significance within r px of report position (x, y) on
    map `sig`, or NaN off the map."""
    H, W = sig.shape
    r0, c0 = int(round(H - 1 - y)), int(round(x))
    if not (r <= r0 < H - r and r <= c0 < W - r):
        return np.nan
    win = sig[r0 - r:r0 + r + 1, c0 - r:c0 + r + 1]
    if not np.isfinite(win).any():  # off the detector (occulter, corners): no warning, no value
        return np.nan
    return float(np.nanmax(win))


class Stack:
    """The stack along a path and along NULL offsets of it, frame by frame."""

    def __init__(self, offs=None, r=R):
        self.offs = offsets() if offs is None else np.asarray(offs, dtype=float)
        self.r = r
        self.times, self.values, self.null = [], [], []  # per frame: t, peak there, peaks at the offsets

    def add_frame(self, sig, x, y, t):
        """Add map `sig` (frame at t) with the path at (x, y). Returns False
        when the position is off the map (nothing added)."""
        v = peak(sig, x, y, self.r)
        if not np.isfinite(v):
            return False
        self.times.append(t)
        self.values.append(v)
        self.null.append([0.0 if not np.isfinite(u) else u
                          for u in (peak(sig, x + dx, y + dy, self.r) for dx, dy in self.offs)])
        return True

    @property
    def n(self):
        return len(self.values)

    def _at(self, k):
        """(stack, offsets' mean, offsets' std, beaten) over the first k frames."""
        v = np.asarray(self.values[:k])
        nl = np.asarray(self.null[:k])
        s = v.sum() / np.sqrt(k)
        ns = nl.sum(axis=0) / np.sqrt(k)
        return float(s), float(ns.mean()), float(ns.std()), int((ns >= s).sum())

    def _k(self, t):
        return self.n if t is None else sum(1 for u in self.times if u <= t)

    def z_at(self, t=None):
        """z of the stack over the frames up to t (all of them if None):
        (stack - offsets' mean) / offsets' std, or None without frames."""
        k = self._k(t)
        if k < 1:
            return None
        s, m, sd, _ = self._at(k)
        return float((s - m) / sd) if sd > 0 else None

    def summary(self, t=None):
        """{"n", "stack", "null_mean", "null_std", "z", "beaten"} up to t, or None."""
        k = self._k(t)
        if k < 1:
            return None
        s, m, sd, beaten = self._at(k)
        return {"n": k, "stack": round(s, 1), "null_mean": round(m, 1), "null_std": round(sd, 1),
                "z": round((s - m) / sd, 1) if sd > 0 else None, "beaten": beaten}

    def to_dict(self):
        return {"offs": [[round(float(a), 2), round(float(b), 2)] for a, b in self.offs], "r": self.r,
                "times": [t.isoformat() if isinstance(t, datetime) else t for t in self.times],
                "values": [round(float(v), 3) for v in self.values],
                "null": [[round(float(u), 3) for u in row] for row in self.null]}

    @classmethod
    def from_dict(cls, d):
        s = cls(d["offs"], d.get("r", R))
        s.times = [datetime.fromisoformat(t) if isinstance(t, str) else t for t in d["times"]]
        s.values = list(d["values"])
        s.null = [list(row) for row in d["null"]]
        return s


def line_through(track):
    """where(t) -> (x, y) on the straight line fitted to the track's
    points (dicts with "time", "x", "y"; two or more, or one with a
    velocity in px/frame)."""
    ts = [datetime.fromisoformat(c["time"]) if isinstance(c["time"], str) else c["time"] for c in track]
    t0 = ts[0]
    h = np.array([(t - t0).total_seconds() / 3600 for t in ts])
    if len(set(h)) < 2:
        per_h = 3600 / CADENCE.total_seconds()
        px = (track[-1].get("vx", 0.0) * per_h, track[-1]["x"])
        py = (track[-1].get("vy", 0.0) * per_h, track[-1]["y"])
    else:
        px = np.polyfit(h, [c["x"] for c in track], 1)
        py = np.polyfit(h, [c["y"] for c in track], 1)

    def where(t):
        k = (t - t0).total_seconds() / 3600
        return float(np.polyval(px, k)), float(np.polyval(py, k))
    return where


def stack(seq, where, start, end, r=R, offs=None):
    """A Stack of seq's significance maps along where(t) for the frames
    from start to end."""
    out = Stack(offs, r)
    for i, t in enumerate(seq.times):
        if t < start or t > end:
            continue
        x, y = where(t)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sig = seq.significance(i)
        except Exception:
            continue
        out.add_frame(sig, x, y, t)
    return out


def span(track, n_stacked=None):
    """(start, end) of a track's own frames: from the first point's
    stacking window (n_stacked frames ending at it; the track's per_frame
    length by default) to the last point."""
    ts = [datetime.fromisoformat(c["time"]) if isinstance(c["time"], str) else c["time"] for c in track]
    n = n_stacked or len(track[-1].get("per_frame") or ()) or 1
    return ts[0] - CADENCE * (n - 1), ts[-1]


def check(track, end=None, n_stacked=None, **kw):
    """The Stack along the line through `track` over its own frames
    (span()) and on to `end`, loading the frames from the archive; the
    frames before the span load for the background only."""
    start, last = span(track, n_stacked)
    end = end or last
    paths = frame_paths(start - CADENCE * (BG_FRAMES + BG_SKIP + 1), end + timedelta(seconds=1))
    if not paths:
        return Stack()
    return stack(Sequence(paths), line_through(track), start, end, **kw)


def verdict(z):
    """How a stack's z reads: "real" at or past REAL_Z, "weak" below, None
    without one."""
    if z is None:
        return None
    return "real" if z >= REAL_Z else "weak"
