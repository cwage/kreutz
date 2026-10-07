"""Score a run against Sungrazer reports: what we'd have found, and when.

    docker compose run --rm kreutz kreutz.evaluate /data/runs/n4/20250717T*.jsonl

Candidates from consecutive windows are linked into tracks: the same body
shows up one frame later where its velocity says it should be. A track
"fires" once it has been seen in LINK consecutive windows with every
window scoring at least the threshold, which is when we'd report it. With
4-frame windows, three linked windows span 6 frames (Sungrazer's minimum
is 5); on July 17 2025 that, with a 5.5 threshold, kept both confirmed
comets and cut unexplained tracks to ~13/day.

Each fired track is matched to reports (any report with points that day,
confirmed or not, so a real-but-unconfirmed comet isn't counted as a false
alarm), and for confirmed comets the lead is the reporter's posting time
minus when we could have posted: the track's last frame plus NOAA's
publication delay plus compute.
"""

import argparse
import json
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits

from . import DATA

FRAME = timedelta(minutes=15)
# Sun center in report coords; drifts a few px, which is fine for a radius
# cut on stored candidates (the search itself uses each frame's WCS).
SUN_XY = (1001, 951)
# NOAA publishes each real-time frame 12-19 min after it's taken (measured
# on the SWPC feed); one more minute covers our search.
PUBLISH_DELAY = timedelta(minutes=16)
LINK_PX = 3.0       # predicted vs found position, one frame on
LINK_DV = 1.0       # velocity change between windows, px/frame
MATCH_PX = 8.0      # track vs reported position at a reported time


def load(paths):
    """Candidates by window time, filtered as search() now filters them
    (so runs made before a filter was added still score consistently)."""
    from .search import EDGE_PX, R_MIN, SPEED_MIN, consistent
    by_time = defaultdict(list)
    seen = set()  # the same window searched twice (a day rerun in chunks) counts once
    for ln in (ln for p in paths for ln in open(p)):
        c = json.loads(ln)
        key = (c["time"], c["x"], c["y"], c["vx"], c["vy"])
        if key in seen:
            continue
        seen.add(key)
        if np.hypot(c["vx"], c["vy"]) < SPEED_MIN:
            continue
        if np.hypot(c["x"] - SUN_XY[0], c["y"] - SUN_XY[1]) < R_MIN:
            continue
        if not consistent(c["per_frame"]) or not EDGE_PX <= c["x"] <= 2047 - EDGE_PX \
                or not EDGE_PX <= c["y"] <= 1919 - EDGE_PX:
            continue
        c["t"] = datetime.fromisoformat(c["time"])
        by_time[c["t"]].append(c)
    return by_time


# A chain must actually travel: over its linked windows the found
# positions have to move at least this fraction of what the windows'
# velocity says (and a couple of pixels at least). Linking alone lets a
# stationary blip through when its claimed velocity is ~1 px/frame and the
# tolerance is 3 px (live alert 2026-10-06 08:30).
MOVE_FRAC = 0.5
MOVE_MIN_PX = 2.0


def moves(chain):
    a, b = chain[0], chain[-1]
    n = (b["t"] - a["t"]) / FRAME
    if n <= 0:
        return True
    moved = np.hypot(b["x"] - a["x"], b["y"] - a["y"])
    expect = np.hypot(b["vx"], b["vy"]) * n
    return moved >= max(MOVE_MIN_PX, MOVE_FRAC * expect)


# Constant false-alarm rate: a window's threshold is the base threshold or
# its CFAR_K-th highest candidate score, whichever is higher. A normal
# 8-frame window has ~4 candidates above 6 sigma; in eclipse season
# (Aug-Oct 2026) GOES-19 passes through Earth's shadow near 05 UTC, frames
# are disturbed by stray light from ~03:45 to 07:15, and the background and
# star models built from them stay contaminated for hours after, so
# windows at 07-08 UTC have ~50 candidates above 6 sigma: structured
# residue that stacks like a track. 80% of the archive's unexplained
# tracks fired there. With CFAR_K = 10 those windows need a much higher
# score, the rest are untouched, and all 47 known comets in the Aug-Oct
# 2026 archive are still found while unexplained tracks drop by 77%.
# July 2025 shows no such window; the rule only engages where it's needed.
CFAR_K = 10


def window_thresholds(by_time, threshold, k=CFAR_K):
    """{window time: threshold} per the CFAR rule."""
    out = {}
    for t, cands in by_time.items():
        s = sorted((c["score"] for c in cands), reverse=True)
        out[t] = max(threshold, s[k - 1]) if len(s) >= k else threshold
    return out


def tracks(by_time, threshold, link):
    """Chains of candidates in consecutive windows, each >= its window's
    threshold (window_thresholds)."""
    times = sorted(by_time)
    thr = window_thresholds(by_time, threshold)
    open_ = []  # chains still growing
    done = []
    for t in times:
        cands = [c for c in by_time[t] if c["score"] >= thr[t]]
        used = set()
        grown = []
        for ch in open_:
            last = ch[-1]
            if t - last["t"] != FRAME:
                done.append(ch)
                continue
            px, py = last["x"] + last["vx"], last["y"] + last["vy"]
            best, bd = None, LINK_PX
            for k, c in enumerate(cands):
                if k in used:
                    continue
                d = np.hypot(c["x"] - px, c["y"] - py)
                if d <= bd and np.hypot(c["vx"] - last["vx"], c["vy"] - last["vy"]) <= LINK_DV:
                    best, bd = k, d
            if best is None:
                done.append(ch)
            else:
                used.add(best)
                grown.append(ch + [cands[best]])
        open_ = grown + [[c] for k, c in enumerate(cands) if k not in used]
    done += open_
    return [ch for ch in done if len(ch) >= link and moves(ch[:link])]


# One reporter measures on SWPC's public CCOR-1 image rather than the L1b
# frame: that image is the frame derotated to solar north up (the header's
# PC matrix), scaled by SWPC_SCALE, with the Sun at SWPC_SUN. Fitted on
# all nine of his Aug-Sep 2026 Kreutz reports against our tracks of the
# same comets: 93 of 123 point pairs within 20 px (the rest are pairings
# with the wrong track), median residual 3 px. Converting needs the frame
# at the report time for its rotation; without it the report is skipped.
SWPC_CANVAS_REPORTERS = {"Bill Collins"}
SWPC_SCALE = 0.4726
SWPC_SUN = (604.9, 603.4)
_HDR = {}


def _header_at(t):
    from .frames import L1B
    key = f"{t:%Y%m%dT%H%M}"
    if key not in _HDR:
        fs = sorted((L1B / f"{t:%Y/%m/%d}").glob(f"CCOR1_1B_{key}*_NC.fits"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the files carry a BLANK on float data
            _HDR[key] = fits.getheader(fs[0], 1) if fs else None
    return _HDR[key]


def swpc_canvas_to_report(t, x, y):
    """A point measured on SWPC's derotated image -> report coordinates in
    the L1b frame at time t, or None without that frame."""
    h = _header_at(t)
    if h is None:
        return None
    west, north = (x - SWPC_SUN[0]) / SWPC_SCALE, -(y - SWPC_SUN[1]) / SWPC_SCALE
    pc = np.array([[h["PC1_1"], h["PC1_2"]], [h["PC2_1"], h["PC2_2"]]])
    dcol, drow = np.linalg.solve(pc, [west, north])
    col, row = h["CRPIX1"] - 1 + dcol, h["CRPIX2"] - 1 + drow
    return float(col), float(h["NAXIS2"] - 1 - row)


def report_points(reports, day):
    """(report, [(time, x, y)]) for reports with points on or around `day`,
    in L1b report coordinates (see swpc_canvas_to_report)."""
    out = []
    for r in reports:
        if not r["points"] or r["telescope"] != "CCOR1":
            continue
        posted = datetime.strptime(r["posted_utc"], "%Y-%m-%dT%H:%M:%SZ")
        pts = []
        for hhmm, x, y in r["points"]:
            t = posted.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]), second=0)
            if t > posted:
                t -= timedelta(days=1)
            if r.get("reporter") in SWPC_CANVAS_REPORTERS:
                xy = swpc_canvas_to_report(t.replace(second=25), x, y)
                if xy is None:
                    continue
                x, y = xy
            pts.append((t, x, y))
        if pts and any(abs((t.date() - day).days) <= 1 for t, _, _ in pts):
            out.append((r, pts, posted))
    return out


OTHER_BEFORE = timedelta(hours=24)
OTHER_AFTER = timedelta(hours=2)
OTHER_KINDS = ("Potential Comet", "Additional Observations", "Known Comet Now Entering")


def other_kreutz(reports, t_first, t_last):
    """Kreutz reports from instruments other than CCOR-1 whose observed
    times (or, without points, posting time) fall within OTHER_BEFORE of
    t_first to OTHER_AFTER of t_last. Their pixel coordinates can't be
    compared with ours, so this is matching by time; for a track in the
    Kreutz lane (lane.in_lane) that's nearly always the same object, since
    Kreutz comets arrive one at a time along one path."""
    t0, t1 = t_first - OTHER_BEFORE, t_last + OTHER_AFTER
    out = []
    for r in reports or []:
        if r.get("group") != "Kreutz" or r.get("telescope") in (None, "CCOR1") \
                or r.get("kind") not in OTHER_KINDS:
            continue
        posted = datetime.strptime(r["posted_utc"], "%Y-%m-%dT%H:%M:%SZ")
        times = []
        for hhmm, _, _ in r["points"]:
            ot = posted.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]), second=0)
            times.append(ot - timedelta(days=1) if ot > posted else ot)
        if not times:
            times = [posted]
        if any(t0 <= ot <= t1 for ot in times):
            out.append({"reporter": r["reporter"], "id": r["id"], "telescope": r["telescope"],
                        "posted_utc": r["posted_utc"], "kind": r["kind"]})
    return sorted(out, key=lambda o: o["posted_utc"])


def matches(track, pts):
    """Does the track pass within MATCH_PX of the report at a shared time?
    Positions are extrapolated along the track's velocity to report times
    within its span (+-2 frames)."""
    for t, x, y in pts:
        for c in track:
            k = (t - c["t"]) / FRAME
            if abs(k) <= 2:
                if np.hypot(c["x"] + c["vx"] * k - x, c["y"] + c["vy"] * k - y) <= MATCH_PX:
                    return True
    return False


def evaluate(runs, threshold, link):
    by_time = load(runs)
    day = min(by_time).date()
    reps = [json.loads(l) for p in (DATA / "reports").glob("*.jsonl") for l in open(p)]
    confirmed = {c["report"]["id"]: c["num"] for p in (DATA / "reports").glob("confirmed-*.json")
                 for c in json.load(open(p)) if c.get("report")}
    lo, hi = min(by_time), max(by_time)
    rp = [(r, pts, posted) for d in {t.date() for t in by_time}
          for r, pts, posted in report_points(reps, d)
          if any(lo <= t <= hi for t, _, _ in pts)]
    rp = list({r["id"]: (r, pts, posted) for r, pts, posted in rp}.values())
    trs = tracks(by_time, threshold, link)
    fired = []
    for tr in trs:
        fire = tr[link - 1]  # the window where it qualified
        hit = [(r, posted) for r, pts, posted in rp if matches(tr, pts)]
        fired.append((tr, fire, hit))
    return fired, rp, confirmed, by_time


_SEQ = {}


def _persist(track, link):
    """persist.persistence for a fired track, sharing one Sequence per day
    so significance maps are computed once."""
    from .frames import BG_FRAMES, BG_SKIP, STAR_FRAMES, Sequence, frame_paths
    from .persist import AHEAD, persistence
    fire = track[link - 1]
    day = fire["t"].date()
    if day not in _SEQ:
        _SEQ.clear()  # one day's maps at a time
        start = datetime.combine(day, datetime.min.time())
        warm = FRAME * (BG_FRAMES + BG_SKIP + STAR_FRAMES + 2)
        _SEQ[day] = Sequence(frame_paths(start - warm, start + timedelta(days=1, hours=3)))
    seq = _SEQ[day]
    last = seq.times.index(fire["t"])
    return persistence(seq, last, track[:link])[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--threshold", type=float, nargs="+", default=[5.0, 5.5, 6.0, 7.0])
    ap.add_argument("--link", type=int, default=3)
    ap.add_argument("--persist", action="store_true",
                    help="score each fired track's persistence in the frames after it (slow)")
    args = ap.parse_args()
    for th in args.threshold:
        fired, rp, confirmed, by_time = evaluate(args.runs, th, args.link)
        hours = len(by_time) * FRAME / timedelta(hours=1)
        real = [f for f in fired if f[2]]
        print(f"threshold {th}: {len(fired)} tracks fired over {hours:.0f} h, "
              f"{len(real)} on reported objects, {len(fired) - len(real)} unexplained")
        for r, pts, posted in rp:
            num = confirmed.get(r["id"])
            ours = [f for f in fired if any(h[0] is r for h in f[2])]
            label = f"{r['id']} ({r['reporter']}, {r['kind']}" + (f", confirmed {num})" if num else ")")
            if not ours:
                print(f"   missed  {label}, first point {pts[0][0]:%H:%M}")
                continue
            first = min(ours, key=lambda f: f[1]["t"])[1]
            we_post = first["t"] + PUBLISH_DELAY
            lead = (posted - we_post).total_seconds() / 60
            print(f"   found   {label}: our last frame {first['t']:%H:%M}, could post ~{we_post:%H:%M}; "
                  f"they posted {posted:%H:%M}Z -> lead {lead:+.0f} min; their first point {pts[0][0]:%H:%M}")
        for tr, fire, hit in fired:
            p = ""
            if args.persist:
                p = f" persistence {_persist(tr, args.link):+.1f}"
            if not hit:
                print(f"   unexplained: {fire['t']:%Y-%m-%d %H:%M} ({fire['x']:.0f},{fire['y']:.0f}) "
                      f"v=({fire['vx']:+.1f},{fire['vy']:+.1f}) scores {[c['score'] for c in tr]}{p}")
            elif args.persist:
                print(f"   REAL ({hit[0][0]['id']}): {fire['t']:%Y-%m-%d %H:%M}{p}")


if __name__ == "__main__":
    main()
