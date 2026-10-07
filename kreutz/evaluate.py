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
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

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


def tracks(by_time, threshold, link):
    """Chains of candidates in consecutive windows, each >= threshold."""
    times = sorted(by_time)
    open_ = []  # chains still growing
    done = []
    for t in times:
        cands = [c for c in by_time[t] if c["score"] >= threshold]
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


def report_points(reports, day):
    """(report, [(time, x, y)]) for reports with points on or around `day`."""
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
            pts.append((t, x, y))
        if any(abs((t.date() - day).days) <= 1 for t, _, _ in pts):
            out.append((r, pts, posted))
    return out


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
