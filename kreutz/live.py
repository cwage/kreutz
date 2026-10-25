"""Watch NOAA's real-time coronagraph feed and stop on the first candidate comet.

    docker compose run --rm kreutz kreutz.live [--threshold 5.5] [--link 3]

Polls the SWPC directory for new L1b frames, files them where the archive
fetcher does (data/ccor1/l1b/YYYY/MM/DD/), and searches each new window as
it lands. Candidates are linked across windows exactly as in evaluate.py;
when a track fires it's checked against Sungrazer's latest reports: one
somebody already reported is noted as such; a track isn't raised twice
(data/live/known.jsonl). Either way the track is written to
data/live/alerts/<time>.json with a crop animation beside it, and the
object is then measured in each new frame (report.py); once it has been
seen in five frames a second, "confirmed" file is written with the report
block ready to paste into Sungrazer's form, and it keeps being followed
until it's lost. A track that fires is vetted first (vet.py): followed
back through the raw frames, it's dropped as a star if it moves with the
star drift, as stationary if it doesn't move, as off-lane if its
motion runs more than lane.FLOW_ANGLE from the Kreutz flow there, as a
ghost if the same pixels held the same rise and fall at this time
yesterday (vet.same_as_yesterday: the eclipse-season stray light comes
back daily), and as a line if its brightest frame shows a streak rather
than a point (vet.line_like), or its own stack is a line (vet.line_stack:
CCOR-2's radial stray-light stripes move exactly like a sungrazer). A
follow-up is rejected rather than confirmed when its measured frames
don't lie on a line, don't move, or run against the lane; the
eclipse-season glare feature that was confirmed on 2026-10-11 fails all
three. Every outcome (alert, star, stationary, off-lane, already
reported, confirmed, rejected, faded, ended, ghost, line) is appended to
data/live/log.jsonl with the wall-clock time it was decided, which is
what score.py turns into a table of leads and lags against the reports.

data/live/state.json is rewritten after every window too: the windows
searched, the open chains and the follow-ups. A restart loads it and
carries on from the last window instead of re-searching the whole span,
and an object being followed keeps being measured and rendered. State
older than the span is ignored, so a long-stopped watcher starts clean.

data/live/status.json is rewritten after every window: open chains,
objects being followed with their stage (candidate at three windows,
confirmed at five frames) and whether and by whom they're already
reported, and the recent decisions. data/live/status.html reads it:

    python -m http.server -d data 8000   # then http://localhost:8000/live/status.html

Kreutz reports from other coronagraphs (CCOR-2, LASCO) can't be matched
by position. A track in the Kreutz lane (lane.in_lane) while another
instrument has a Kreutz comet in view counts as that comet, matched "by
time+lane"; the alert for anything else lists such reports as a hint.

--notify CMD runs CMD with the alert's .json path appended, e.g.
--notify "python -m kreutz.notify" to post to an ntfy topic (notify.py);
it's the command's job to be quiet about known comets, and about
candidates that don't move like a sungrazer: the alert file carries
"heading_deg", how far off straight at the Sun the track points, and
"paged", whether that's within PAGE_HEADING at the instrument's
page_speed or more (pageworthy). One that isn't is searched, followed,
pictured and listed like any other, just not paged until it's confirmed.
Every real alert so far was within 14 degrees of sunward at 4 px/frame or
more; the satellite of 2026-10-21 07:00 was 24 off, and the false alerts
scatter out to 90. (This replaced a gate on distance to the archived
Kreutz tracks, which the stream outgrows: the comet of 2026-10-21 11:00,
Ruan's, ran 273 px from the nearest track, and its first page was held
back for 15 minutes.) "lane_px", that distance, is still written for
the record.
--orbit-threshold T adds the orbit-constrained search (orbitsearch.py) to
every window, at threshold T: candidates along Kreutz orbits, which it
finds fainter and earlier than the full search and with fewer false
tracks (CCOR-1, 2026-10-01..24 replayed through these rules at 6.0: 19 of
23 reported comets, 0.2 unexplained pages a day; CCOR-2: 10 of 17, 0.4 a
day, against 1.2 for the full search). Its candidates join the full
search's, less any within orbitsearch.SEP_PX of one, and are linked,
vetted and followed with them. A track made only of orbit candidates is
"found_by": "orbit" in its files, and isn't paged until it's confirmed
unless --orbit-pages is given: a review tier while it proves itself live.
--render makes a confirmed object's cutouts and pictures (render.py,
style.py) in the background under data/render/<alert name>/, and again
every RENDER_EVERY frames while the object is still in view, so the
pictures grow with the passage.
--keep-days N deletes frames and significance maps more than N days old
(prune), for an always-on watcher on a small disk. The frames directory is
the archive fetcher's too, so don't use it where archive runs keep frames.
Alerts, logs and renders are kept; a pruned day comes back with kreutz.fetch.
"""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from . import DATA, page
from .instrument import CURRENT as INSTRUMENT
from .evaluate import (CFAR_K, FRAME, LINK_DV, LINK_PX, matches, moves, other_kreutz,
                       report_points)
from .lane import build_field, in_lane, lane_px, off_flow
from .reports import _fetch, parse, report_url
from .frames import BG_FRAMES, BG_SKIP, CADENCE, L1B, SIG_CACHE, STAR_FRAMES, Sequence, frame_paths
from .report import MIN_POINTS, block, fit, measure
from .search import MAX_MISSING, R_MIN, consistent, search
from .shortlist import FRAME_COLS, FRAME_ROWS, _known_by_position
from .orbitsearch import SAME_RMS, SEP_PX as ORBIT_SEP_PX, joint_rms, orbit_of, orbit_search
from .vet import STATIONARY, follow, line_like, line_stack, same_as_yesterday

FEED = INSTRUMENT.feed
POLL_S = 120
# Live alerts need a body that's clearly moving. The two false alerts of
# 2026-10-06 08:30 were blips at 1.1 and 1.5 px/frame; 10 of 172 CCOR-1
# comet reports (Jul-Aug 2025, Jul-Oct 2026) were slower than 2 when first
# measured, and those speed up as they near the Sun, so they're caught a
# little later rather than missed. The archive search keeps search.SPEED_MIN
# and leans on persist.py instead.
LIVE_SPEED_MIN = 2.0
LIVE = DATA / f"live{INSTRUMENT.tag}"
PRUNE_EVERY = timedelta(hours=1)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kreutz"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def sync(since):
    """Download feed frames taken after `since` that aren't on disk.
    (The feed directory keeps about three weeks of frames; only the recent
    hours matter here, and the archive fetcher is the way to get the rest.)
    Returns how many were new."""
    from .sync import listing, pull  # (sync imports this module)
    return pull(sorted(listing(since)))


def prune(keep_days, now, roots=None):
    """Delete the day directories (YYYY/MM/DD) of frames and significance
    maps, under every settings key of this instrument, whose whole day ended more than
    `keep_days` before `now`, and any month or year directories left empty.
    The date comes from the path, not file times (an archive-fetched frame
    is new on disk however old it is). Returns the directories removed."""
    if roots is None:
        # Cache keys are 'bg16...' for CCOR-1 and '<key>-bg16...' for the
        # others ('ccor2-', 'c2-'): each watcher prunes its own maps only.
        mine = (lambda p: p.name.startswith("bg")) if not INSTRUMENT.tag \
            else (lambda p: p.name.startswith(INSTRUMENT.key + "-"))
        roots = [L1B] + (sorted(p for p in SIG_CACHE.iterdir() if p.is_dir() and mine(p))
                         if SIG_CACHE.exists() else [])
    cutoff = now - timedelta(days=keep_days)
    gone = []
    for root in roots:
        for day in sorted(root.glob("[0-9]*/[0-9]*/[0-9]*")):
            try:
                d = datetime.strptime("/".join(day.parts[-3:]), "%Y/%m/%d")
            except ValueError:
                continue
            if not day.is_dir() or d + timedelta(days=1) > cutoff:
                continue
            shutil.rmtree(day)
            gone.append(day)
            for parent in (day.parent, day.parent.parent):
                try:
                    parent.rmdir()
                except OSError:  # not empty
                    break
    return gone


def known():
    p = LIVE / "known.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def is_known(c, t):
    """Near a vetted/known object's predicted position at time t."""
    for k in known():
        kt = datetime.fromisoformat(k["time"])
        n = (t - kt) / FRAME
        if abs(n) > 96:
            continue
        if np.hypot(k["x"] + k["vx"] * n - c["x"], k["y"] + k["vy"] * n - c["y"]) < 15:
            return True
    return False


def remember(c, why):
    with open(LIVE / "known.jsonl", "a") as fh:
        fh.write(json.dumps({k: c[k] for k in ("time", "x", "y", "vx", "vy")} | {"why": why}) + "\n")


def latest_reports():
    try:
        return parse(_fetch(0)) + parse(_fetch(1))
    except Exception:
        return None  # the site hiccups; treat as unknown and let a human check


_FIELD = {}


def lane_field(when):
    """The Kreutz lane's flow field, built once (once a day for an
    instrument whose lane moves with the seasons, lane.season_samples)."""
    key = when.date() if INSTRUMENT.season_days else None
    if key not in _FIELD:
        _FIELD.clear()
        _FIELD[key] = build_field((FRAME_ROWS, FRAME_COLS), when=when)  # lane.py: the instrument's samples
    return _FIELD[key]


def reported_by(track, t, reps):
    """The report from our instrument this track matches by position, if any, as
    {reporter, id, telescope, posted_utc, kind, how}. A match is the
    track passing through the report's points while they were measured,
    or along their path extrapolated as the archive's shortlist does:
    straight ahead for up to 12 h and back along the lane for up to 12 h,
    with a tolerance that grows with the distance from the measured span.
    So a report measured in the morning still claims the same comet's
    track in the evening. Of several that fit (a comet gets additional
    observations), the earliest posted: that's who found it."""
    track = [dict(c, t=datetime.fromisoformat(c["time"])) for c in track]
    candidates = report_points(reps, t.date())
    fits = by_orbit(track, candidates) if ORBIT_MATCH else None
    if fits is None:  # no orbit to test with: every report by position
        fits, rest = [], candidates
    else:  # the orbit decides for reports with a path; one-point reports by position
        fits, rest = [(r, "orbit") for r in fits], [c for c in candidates if len(c[1]) < 2]
    fits += [(r, "position") for r, pts, posted in rest
             if matches(track, pts) or (len(pts) >= 2
                                        and _known_by_position(track, [(r, pts)], lane_field(t)) is not None)]
    if not fits:
        return None
    r, how = min(fits, key=lambda f: f[0]["posted_utc"])
    return {k: r.get(k) for k in ("reporter", "id", "telescope", "posted_utc", "kind")} | {"how": how}


# A straight line from a report's points is a loose test. It can't reach
# a comet followed for a day, which speeds up as it nears the Sun with the
# report 15 h behind it (the Cerati/Ruan comet of 2026-10-24, alerted again
# at 01:30 the next morning as unreported); and extrapolated 12 h it
# claims the wrong comet on the same path (a CCOR-2 track of 2026-10-25
# 08:45 matched to Ruan's report of the evening before, 1100" off any one
# orbit with it, when it was Cerati's comet of that morning, 19"). Where
# frames carry a helioprojective WCS, one Kreutz orbit through a report's
# points and our track decides, however far apart in time.
ORBIT_MATCH = INSTRUMENT.key in ("ccor1", "ccor2")


def by_orbit(track, candidates):
    """The reports (with 2+ points) one family orbit fits together with
    the track, to SAME_RMS; None when the track itself has no such orbit
    (too few points on frames on disk, or not a Kreutz path) to test with."""
    own = [(c["t"], c["x"], c["y"]) for c in track]
    try:
        best = orbit_of(own)
        if best is None or best[1] > SAME_RMS:
            return None
        return [r for r, pts, posted in candidates if len(pts) >= 2 and joint_rms(best[0], own, pts) < SAME_RMS]
    except Exception as e:  # a missing frame or a failed fit must not stop the watcher
        print(f"orbit match: {e}", file=sys.stderr)
        return None


def recheck(pending, t, reps):
    """Follow-ups not yet settled on one of our instrument's reports, matched
    now against the latest reports: a report of the comet we're following
    usually arrives hours after our alert. Settled is a match by orbit
    where there are orbits (a match by position there is only provisional:
    it can be the wrong comet on the same path), else by position."""
    settled = ("orbit",) if ORBIT_MATCH else ("position", "orbit")
    for f in pending:
        old = f.get("known") or {}
        if reps and old.get("how") not in settled:
            k = reported_by(passage(f), t, reps)
            if k and (k["id"], k["how"]) != (old.get("id"), old.get("how")):
                f["known"] = k
                print(f"{t:%H:%M} {Path(f['alert']).stem} is {who(k)}, by {k['how']}", file=sys.stderr)


PASSAGE = timedelta(hours=INSTRUMENT.passage)  # a Kreutz comet stays in the field about this long after it's found


def matched_report(track, reps, held=False):
    """The report that fits a track, looked up after the fact (reports keep
    arriving for hours after a decision): one of our instrument's reports by position, or,
    for an object that held (confirmed), the earliest Kreutz report from
    another instrument observed between a day before the track and the end
    of its crossing, which is only 'likely' the same object (how: "time").
    A blip that faded gets no such guess."""
    if not reps or not track:
        return None
    t_last = datetime.fromisoformat(track[-1]["time"])
    found = reported_by(track, t_last, reps)
    if found or not held:
        return found
    others = other_kreutz(reps, datetime.fromisoformat(track[0]["time"]), t_last + PASSAGE)
    if others:
        return dict(min(others, key=lambda o: o["posted_utc"]), how="time")
    return None


def best_report(known, track, reps, held=False):
    """What an object is known as now: a match by position or orbit (certain), from
    when it was decided or against today's reports, else the guess by
    time it was filed under, else a fresh guess by time."""
    if known and known.get("how") in ("position", "orbit"):
        return known
    found = matched_report(track, reps, held=held) if reps else None
    if found and found["how"] in ("position", "orbit"):
        return found
    return known or found


def who(known):
    """'Jiangao Ruan report_jr_... (CCOR1)' for a reported_by record."""
    return f"{known['reporter']} {known['id']} ({known['telescope']})"


def other_instruments(track, reps):
    """Kreutz reports from other instruments around the track's time."""
    return other_kreutz(reps, datetime.fromisoformat(track[0]["time"]),
                        datetime.fromisoformat(track[-1]["time"]))


CONFIRM_LOST = 3  # consecutive frames without the object before a candidate is dropped
PAGE_HEADING = 20  # deg off straight at the Sun: a candidate pointed farther off than this isn't paged
CHAIN_PX = 8.0  # a search chain this close to a follow-up's prediction is the same object
FOLLOWED_PX = 15.0  # and so is a new chain this close to a follow-up's prediction (+ FOLLOWED_GROW/frame)
FOLLOWED_GROW = 1.0
CONFIRMED_LOST = 12  # the same for a confirmed object: it's real, so wait out a stray-light spell (3 h)
RENDER_EVERY = 4  # frames between progress renders of a confirmed object
RENDER_LOST = 8  # the render's follower tolerates this many misses (a stray-light hour)


def start_render(f):
    """kreutz.render and kreutz.style for a followed object, in the
    background, covering the frames seen so far; the Popen is kept on the
    entry so a slow render isn't started twice."""
    c = f["track"][-1]
    name = Path(f["alert"]).stem
    ahead = max(24, len(f["points"]) + RENDER_LOST)
    cmd = (f"python -m kreutz.render {c['time'][:16]} {c['x']} {c['y']} {c['vx']} {c['vy']} "
           f"--back 24 --ahead {ahead} --lost {RENDER_LOST} --name {name} "
           f"&& python -m kreutz.style /data/render{INSTRUMENT.tag}/{name} --all")
    f["render_proc"] = subprocess.Popen(["sh", "-c", cmd], cwd="/srv",
                                        stdout=open(LIVE / "alerts" / f"{name}.render.log", "w"),
                                        stderr=subprocess.STDOUT)
    f["since_render"] = 0


def rendering(f):
    return f.get("render_proc") is not None and f["render_proc"].poll() is None


def sunward(c, sun_xy):
    """(degrees off straight at the Sun, px/frame) of a candidate's motion."""
    sx, sy = sun_xy
    to_sun = np.degrees(np.arctan2(sy - c["y"], sx - c["x"]))
    heading = np.degrees(np.arctan2(c["vy"], c["vx"]))
    return float(abs((heading - to_sun + 180) % 360 - 180)), float(np.hypot(c["vx"], c["vy"]))


def cfar(cands, threshold, t, what=""):
    """Constant false-alarm rate (evaluate.window_thresholds): a window
    with many candidates above threshold is a disturbed one, and only its
    CFAR_K best and anything as good stand."""
    top = sorted((c["score"] for c in cands), reverse=True)
    thr = max(threshold, top[CFAR_K - 1]) if len(top) >= CFAR_K else threshold
    if thr > threshold:
        print(f"{t:%Y-%m-%d %H:%M}: {len(cands)} {what}candidates, threshold raised to {thr:.1f}",
              file=sys.stderr)
        cands = [c for c in cands if c["score"] >= thr]
    return cands


def pageworthy(c, sun_xy):
    """Does a candidate move like a sungrazer: at the Sun, within
    PAGE_HEADING, at the instrument's page_speed or more?"""
    off, speed = sunward(c, sun_xy)
    return off <= PAGE_HEADING and speed >= INSTRUMENT.page_speed


def passage(f):
    """A follow-up's track with every measured point after it: the whole
    path, for matching it to a report by orbit."""
    return f["track"] + [{"time": t.isoformat(), "x": x, "y": y, "vx": f["vx"], "vy": f["vy"]}
                         for t, x, y, _ in f["points"]]


def seen(f):
    """A follow-up's track extended to its latest measured point, so a log
    record's first/last and position cover the passage, not just the alert."""
    if not f["points"]:
        return f["track"]
    t, x, y, _ = f["points"][-1]
    return f["track"] + [{"time": t.isoformat(), "x": x, "y": y, "vx": f["vx"], "vy": f["vy"]}]


def chained(f, chains, t, dt):
    """The search's own sighting of a followed object in the window ending
    at t, as a measured point (t, x, y, score), if a chain's newest
    candidate sits where the follow-up predicts it and moves with it. The
    search stacks frames, so it holds a comet too faint to stand out in
    any single frame; the follow-up's single-frame measurement used to
    lose those within an hour while the search went on chaining them for
    the rest of the day."""
    px, py = f["x"] + f["vx"] * dt, f["y"] + f["vy"] * dt
    for ch in chains:
        c = ch[-1]
        if datetime.fromisoformat(c["time"]) == t and np.hypot(c["x"] - px, c["y"] - py) <= CHAIN_PX \
                and np.hypot(c["vx"] - f["vx"], c["vy"] - f["vy"]) <= 2 * LINK_DV:
            return (t, float(c["x"]), float(c["y"]), float(c["score"]))
    return None


REANCHOR_PX = 25.0  # a confirmed object's follow-up takes over a search chain this close to its prediction
REANCHOR_DV = 2.0   # px/frame, and moving this much like it
REANCHOR_FIT = 3    # measured points needed to predict from their fit rather than the latest step
FIT_RECENT = 12     # the fit uses this many latest points: a comet speeds up, and a day-long line lags it


def predicted(f, t, dt):
    """Where a followed object should be at t and how fast it moves, (x, y,
    vx, vy): from a straight line through its latest FIT_RECENT measured
    points when it has REANCHOR_FIT of them, else its latest step. The
    latest step is what the follow-up normally predicts from, and one bad
    point throws it: the frame of 2026-10-20 07:00, taken off-pointed,
    turned Stark's comet's 9 px/frame into 6, and the follow-up lost a
    40-sigma comet that was where the line said it would be."""
    pts = (f.get("points") or [])[-FIT_RECENT:]
    if len(pts) < REANCHOR_FIT:
        return f["x"] + f["vx"] * dt, f["y"] + f["vy"] * dt, f["vx"], f["vy"]
    vx, vy, _ = fit(pts)
    k = (t - pts[0][0]) / FRAME
    ks = np.array([(p[0] - pts[0][0]) / FRAME for p in pts])
    return (float(np.mean([p[1] for p in pts]) + vx * (k - ks.mean())),
            float(np.mean([p[2] for p in pts]) + vy * (k - ks.mean())), vx, vy)


def reanchored(f, chains, t, dt):
    """For a confirmed object the single-frame measurement and chained()
    both missed: the newest candidate of a search chain within REANCHOR_PX
    of where the object should be and REANCHOR_DV of its motion, as a
    measured point, with its velocity. Where and how fast it should be come
    from a straight line through all the follow-up's measured points, not
    from its latest frame-to-frame step: that step is what drifts in the
    eclipse-season glare (6 px/frame where the comet of 2026-10-15 was
    doing 9.5), so a prediction from it falls behind, and the follow-up
    ended that comet at 07:16 while the search held it all day."""
    px, py, vx, vy = predicted(f, t, dt)
    best = None
    for ch in chains:
        c = ch[-1]
        d = np.hypot(c["x"] - px, c["y"] - py)
        if datetime.fromisoformat(c["time"]) == t and d <= REANCHOR_PX \
                and np.hypot(c["vx"] - vx, c["vy"] - vy) <= REANCHOR_DV and (best is None or d < best[0]):
            best = (d, c)
    if best is None:
        return None
    c = best[1]
    return (t, float(c["x"]), float(c["y"]), float(c["score"])), (float(c["vx"]), float(c["vy"]))


EDGE_NEAR = 60  # px: an object lost this close to the frame's edge or the search's inner edge left the field


def where_lost(x, y, shape, sun):
    """'edge' when a follow-up's last position (report coords) is within
    EDGE_NEAR of the frame's edge or of the search annulus' inner radius
    around the Sun (row, col), else 'lost': the follow-up lost it in view."""
    H, W = shape
    sx, sy = sun[1], H - 1 - sun[0]
    if min(x, y, W - 1 - x, H - 1 - y) <= EDGE_NEAR or np.hypot(x - sx, y - sy) <= R_MIN + EDGE_NEAR:
        return "edge"
    return "lost"


def followed(c, t, pending):
    """Is candidate c at time t one of the objects already being followed?
    The follow-up knows where its comet is now, which the memory of where
    it was alerted doesn't: Kreutz comets speed up, so a straight line
    from the alert falls behind by the hour, and the search was re-alerting
    the comet it was following."""
    for f in pending:
        dt = (t - f["t"]) / FRAME
        if 0 <= dt <= CONFIRMED_LOST and np.hypot(f["x"] + f["vx"] * dt - c["x"],
                                                  f["y"] + f["vy"] * dt - c["y"]) <= FOLLOWED_PX + FOLLOWED_GROW * dt:
            return True
    return False


# A confirmation's measured frames have to lie on a line. The largest
# residual allowed grows with the arc, since a comet accelerates: 0.5 px
# over 5 frames and 1.2-1.5 px over 14 for this season's real comets,
# 3.1 px over 6 for the glare feature of 2026-10-11.
RESIDUAL_PER_FRAME = 0.3


def unlikely(points, field=None, H=FRAME_ROWS):
    """Why measured frames aren't a comet, or None: they don't lie on a
    line, the fitted motion is stationary, or it runs against the lane."""
    vx, vy, resid = fit(points)
    if resid > RESIDUAL_PER_FRAME * len(points):
        return f"straight-line residual {resid:.1f} px over {len(points)} frames"
    if np.hypot(vx, vy) < STATIONARY:
        return f"stationary ({np.hypot(vx, vy):.1f} px/frame over {len(points)} frames)"
    if field is not None:
        _, x, y, _ = points[-1]
        return off_flow(field, x, y, vx, vy, H)
    return None


def follow_up(seq, last, pending, notify, render=False, chains=(), field=None):
    """Keep measuring alerted objects in each new frame. Once one has been
    seen in MIN_POINTS frames it's 'confirmed': the alert's .confirmed.json
    gets the report block and the notify command runs on it. It's then
    still followed, re-rendered every RENDER_EVERY frames if --render, until
    it has been missing for CONFIRMED_LOST frames in a row (logged as
    ended); a candidate missing CONFIRM_LOST frames before confirmation is
    logged as faded. A frame where the single-frame measurement fails but
    the search's chains still hold the object (chained, or for a confirmed
    object reanchored) holds it: the follow-up moves to where the chain
    says and the misses don't grow, but nothing is measured, so the frame
    counts toward neither confirmation nor the report block. (It used to
    count: the search stacks eight frames, so a chain goes on holding an
    object for two hours after it's gone, and the satellite glint of
    2026-10-21 05:30-06:30 was "confirmed" at 07:31 on a fifth point no
    frame showed.) An object is confirmed on a frame it was measured in.
    At the confirming frame the measured points are vetted (unlikely):
    what fails is logged as rejected and dropped, with no notification
    and no render."""
    keep = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for f in pending:
        dt = (seq.times[last] - f["t"]) / FRAME
        if round(dt) < 1:
            keep.append(f)
            continue
        pts = measure(seq, last, f["x"] + f["vx"] * dt, f["y"] + f["vy"] * dt, f["vx"], f["vy"],
                      back=0, ahead=0, lost=0)
        held = chained(f, chains, seq.times[last], dt) if not pts else None
        velocity = None
        if not pts and not held and f.get("confirmed"):
            # where the line through its recent points says, rather than
            # where its latest step says; then a search chain near there
            px, py, vx, vy = predicted(f, seq.times[last], dt)
            pts = measure(seq, last, px, py, vx, vy, back=0, ahead=0, lost=0)
            if pts:
                velocity = (vx, vy)
            else:
                hit = reanchored(f, chains, seq.times[last], dt)
                if hit:
                    held, velocity = hit
        if held and not pts:
            # the search still has it: follow its position, measure nothing
            t, x, y, _ = held
            f["vx"], f["vy"] = velocity or ((x - f["x"]) / dt, (y - f["y"]) / dt)
            f["t"], f["x"], f["y"] = t, x, y
            keep.append(f)
            continue
        if not pts:
            f["misses"] += 1
            if f["misses"] < (CONFIRMED_LOST if f.get("confirmed") else CONFIRM_LOST):
                keep.append(f)
            elif f.get("confirmed"):
                log("ended", seen(f), now, alert=f["alert"], frames=len(f["points"]), known=f.get("known"),
                    where=where_lost(f["x"], f["y"], seq.shape, seq.sun_rc()))
                if render and not rendering(f):
                    start_render(f)  # the whole passage
            else:
                log("faded", seen(f), now, alert=f["alert"])
            continue
        f["misses"] = 0
        f["points"].append(pts[-1])
        f["since_render"] = f.get("since_render", 0) + 1
        t, x, y, _ = pts[-1]
        f["vx"], f["vy"] = velocity or ((x - f["x"]) / dt, (y - f["y"]) / dt)
        f["t"], f["x"], f["y"] = t, x, y
        if not f.get("confirmed") and len(f["points"]) >= MIN_POINTS:
            why = unlikely(f["points"], field, seq.shape[0] if field is not None else FRAME_ROWS)
            if why:
                remember(seen(f)[-1], f"auto: {why}")
                log("rejected", seen(f), now, alert=f["alert"], frames=len(f["points"]), note=why)
                print(f"REJECTED {f['alert']} ({len(f['points'])} frames, {why})", flush=True)
                continue
            f["confirmed"] = True
            f["confirmed_utc"] = now.isoformat(timespec="seconds")
            lines, note = block(f["points"])
            path = f["alert"].replace(".json", ".confirmed.json")
            json.dump({"kind": "confirmed", "track": f["track"], "report": lines, "note": note,
                       "frames": len(f["points"]), "other_instruments": f["others"],
                       "known": f.get("known"), "decided_utc": now.isoformat(timespec="seconds"),
                       "lane_px": f.get("lane_px"), "heading_deg": f.get("heading_deg"), "paged": f.get("paged"),
                       "found_by": f.get("found_by")},
                      open(path, "w"), indent=1)
            log("confirmed", seen(f), now, alert=f["alert"], frames=len(f["points"]), note=note,
                known=f.get("known"))
            print(f"CONFIRMED {path} ({len(f['points'])} frames, {note})", flush=True)
            if notify:
                try:
                    subprocess.Popen(shlex.split(notify) + [path])
                except OSError as e:
                    print(f"notify failed: {e}", file=sys.stderr)
            if render:
                start_render(f)
        elif f.get("confirmed") and render and f["since_render"] >= RENDER_EVERY and not rendering(f):
            start_render(f)
        keep.append(f)
    return keep


def log(kind, track, decided, **extra):
    rec = {"kind": kind, "decided_utc": decided.isoformat(timespec="seconds"),
           "first": track[0]["time"], "last": track[-1]["time"],
           "x": track[-1]["x"], "y": track[-1]["y"], "vx": track[-1]["vx"], "vy": track[-1]["vy"]}
    rec.update(extra)
    with open(LIVE / "log.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def tile(img, r, c, half=40):
    """The (2*half+1)-square crop of img centred on (r, c), NaN where it
    runs off the frame. A plain slice with a negative start wraps around
    and comes back empty, which failed the animation of every alert within
    40 px of an edge (2026-10-15 01:30, (1544,1867))."""
    out = np.full((2 * half + 1, 2 * half + 1), np.nan, np.float32)
    r0, r1 = max(r - half, 0), min(r + half + 1, img.shape[0])
    c0, c1 = max(c - half, 0), min(c + half + 1, img.shape[1])
    if r0 < r1 and c0 < c1:
        out[r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = img[r0:r1, c0:c1]
    return out


def animation(seq, track, path):
    """Crops of the raw and significance frames along the track, as a GIF."""
    last = track[-1]
    i_last = last["frame"]
    frames = []
    for i in range(i_last - 7, i_last + 1):
        k = i - i_last
        x, y = last["x"] + last["vx"] * k, last["y"] + last["vy"] * k
        r, c = seq.report_to_rc(x, y)
        r, c = int(round(r)), int(round(c))
        tiles = []
        for img, lo, hi in ((seq.raw(i), 1, 99.5), (seq.significance(i), 1, 99.7)):
            crop = tile(img, r, c)
            v = np.nanpercentile(crop, [lo, hi]) if np.isfinite(crop).any() else np.zeros(2)
            t = np.clip((np.nan_to_num(crop, nan=v[0]) - v[0]) / (v[1] - v[0] + 1e-30) * 255, 0, 255)
            t = np.flipud(t.astype(np.uint8))  # as displayed: north up
            t[40, 34:38] = t[40, 44:48] = 255  # tick marks either side of the prediction
            tiles.append(t)
        frames.append(Image.fromarray(np.kron(np.hstack(tiles), np.ones((3, 3), np.uint8))))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=400, loop=0)


STATE = LIVE / "state.json"


def save_state(done, chains, pending, span, path=STATE):
    """The watcher's working state, for a restart: the windows searched
    within `span` of the newest, the open chains, and the follow-ups (a
    render's process handle is dropped)."""
    if not done:
        return
    newest = max(done)
    d = {"done": sorted(t.isoformat() for t in done if newest - t <= span),
         "chains": chains,
         "pending": [{k: v for k, v in f.items() if k != "render_proc"}
                     | {"t": f["t"].isoformat(),
                        "points": [[t.isoformat(), float(x), float(y), float(s)] for t, x, y, s in f["points"]]}
                     for f in pending]}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(path)


def load_state(now, span, path=STATE):
    """(done, chains, pending) as saved, or empty when there's no state or
    its newest window is more than `span` before `now`."""
    if not path.exists():
        return set(), [], []
    d = json.loads(path.read_text())
    done = {datetime.fromisoformat(t) for t in d["done"]}
    if not done or now - max(done) > span:
        return set(), [], []
    pending = [f | {"t": datetime.fromisoformat(f["t"]),
                    "points": [(datetime.fromisoformat(t), x, y, s) for t, x, y, s in f["points"]]}
               for f in d["pending"]]
    for f in pending:  # saved before the follow-up carried these: the alert files have them
        files = alert_files(Path(f["alert"]).stem)
        f.setdefault("decided_utc", (files["alert"] or {}).get("decided_utc"))
        f.setdefault("confirmed_utc", (files["confirmed"] or {}).get("decided_utc"))
    return done, d["chains"], pending


def alert_files(name):
    """An alert's .json and .confirmed.json as dicts (None where missing);
    an old alert file that's a bare track list is wrapped."""
    out = {}
    for key, suffix in (("alert", ".json"), ("confirmed", ".confirmed.json")):
        p = LIVE / "alerts" / f"{name}{suffix}"
        if not p.exists():
            out[key] = None
            continue
        d = json.load(open(p))
        out[key] = d if isinstance(d, dict) else {"track": d}
    return out


STATUS_RECENT = 20  # decisions shown on the status page
STATUS_REPORTS_EVERY = timedelta(hours=1)  # how often the page's after-the-fact matches refresh


def pictures(name, root=None):
    """A render's pictures for the status page, as paths relative to
    the live directory: the plain preview and each style preset's crop, field and
    still, whichever exist; None without a render."""
    d = (root or DATA / f"render{INSTRUMENT.tag}") / name
    if not (d / "preview.gif").exists():
        return None
    out = {"preview": f"../render{INSTRUMENT.tag}/{name}/preview.gif", "styles": {}}
    files = [d / "preview.gif"]
    for sd in sorted(d.glob("style-*")):
        have = {k: f"../render{INSTRUMENT.tag}/{name}/{sd.name}/{f}" for k, f in
                (("crop", "crop.gif"), ("field", "field.gif"), ("still", "still.png")) if (sd / f).exists()}
        if have:
            out["styles"][sd.name[len("style-"):]] = have
            files += [sd / f for f in ("crop.gif", "field.gif", "still.png") if (sd / f).exists()]
    # the newest of them, so the page's cache-buster changes whenever any
    # picture is remade (kreutz.style can rerun without touching the preview)
    made = datetime.fromtimestamp(max(f.stat().st_mtime for f in files), timezone.utc).replace(tzinfo=None)
    out["made_utc"] = made.isoformat(timespec="seconds")
    return out


def status(chains, pending, t, recent, now, rendered=lambda name: None,
           files_of=lambda name: {"alert": None, "confirmed": None}, reps=None):
    """The status page's data: what's open, what's being followed and at
    which stage, and the last decisions. Paths are relative to the live
    directory; "measure_prefix" + an object's "alert" is its kreutz.report command.
    Each object's "report" is the Sungrazer report that fits it as of now
    (best_report): a match by position, from when it was decided or found
    now in reps, else the guess by time it was filed under.
    rendered(name) gives a render's pictures() or None; files_of(name) an
    alert's files (alert_files), which carry the full track and the times
    the object was found and confirmed, whatever the log window shows."""
    longest = max(chains, key=len) if chains else None

    by_id = {r["id"]: r for r in reps or []}

    def with_url(known, report=None):
        """The known record with the report's page; older log entries have
        only the 'Reporter report_xx_... (CCOR1)' string, so pull the id out
        and fill the rest in from the fetched reports when it's there."""
        if not known and report:
            m = re.match(r"(.*?)\s*(report_[a-z]+_\d{14})\s*(?:\((\w+)\))?", report)
            known = {"reporter": m.group(1), "id": m.group(2), "telescope": m.group(3)} if m else None
        if not known or not known.get("id"):
            return known
        r = by_id.get(known["id"])
        if r and not known.get("posted_utc"):
            known = {k: r.get(k) for k in ("reporter", "id", "telescope", "posted_utc", "kind")} \
                | {k: v for k, v in known.items() if v is not None}
        return dict(known, url=report_url(known["id"]))

    following = []
    for f in pending:
        name = Path(f["alert"]).stem
        following.append({
            "name": name, "alert": f"alerts/{name}.json", "gif": f"alerts/{name}.gif",
            "stage": "confirmed" if f.get("confirmed") else "candidate",
            "points": len(f["points"]), "misses": f["misses"],
            "first": f["track"][0]["time"], "last_seen": f["t"].isoformat(timespec="seconds"),
            "x": f["x"], "y": f["y"], "vx": f["vx"], "vy": f["vy"],
            "known": with_url(f.get("known")), "others": len(f.get("others") or []),
            "lane_px": f.get("lane_px"), "heading_deg": f.get("heading_deg"), "paged": f.get("paged"),
            "found_by": f.get("found_by"),
            "near_lane": f.get("near_lane"),  # files from before the heading gate
            "decided_utc": f.get("decided_utc"), "confirmed_utc": f.get("confirmed_utc"),
            "report": with_url(best_report(f.get("known"), seen(f), reps, held=bool(f.get("confirmed")))),
            "pictures": rendered(name), "rendering": rendering(f)})
    decisions = []
    for r in recent:
        name = Path(r["alert"]).stem if r.get("alert") else None
        files = files_of(name) if name else {"alert": None, "confirmed": None}
        track = (files["alert"] or {}).get("track")
        known = with_url(r.get("known"), r.get("report"))
        decisions.append({
            "kind": r["kind"], "decided_utc": r["decided_utc"], "first": r["first"], "last": r["last"],
            "x": r["x"], "y": r["y"], "name": name, "gif": f"alerts/{name}.gif" if name else None,
            "known": known, "how": r.get("how"), "frames": r.get("frames"), "note": r.get("note"),
            "where": r.get("where"),
            "found_utc": (files["alert"] or {}).get("decided_utc") or (r["decided_utc"] if r["kind"] in ("alert", "reported") else None),
            "confirmed_utc": (files["confirmed"] or {}).get("decided_utc"),
            "report": with_url(best_report(known, track, reps, held=r["kind"] in ("confirmed", "ended"))),
            "pictures": rendered(name) if name else None})
    return {"instrument": INSTRUMENT.label, "measure_prefix": INSTRUMENT.measure_cmd(""),
            "updated_utc": now.isoformat(timespec="seconds"), "window_utc": t.isoformat(timespec="seconds"),
            "open": len(chains), "longest": len(longest) if longest else 0,
            "longest_at": [longest[-1]["x"], longest[-1]["y"]] if longest else None,
            "following": following, "recent": decisions}


def write_status(chains, pending, t, reps=None):
    p = LIVE / "log.jsonl"
    recent = [json.loads(ln) for ln in open(p).readlines()[-STATUS_RECENT:]] if p.exists() else []
    recent.reverse()

    s = status(chains, pending, t, recent, datetime.now(timezone.utc).replace(tzinfo=None),
               rendered=pictures, files_of=alert_files, reps=reps)
    tmp = LIVE / "status.json.tmp"
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(LIVE / "status.json")


def open_tracks(chains):
    """The open-tracks part of the per-window log line: how many chains
    are open and, if any, how long the longest is and where it was last
    seen, so one object persisting reads differently from blips."""
    if not chains:
        return "0 open tracks"
    longest = max(chains, key=len)
    return (f"{len(chains)} open tracks (longest {len(longest)} windows, "
            f"at {longest[-1]['x']:.0f},{longest[-1]['y']:.0f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=5.5)
    ap.add_argument("--link", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--notify", help="command run with the alert .json path as argument")
    ap.add_argument("--orbit-threshold", type=float, metavar="T",
                    help="also run the orbit-constrained search (orbitsearch.py) at threshold T")
    ap.add_argument("--orbit-pages", action="store_true",
                    help="page tracks found only by the orbit search at the candidate stage too")
    ap.add_argument("--render", action="store_true",
                    help="on confirmation, run kreutz.render and kreutz.style for it in the background")
    ap.add_argument("--keep-days", type=int, metavar="N",
                    help="delete frames and significance maps more than N days old (N >= 1); "
                         "the frames directory is kreutz.fetch's too, so this deletes archive frames")
    args = ap.parse_args()
    # The watcher works on the last ~11 h of frames and sync() re-downloads
    # anything missing from that span, so a day is the floor.
    if args.keep_days is not None and args.keep_days < 1:
        ap.error("--keep-days must be at least 1")
    (LIVE / "alerts").mkdir(parents=True, exist_ok=True)
    (LIVE / "log.jsonl").touch()  # so a tail -f has something to follow before anything fires
    page.write()  # the one status page over every instrument, data/index.html
    history = CADENCE * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.n + args.link)
    # done: windows searched; chains: open tracks, as in evaluate.tracks;
    # pending: alerted objects still being followed. Resumed from the last
    # save when it's recent enough to matter.
    done, chains, pending = load_state(datetime.now(timezone.utc).replace(tzinfo=None), history + timedelta(hours=2))
    if done:
        print(f"resumed: {len(done)} windows searched through {max(done):%Y-%m-%d %H:%M}, "
              f"{len(chains)} open tracks, {len(pending)} follow-ups", file=sys.stderr)

    def checkpoint(t):
        try:
            save_state(done, chains, pending, history + timedelta(hours=2))
        except Exception as e:  # disk trouble must not take the watcher down
            print(f"{t:%H:%M} state error: {e}", file=sys.stderr)
    field = None  # the Kreutz lane, built once a frame shape is known
    pruned = None  # when prune last ran
    reps, reps_at = None, None  # Sungrazer's latest reports, for the status page's matches
    while True:
        try:
            new = sync(datetime.now(timezone.utc).replace(tzinfo=None) - history - timedelta(hours=1))
        except Exception as e:  # the feed hiccups; just try again
            print(f"{datetime.now(timezone.utc).replace(tzinfo=None):%H:%M} feed error: {e}", file=sys.stderr)
            time.sleep(POLL_S)
            continue
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        seq = Sequence(frame_paths(now - history - timedelta(hours=2), now))
        sun_xy = seq.rc_to_report(*seq.sun_rc())  # report coords, for pageworthy
        if field is None or INSTRUMENT.season_days:
            field = lane_field(now) if seq.shape == (FRAME_ROWS, FRAME_COLS) else build_field(seq.shape, when=now)
        todo = [i for i, t in enumerate(seq.times)
                if t not in done and i >= BG_FRAMES + BG_SKIP + args.n]
        for last in todo:
            t = seq.times[last]
            try:
                cands = [asdict(c) for c in search(seq, last, args.n, args.threshold)
                         if consistent(c.per_frame) and np.hypot(c.vx, c.vy) >= LIVE_SPEED_MIN]
            except ValueError as e:
                print(f"{t:%Y-%m-%d %H:%M}: skipped ({e})", file=sys.stderr)
                done.add(t)
                checkpoint(t)
                continue
            cands = cfar(cands, args.threshold, t)
            if args.orbit_threshold is not None:
                try:
                    extra = [dict(asdict(c), found_by="orbit")
                             for c in orbit_search(seq, last, args.n, args.orbit_threshold)
                             if consistent(c.per_frame) and np.hypot(c.vx, c.vy) >= LIVE_SPEED_MIN]
                except ValueError:
                    extra = []
                extra = cfar(extra, args.orbit_threshold, t, "orbit ")
                cands += [c for c in extra
                          if all(np.hypot(c["x"] - o["x"], c["y"] - o["y"]) > ORBIT_SEP_PX for o in cands)]
            for c in cands:
                c["time"] = t.isoformat()
            # link onto open chains
            grown, used = [], set()
            for ch in chains:
                prev = ch[-1]
                steps = (t - datetime.fromisoformat(prev["time"])) / FRAME
                if not 1 <= round(steps) <= 1 + MAX_MISSING:
                    continue
                px, py = prev["x"] + prev["vx"] * steps, prev["y"] + prev["vy"] * steps
                for k, c in enumerate(cands):
                    if k not in used and np.hypot(c["x"] - px, c["y"] - py) <= LINK_PX * steps \
                            and np.hypot(c["vx"] - prev["vx"], c["vy"] - prev["vy"]) <= LINK_DV:
                        used.add(k)
                        grown.append(ch + [c])
                        break
            chains = grown + [[c] for k, c in enumerate(cands) if k not in used]
            done.add(t)
            print(f"{t:%Y-%m-%d %H:%M}: {len(cands)} candidates, {open_tracks(chains)}", file=sys.stderr)
            # The latest reports first, so a confirmation this window is
            # filed and announced under the report that fits it, not as
            # unreported because the report arrived since the alert.
            try:
                if reps_at is None or datetime.now(timezone.utc).replace(tzinfo=None) - reps_at >= STATUS_REPORTS_EVERY:
                    reps = latest_reports() or reps
                    reps_at = datetime.now(timezone.utc).replace(tzinfo=None)
                recheck(pending, t, reps)
            except Exception as e:
                print(f"{t:%H:%M} reports error: {e}", file=sys.stderr)
            try:
                pending = follow_up(seq, last, pending, args.notify, args.render, chains, field)
            except Exception as e:  # a follow-up must never take the watcher down
                print(f"{t:%H:%M} follow-up error: {e}", file=sys.stderr)
            for ch in chains:
                if len(ch) == args.link and not is_known(ch[-1], t) and not followed(ch[-1], t, pending) \
                        and moves([dict(c, t=datetime.fromisoformat(c["time"])) for c in ch]):
                    c = ch[-1]
                    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                    m = follow(seq, last, c["x"], c["y"], c["vx"], c["vy"])
                    if m.is_star:
                        remember(c, f"auto: moves with the stars ({m.vx:+.2f},{m.vy:+.2f}) "
                                    f"vs drift ({m.star_vx:+.2f},{m.star_vy:+.2f})")
                        log("star", ch, now_utc)
                        print(f"{t:%H:%M} track at ({c['x']:.0f},{c['y']:.0f}) is a star", file=sys.stderr)
                        continue
                    if m.is_stationary:
                        why = f"stationary ({np.hypot(m.vx, m.vy):.2f} px/frame over {len(m.points)} frames)"
                        remember(c, f"auto: {why}")
                        log("stationary", ch, now_utc, note=why)
                        print(f"{t:%H:%M} track at ({c['x']:.0f},{c['y']:.0f}) is {why}", file=sys.stderr)
                        continue
                    # the raw-frame motion where there was enough to fit it, else the search's
                    why = off_flow(field, c["x"], c["y"], m.vx, m.vy, seq.shape[0])
                    if why:
                        remember(c, f"auto: {why}")
                        log("off-lane", ch, now_utc, note=why)
                        print(f"{t:%H:%M} track at ({c['x']:.0f},{c['y']:.0f}) is {why}", file=sys.stderr)
                        continue
                    # what the significance maps say: the same thing was here
                    # at this time yesterday (a stray-light ghost), or it's a
                    # line rather than a point, in its brightest frame or in
                    # its own stack (CCOR-2's radial striping)
                    vetoed = False
                    for kind, check in (("ghost", same_as_yesterday), ("line", line_like), ("line", line_stack)):
                        try:
                            why = check(seq, last, c["x"], c["y"], c["vx"], c["vy"])
                        except Exception as e:  # a check that can't run mustn't hold up an alert
                            print(f"{t:%H:%M} {kind} check failed: {e}", file=sys.stderr)
                            why = None
                        if why:
                            remember(c, f"auto: {why}")
                            log(kind, ch, now_utc, note=why)
                            print(f"{t:%H:%M} track at ({c['x']:.0f},{c['y']:.0f}) is {why}", file=sys.stderr)
                            vetoed = True
                            break
                    if vetoed:
                        continue
                    reps = latest_reports() or reps
                    reps_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    known_rep = reported_by(ch, t, reps) if reps else None
                    others = other_instruments(ch, reps)
                    if not known_rep and others and in_lane(field, c["x"], c["y"], c["vx"], c["vy"], seq.shape[0]):
                        # A lane track while another coronagraph has a Kreutz
                        # comet in view: the same object.
                        known_rep = dict(others[0], how="time+lane")
                    # Known or not, the track gets an alert file, a vetting
                    # animation and a follow-up, so a reported comet is
                    # measured and pictured like any other; only the log
                    # kind and the notification differ.
                    stem = LIVE / "alerts" / f"{t:%Y%m%dT%H%M}_{c['x']:.0f}_{c['y']:.0f}"
                    px = round(lane_px(field, c["x"], c["y"], seq.shape[0]))
                    off, speed = sunward(c, sun_xy)
                    paged = pageworthy(c, sun_xy)
                    found_by = "orbit" if all(k.get("found_by") == "orbit" for k in ch) else None
                    held = found_by and paged and not args.orbit_pages
                    if held:
                        paged = False
                    stem.with_suffix(".json").write_text(json.dumps(
                        {"track": ch, "decided_utc": now_utc.isoformat(timespec="seconds"),
                         "reports_checked": reps is not None, "other_instruments": others,
                         "known": known_rep, "lane_px": px, "heading_deg": round(off), "paged": paged,
                         "found_by": found_by},
                        indent=1))
                    try:
                        animation(seq, ch, stem.with_suffix(".gif"))
                    except Exception as e:
                        print(f"{t:%H:%M} animation failed: {e}", file=sys.stderr)
                    if known_rep:
                        remember(c, f"reported: {who(known_rep)}")
                        log("reported", ch, now_utc, report=who(known_rep), how=known_rep["how"],
                            known=known_rep, others=others, alert=str(stem.with_suffix(".json")))
                        print(f"{t:%H:%M} track at ({c['x']:.0f},{c['y']:.0f}) "
                              f"already reported ({who(known_rep)}, by {known_rep['how']})", file=sys.stderr)
                    else:
                        remember(c, f"alert: {stem.name}")
                        quiet = None if paged else "found by the orbit search only, not paged" if held \
                            else f"{off:.0f} deg off sunward at {speed:.1f} px/frame, not paged"
                        log("alert", ch, now_utc, alert=str(stem.with_suffix(".json")),
                            reports_checked=reps is not None, others=others, lane_px=px,
                            heading_deg=round(off), paged=paged, note=quiet)
                        hint = f"; {len(others)} Kreutz report(s) from other instruments in the last day" \
                            if others else ""
                        print(f"ALERT {stem}{' (' + quiet + ')' if quiet else ''}{hint}", flush=True)
                    # follow it for a confirmed notification; start from what's
                    # measurable already
                    pts = measure(seq, last, c["x"], c["y"], c["vx"], c["vy"], back=12, ahead=0)
                    pending.append({"alert": str(stem.with_suffix(".json")), "track": ch, "others": others,
                                    "known": known_rep, "points": pts, "t": t, "x": c["x"], "y": c["y"],
                                    "vx": c["vx"], "vy": c["vy"], "misses": 0, "lane_px": px,
                                    "heading_deg": round(off), "paged": paged, "found_by": found_by,
                                    "decided_utc": now_utc.isoformat(timespec="seconds")})
                    if args.notify:
                        try:
                            subprocess.Popen(shlex.split(args.notify) + [str(stem.with_suffix(".json"))])
                        except OSError as e:
                            print(f"notify failed: {e}", file=sys.stderr)
            try:
                write_status(chains, pending, t, reps)
            except Exception as e:
                print(f"{t:%H:%M} status error: {e}", file=sys.stderr)
            checkpoint(t)
        if args.keep_days and (pruned is None or now - pruned >= PRUNE_EVERY):
            pruned = now
            try:
                gone = prune(args.keep_days, now)
                if gone:
                    print(f"{now:%H:%M} pruned {len(gone)} day directories older than "
                          f"{args.keep_days} d", file=sys.stderr)
            except Exception as e:  # disk trouble must not take the watcher down
                print(f"{now:%H:%M} prune error: {e}", file=sys.stderr)
        if new == 0:
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
