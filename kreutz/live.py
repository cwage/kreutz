"""Watch NOAA's real-time CCOR-1 feed and stop on the first candidate comet.

    docker compose run --rm kreutz kreutz.live [--threshold 5.5] [--link 3]

Polls the SWPC directory for new L1b frames, files them where the archive
fetcher does (data/ccor1/l1b/YYYY/MM/DD/), and searches each new window as
it lands. Candidates are linked across windows exactly as in evaluate.py;
when a track fires it's checked against Sungrazer's latest reports: one
somebody already reported from CCOR-1 is logged to data/live/known.jsonl
and the watch goes on; anything else (and not already vetted into
known.jsonl) is written to data/live/alerts/<time>.json with a crop
animation beside it, remembered so it isn't raised again, and the watch
goes on. Every outcome (alert, star, already reported) is appended to
data/live/log.jsonl with the wall-clock time it was decided, which is
what score.py turns into a table of leads and lags against the reports.

Kreutz reports from other coronagraphs (CCOR-2, LASCO) can't be matched
by position. A track in the Kreutz lane (lane.in_lane) while another
instrument has a Kreutz comet in view counts as that comet, matched "by
time+lane"; the alert for anything else lists such reports as a hint.

--notify CMD runs CMD with the alert's .json path as its argument.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from . import DATA
from .evaluate import FRAME, LINK_DV, LINK_PX, matches, moves, other_kreutz, report_points
from .lane import build_field, in_lane
from .reports import _fetch, parse
from .frames import BG_FRAMES, BG_SKIP, L1B, STAR_FRAMES, Sequence, frame_paths
from .search import consistent, search
from .vet import follow

FEED = "https://services.swpc.noaa.gov/products/ccor1/fits/"
POLL_S = 120
# Live alerts need a body that's clearly moving. The two false alerts of
# 2026-10-06 08:30 were blips at 1.1 and 1.5 px/frame; 10 of 172 CCOR-1
# comet reports (Jul-Aug 2025, Jul-Oct 2026) were slower than 2 when first
# measured, and those speed up as they near the Sun, so they're caught a
# little later rather than missed. The archive search keeps search.SPEED_MIN
# and leans on persist.py instead.
LIVE_SPEED_MIN = 2.0
LIVE = DATA / "live"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kreutz"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def sync(since):
    """Download feed frames taken after `since` that aren't on disk.
    (The feed directory keeps about three weeks of frames; only the recent
    hours matter here, and the archive fetcher is the way to get the rest.)
    Returns how many were new."""
    names = sorted(set(re.findall(r'href="(CCOR1_1B_\d{8}T\d{6}_V\d+_NC\.fits)"',
                                  _get(FEED).decode())))
    new = 0
    for n in names:
        t = datetime.strptime(n[9:24], "%Y%m%dT%H%M%S")
        if t < since:
            continue
        dest = L1B / f"{t:%Y/%m/%d}" / n
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(_get(FEED + n))
        tmp.rename(dest)
        new += 1
    return new


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


def reported_by(track, t, reps):
    """The CCOR-1 report this track matches by position, if any."""
    track = [dict(c, t=datetime.fromisoformat(c["time"])) for c in track]
    for r, pts, posted in report_points(reps, t.date()):
        if matches(track, pts):
            return f"{r['reporter']} {r['id']}"
    return None


def other_instruments(track, reps):
    """Kreutz reports from other instruments around the track's time."""
    return other_kreutz(reps, datetime.fromisoformat(track[0]["time"]),
                        datetime.fromisoformat(track[-1]["time"]))


def log(kind, track, decided, **extra):
    rec = {"kind": kind, "decided_utc": decided.isoformat(timespec="seconds"),
           "first": track[0]["time"], "last": track[-1]["time"],
           "x": track[-1]["x"], "y": track[-1]["y"], "vx": track[-1]["vx"], "vy": track[-1]["vy"]}
    rec.update(extra)
    with open(LIVE / "log.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


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
            crop = img[r - 40:r + 41, c - 40:c + 41]
            v = np.nanpercentile(crop, [lo, hi])
            t = np.clip((np.nan_to_num(crop, nan=v[0]) - v[0]) / (v[1] - v[0] + 1e-30) * 255, 0, 255)
            t = np.flipud(t.astype(np.uint8))  # as displayed: north up
            t[40, 34:38] = t[40, 44:48] = 255  # tick marks either side of the prediction
            tiles.append(t)
        frames.append(Image.fromarray(np.kron(np.hstack(tiles), np.ones((3, 3), np.uint8))))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=400, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=5.5)
    ap.add_argument("--link", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--notify", help="command run with the alert .json path as argument")
    args = ap.parse_args()
    (LIVE / "alerts").mkdir(parents=True, exist_ok=True)
    history = timedelta(minutes=15) * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.n + args.link)
    done = set()
    chains = []  # open tracks, as in evaluate.tracks
    field = None  # the Kreutz lane, built once a frame shape is known
    while True:
        try:
            new = sync(datetime.now(timezone.utc).replace(tzinfo=None) - history - timedelta(hours=1))
        except Exception as e:  # the feed hiccups; just try again
            print(f"{datetime.now(timezone.utc).replace(tzinfo=None):%H:%M} feed error: {e}", file=sys.stderr)
            time.sleep(POLL_S)
            continue
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        seq = Sequence(frame_paths(now - history - timedelta(hours=2), now))
        if field is None:
            field = build_field(seq.shape)
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
                continue
            for c in cands:
                c["time"] = t.isoformat()
            # link onto open chains
            grown, used = [], set()
            for ch in chains:
                prev = ch[-1]
                if t - datetime.fromisoformat(prev["time"]) != FRAME:
                    continue
                px, py = prev["x"] + prev["vx"], prev["y"] + prev["vy"]
                for k, c in enumerate(cands):
                    if k not in used and np.hypot(c["x"] - px, c["y"] - py) <= LINK_PX \
                            and np.hypot(c["vx"] - prev["vx"], c["vy"] - prev["vy"]) <= LINK_DV:
                        used.add(k)
                        grown.append(ch + [c])
                        break
            chains = grown + [[c] for k, c in enumerate(cands) if k not in used]
            done.add(t)
            print(f"{t:%Y-%m-%d %H:%M}: {len(cands)} candidates, {len(chains)} open tracks",
                  file=sys.stderr)
            for ch in chains:
                if len(ch) == args.link and not is_known(ch[-1], t) \
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
                    reps = latest_reports()
                    who = reported_by(ch, t, reps) if reps else None
                    others = other_instruments(ch, reps)
                    how = "position"
                    if not who and others and in_lane(field, c["x"], c["y"], c["vx"], c["vy"], seq.shape[0]):
                        # A lane track while another coronagraph has a Kreutz
                        # comet in view: the same object.
                        o = others[0]
                        who, how = f"{o['reporter']} {o['id']} ({o['telescope']})", "time+lane"
                    if who:
                        remember(c, f"reported: {who}")
                        log("reported", ch, now_utc, report=who, how=how, others=others)
                        print(f"{t:%H:%M} track at ({c['x']:.0f},{c['y']:.0f}) "
                              f"already reported ({who}, by {how})", file=sys.stderr)
                        continue
                    stem = LIVE / "alerts" / f"{t:%Y%m%dT%H%M}_{c['x']:.0f}_{c['y']:.0f}"
                    stem.with_suffix(".json").write_text(json.dumps(
                        {"track": ch, "decided_utc": now_utc.isoformat(timespec="seconds"),
                         "reports_checked": reps is not None, "other_instruments": others}, indent=1))
                    try:
                        animation(seq, ch, stem.with_suffix(".gif"))
                    except Exception as e:
                        print(f"{t:%H:%M} animation failed: {e}", file=sys.stderr)
                    remember(c, f"alert: {stem.name}")
                    log("alert", ch, now_utc, alert=str(stem.with_suffix(".json")),
                        reports_checked=reps is not None, others=others)
                    hint = f"; {len(others)} Kreutz report(s) from other instruments in the last day" \
                        if others else ""
                    print(f"ALERT {stem}{hint}", flush=True)
                    if args.notify:
                        try:
                            subprocess.Popen([args.notify, str(stem.with_suffix(".json"))])
                        except OSError as e:
                            print(f"notify failed: {e}", file=sys.stderr)
        if new == 0:
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
