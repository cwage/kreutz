"""Watch NOAA's real-time CCOR-1 feed and stop on the first candidate comet.

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
until it's lost. Every outcome (alert, star, already reported, confirmed,
faded) is appended to data/live/log.jsonl with the wall-clock time it was
decided, which is what score.py turns into a table of leads and lags
against the reports.

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
it's the command's job to be quiet about known comets.
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

from . import DATA
from .evaluate import (CFAR_K, FRAME, LINK_DV, LINK_PX, matches, moves, other_kreutz,
                       report_points)
from .lane import build_field, in_lane
from .reports import _fetch, parse, report_url
from .frames import BG_FRAMES, BG_SKIP, L1B, SIG_CACHE, STAR_FRAMES, Sequence, frame_paths
from .report import MIN_POINTS, block, measure
from .search import MAX_MISSING, consistent, search
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


def prune(keep_days, now, roots=None):
    """Delete the day directories (YYYY/MM/DD) of frames and significance
    maps, under every settings key, whose whole day ended more than
    `keep_days` before `now`, and any month or year directories left empty.
    The date comes from the path, not file times (an archive-fetched frame
    is new on disk however old it is). Returns the directories removed."""
    if roots is None:
        roots = [L1B] + (sorted(p for p in SIG_CACHE.iterdir() if p.is_dir())
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


def reported_by(track, t, reps):
    """The CCOR-1 report this track matches by position, if any, as
    {reporter, id, telescope, posted_utc, kind, how}."""
    track = [dict(c, t=datetime.fromisoformat(c["time"])) for c in track]
    for r, pts, posted in report_points(reps, t.date()):
        if matches(track, pts):
            return {k: r.get(k) for k in ("reporter", "id", "telescope", "posted_utc", "kind")} \
                | {"how": "position"}
    return None


PASSAGE = timedelta(hours=12)  # a Kreutz comet stays in CCOR-1's field about this long after it's found


def matched_report(track, reps, held=False):
    """The report that fits a track, looked up after the fact (reports keep
    arriving for hours after a decision): a CCOR-1 report by position, or,
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


def who(known):
    """'Jiangao Ruan report_jr_... (CCOR1)' for a reported_by record."""
    return f"{known['reporter']} {known['id']} ({known['telescope']})"


def other_instruments(track, reps):
    """Kreutz reports from other instruments around the track's time."""
    return other_kreutz(reps, datetime.fromisoformat(track[0]["time"]),
                        datetime.fromisoformat(track[-1]["time"]))


CONFIRM_LOST = 3  # consecutive frames without the object before a candidate is dropped
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
           f"&& python -m kreutz.style /data/render/{name} --all")
    f["render_proc"] = subprocess.Popen(["sh", "-c", cmd], cwd="/srv",
                                        stdout=open(LIVE / "alerts" / f"{name}.render.log", "w"),
                                        stderr=subprocess.STDOUT)
    f["since_render"] = 0


def rendering(f):
    return f.get("render_proc") is not None and f["render_proc"].poll() is None


def seen(f):
    """A follow-up's track extended to its latest measured point, so a log
    record's first/last and position cover the passage, not just the alert."""
    if not f["points"]:
        return f["track"]
    t, x, y, _ = f["points"][-1]
    return f["track"] + [{"time": t.isoformat(), "x": x, "y": y, "vx": f["vx"], "vy": f["vy"]}]


def follow_up(seq, last, pending, notify, render=False):
    """Keep measuring alerted objects in each new frame. Once one has been
    seen in MIN_POINTS frames it's 'confirmed': the alert's .confirmed.json
    gets the report block and the notify command runs on it. It's then
    still followed, re-rendered every RENDER_EVERY frames if --render, until
    it has been missing for CONFIRMED_LOST frames in a row (logged as
    ended); a candidate missing CONFIRM_LOST frames before confirmation is
    logged as faded."""
    keep = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for f in pending:
        dt = (seq.times[last] - f["t"]) / FRAME
        if dt < 1:
            keep.append(f)
            continue
        pts = measure(seq, last, f["x"] + f["vx"] * dt, f["y"] + f["vy"] * dt, f["vx"], f["vy"],
                      back=0, ahead=0, lost=0)
        if not pts:
            f["misses"] += 1
            if f["misses"] < (CONFIRMED_LOST if f.get("confirmed") else CONFIRM_LOST):
                keep.append(f)
            elif f.get("confirmed"):
                log("ended", seen(f), now, alert=f["alert"], frames=len(f["points"]))
                if render and not rendering(f):
                    start_render(f)  # the whole passage
            else:
                log("faded", seen(f), now, alert=f["alert"])
            continue
        f["misses"] = 0
        f["points"].append(pts[-1])
        f["since_render"] = f.get("since_render", 0) + 1
        t, x, y, _ = pts[-1]
        f["vx"], f["vy"] = (x - f["x"]) / dt, (y - f["y"]) / dt
        f["t"], f["x"], f["y"] = t, x, y
        if not f.get("confirmed") and len(f["points"]) >= MIN_POINTS:
            f["confirmed"] = True
            f["confirmed_utc"] = now.isoformat(timespec="seconds")
            lines, note = block(f["points"])
            path = f["alert"].replace(".json", ".confirmed.json")
            json.dump({"kind": "confirmed", "track": f["track"], "report": lines, "note": note,
                       "frames": len(f["points"]), "other_instruments": f["others"],
                       "known": f.get("known"), "decided_utc": now.isoformat(timespec="seconds")},
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
    data/live/: the plain preview and each style preset's crop, field and
    still, whichever exist; None without a render."""
    d = (root or DATA / "render") / name
    if not (d / "preview.gif").exists():
        return None
    out = {"preview": f"../render/{name}/preview.gif", "styles": {}}
    files = [d / "preview.gif"]
    for sd in sorted(d.glob("style-*")):
        have = {k: f"../render/{name}/{sd.name}/{f}" for k, f in
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
    which stage, and the last decisions. Paths are relative to data/live/.
    Each object's "report" is the Sungrazer report that fits it as of now:
    what it was known as when decided, else matched_report() against reps.
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
            "decided_utc": f.get("decided_utc"), "confirmed_utc": f.get("confirmed_utc"),
            "report": with_url(f.get("known") or matched_report(f["track"], reps, held=bool(f.get("confirmed")))),
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
            "found_utc": (files["alert"] or {}).get("decided_utc") or (r["decided_utc"] if r["kind"] in ("alert", "reported") else None),
            "confirmed_utc": (files["confirmed"] or {}).get("decided_utc"),
            "report": known or with_url(matched_report(track, reps, held=r["kind"] in ("confirmed", "ended"))),
            "pictures": rendered(name) if name else None})
    return {"updated_utc": now.isoformat(timespec="seconds"), "window_utc": t.isoformat(timespec="seconds"),
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
    shutil.copy(Path(__file__).with_name("status.html"), LIVE / "status.html")
    history = timedelta(minutes=15) * (BG_FRAMES + BG_SKIP + STAR_FRAMES + args.n + args.link)
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
                checkpoint(t)
                continue
            for c in cands:
                c["time"] = t.isoformat()
            # Constant false-alarm rate (evaluate.window_thresholds): a window
            # with many candidates above threshold is a disturbed one.
            top = sorted((c["score"] for c in cands), reverse=True)
            thr = max(args.threshold, top[CFAR_K - 1]) if len(top) >= CFAR_K else args.threshold
            if thr > args.threshold:
                print(f"{t:%Y-%m-%d %H:%M}: {len(cands)} candidates, threshold raised to {thr:.1f}",
                      file=sys.stderr)
                cands = [c for c in cands if c["score"] >= thr]
            # link onto open chains
            grown, used = [], set()
            for ch in chains:
                prev = ch[-1]
                steps = (t - datetime.fromisoformat(prev["time"])) / FRAME
                if not 1 <= steps <= 1 + MAX_MISSING:
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
            try:
                pending = follow_up(seq, last, pending, args.notify, args.render)
            except Exception as e:  # a follow-up must never take the watcher down
                print(f"{t:%H:%M} follow-up error: {e}", file=sys.stderr)
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
                    stem.with_suffix(".json").write_text(json.dumps(
                        {"track": ch, "decided_utc": now_utc.isoformat(timespec="seconds"),
                         "reports_checked": reps is not None, "other_instruments": others,
                         "known": known_rep}, indent=1))
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
                        log("alert", ch, now_utc, alert=str(stem.with_suffix(".json")),
                            reports_checked=reps is not None, others=others)
                        hint = f"; {len(others)} Kreutz report(s) from other instruments in the last day" \
                            if others else ""
                        print(f"ALERT {stem}{hint}", flush=True)
                    # follow it for a confirmed notification; start from what's
                    # measurable already
                    pts = measure(seq, last, c["x"], c["y"], c["vx"], c["vy"], back=12, ahead=0)
                    pending.append({"alert": str(stem.with_suffix(".json")), "track": ch, "others": others,
                                    "known": known_rep, "points": pts, "t": t, "x": c["x"], "y": c["y"],
                                    "vx": c["vx"], "vy": c["vy"], "misses": 0,
                                    "decided_utc": now_utc.isoformat(timespec="seconds")})
                    if args.notify:
                        try:
                            subprocess.Popen(shlex.split(args.notify) + [str(stem.with_suffix(".json"))])
                        except OSError as e:
                            print(f"notify failed: {e}", file=sys.stderr)
            try:
                if reps_at is None or datetime.now(timezone.utc).replace(tzinfo=None) - reps_at >= STATUS_REPORTS_EVERY:
                    reps = latest_reports() or reps
                    reps_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
