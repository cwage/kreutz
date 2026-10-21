"""Audit a running watcher against Sungrazer's reports: what it should have
alerted on and didn't, and what it alerted on that came to nothing.

    docker compose run --rm kreutz kreutz.audit
    KREUTZ_INSTRUMENT=ccor2 docker compose run --rm kreutz kreutz.audit
    docker compose run --rm kreutz kreutz.audit --since 2026-10-08   # a one-off look back

kreutz.score starts from the alerts; this starts from the reports. Every
report from the instrument with two or more points inside the span the
watcher has searched is matched against the watcher's decisions
(log.jsonl), its follow-ups and its open chains, by position
(shortlist._known_by_position: the report extrapolated along its path and,
for Kreutz reports, back along the lane), or by the watcher's own match
to the report. A verdict line is printed when one is new or has changed:

    OK      an alert, confirmation, 'reported' or ended object matched it
    LOST    only alerts matched, and the follow-up dropped them all (faded
            or rejected): a reported comet the watcher let go
    VETO    only decisions that threw it away matched (star, off-lane, ...)
    MISS    nothing matched, and the report's last point is over two hours
            older than the newest searched window
    (still being followed, or not due yet: nothing; asked again next run)

and, once each, every alert that faded or was rejected with no report
matched to it (FALSE?), to look at by eye.

The searched span starts where this audit first ran on the watcher (the
watcher's own state only keeps its last ~11 hours), and never before its
first logged decision: nothing earlier was recorded. What's been said is
kept in <live dir>/audit.json, so a verdict is printed once. --since looks
back from a given time instead, printing every verdict and remembering
nothing. A failure to reach Sungrazer is only reported after three in a row.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from .evaluate import FRAME, report_points
from .instrument import CURRENT as INSTRUMENT
from .lane import field_or_none
from .live import LIVE
from .reports import _fetch, parse
from .shortlist import FRAME_ROWS, _known_by_position

STATE = LIVE / "audit.json"
MISS_AFTER = timedelta(hours=2)  # the watcher needs ~45 min of frames to decide; give it margin
FETCH_FAILURES = 3


def as_track(rec):
    """A decision record (first, last, and x, y, vx, vy at last) as a track."""
    t1 = datetime.fromisoformat(rec["last"])
    t0 = datetime.fromisoformat(rec.get("first") or rec["last"])
    k = (t1 - t0) / FRAME
    b = {"t": t1, "x": rec["x"], "y": rec["y"], "vx": rec["vx"], "vy": rec["vy"]}
    if k <= 0:
        return [b]
    return [dict(b, t=t0, x=rec["x"] - rec["vx"] * k, y=rec["y"] - rec["vy"] * k), b]


def names_report(rec, rid):
    """Did the watcher itself tie this record to report `rid`? Older records
    carry 'Reporter report_xx_... (CCOR1)', newer ones a dict."""
    k = rec.get("report") or rec.get("known")
    return bool(k) and (k.get("id") == rid if isinstance(k, dict) else rid in str(k))


def _stem(path):
    return Path(path).stem.replace(".confirmed", "")


def candidates(log, live):
    """(kind, record or None, track) for everything the watcher has: each
    logged decision, each follow-up, each open chain."""
    out = [(r["kind"], r, as_track(r)) for r in log if r.get("last") and "x" in r]
    for f in live.get("pending", []):
        out.append(("following", f, [dict(c, t=datetime.fromisoformat(c["time"])) for c in f["track"]]))
    for ch in live.get("chains", []):
        out.append(("open chain", None, [dict(c, t=datetime.fromisoformat(c["time"])) for c in ch]))
    return out


def judge(r, pts, cands, fate, newest, field=None):
    """(verdict, matching (kind, record) pairs) for report `r` with points
    [(time, x, y)], or (None, ...) when there's nothing to say yet. `fate`
    maps an alert's file stem to its last logged kind."""
    hits = [(kind, rec) for kind, rec, tr in cands
            if (rec and names_report(rec, r["id"])) or _known_by_position(tr, [(r, pts)], field)]
    kinds = {k for k, _ in hits}
    fates = [fate.get(_stem(rec["alert"]), k) for k, rec in hits if rec and rec.get("alert")]
    if kinds & {"confirmed", "ended", "reported"} or set(fates) & {"confirmed", "ended", "reported"}:
        return "OK", hits
    if "alert" in kinds:
        return ("LOST" if fates and all(f in ("faded", "rejected") for f in fates) else "OK"), hits
    if kinds & {"following", "open chain"}:
        return None, hits
    if kinds:
        return "VETO", hits
    if newest - pts[-1][0] > MISS_AFTER:
        return "MISS", hits
    return None, hits


def false_alerts(log, matched=()):
    """{alert stem: its records} for alerts that faded or were rejected
    with no report matched to them, by the watcher or (`matched`, alert
    stems) by this audit."""
    by = {}
    for r in log:
        if r.get("alert"):
            by.setdefault(_stem(r["alert"]), []).append(r)
    return {name: hist for name, hist in by.items()
            if any(h["kind"] == "alert" for h in hist) and hist[-1]["kind"] in ("faded", "rejected")
            and name not in matched and not any(h.get("report") or h.get("known") for h in hist)}


def fetch_reports(since):
    """Sungrazer's reports, newest page first, back to a day before `since`."""
    reps = []
    for page in range(12):  # ~30 reports a page
        got = parse(_fetch(page))
        reps += got
        if not got or min(r["posted_utc"] for r in got) < (since - timedelta(days=1)).isoformat():
            break
    return reps


def audit(since=None):
    """The audit's lines for this instrument, and its state updated (unless
    `since` is given)."""
    st = json.loads(STATE.read_text()) if STATE.exists() and not since else {}
    live = json.loads((LIVE / "state.json").read_text())
    done = [datetime.fromisoformat(t) for t in live["done"]]
    if not done:
        return []
    log = [json.loads(ln) for ln in open(LIVE / "log.jsonl")] if (LIVE / "log.jsonl").exists() else []
    newest = max(done)
    start = since or (datetime.fromisoformat(st["start"]) if "start" in st else min(done))
    st["start"] = start.isoformat()
    if log:
        start = max(start, datetime.fromisoformat(log[0]["decided_utc"]))
    said = st.setdefault("verdicts", {})
    flagged = st.setdefault("false", [])

    fate = {_stem(r["alert"]): r["kind"] for r in log if r.get("alert")}
    cands = candidates(log, live)
    field = field_or_none((FRAME_ROWS, 2048))
    reps = fetch_reports(start)
    rp = {}
    for d in sorted({(newest - timedelta(days=n)).date() for n in range((newest - start).days + 2)}):
        for r, pts, _ in report_points(reps, d):
            if len(pts) >= 2:
                rp[r["id"]] = (r, pts)
    out, matched = [], set()
    for rid, (r, pts) in sorted(rp.items(), key=lambda kv: kv[1][1][0][0]):
        if pts[-1][0] < start or pts[0][0] > newest:
            continue  # seen while the watcher wasn't searching
        verdict, hits = judge(r, pts, cands, fate, newest, field)
        matched |= {_stem(rec["alert"]) for _, rec in hits if rec and rec.get("alert")}
        if verdict is None or said.get(rid) == verdict:
            continue
        said[rid] = verdict
        detail = ", ".join(f"{k} {rec.get('decided_utc', '')[11:16]} {rec.get('note', '')}".strip()
                           for k, rec in hits if rec) or "nothing near its path"
        out.append(f"{verdict} {rid} {r['reporter']} {r['group']} {r['kind']}, "
                   f"points {pts[0][0]:%m-%d %H:%M}-{pts[-1][0]:%H:%M} "
                   f"at ({pts[0][1]:.0f},{pts[0][2]:.0f})->({pts[-1][1]:.0f},{pts[-1][2]:.0f}): {detail}")
    for name, hist in false_alerts(log, matched).items():
        if name in flagged or datetime.fromisoformat(hist[-1]["decided_utc"]) < start:
            continue
        flagged.append(name)
        out.append(f"FALSE? {name}: alert {hist[0]['decided_utc'][5:16]}, then {hist[-1]['kind']} "
                   f"{hist[-1]['decided_utc'][11:16]} {hist[-1].get('note', '')}".rstrip())
    if not since:
        st["fetch_failures"] = 0
        STATE.write_text(json.dumps(st, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", type=datetime.fromisoformat,
                    help="look back from this time, print every verdict, remember nothing")
    args = ap.parse_args()
    try:
        lines = audit(args.since)
    except OSError as e:  # Sungrazer resets connections now and then: say so only if it keeps up
        if args.since:
            raise
        st = json.loads(STATE.read_text()) if STATE.exists() else {}
        st["fetch_failures"] = st.get("fetch_failures", 0) + 1
        STATE.write_text(json.dumps(st, indent=1))
        if st["fetch_failures"] >= FETCH_FAILURES:
            print(f"[{INSTRUMENT.label}] audit: Sungrazer unreachable {st['fetch_failures']} times running: {e}",
                  flush=True)
        return
    for line in lines:
        print(f"[{INSTRUMENT.label}] {line}", flush=True)


if __name__ == "__main__":
    main()
