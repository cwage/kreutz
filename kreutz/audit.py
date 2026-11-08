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

    OK      an alert, confirmation, 'reported' or ended object matched it,
            or an alert that paged: the alert did its job even if the
            follow-up let go afterwards (the comet of 2026-11-02: paged
            at 10:32, faded at 12:01, reported from that page at 18:29)
    LOST    only unpaged alerts matched, and the follow-up dropped them all
            (faded or rejected): a reported comet the watcher let go
    VETO    only decisions that threw it away matched (star, off-lane, ...)
    MISS    nothing matched, and the report's last point is over two hours
            older than the newest searched window
    (still being followed, or not due yet: nothing; asked again next run)

A verdict that matched says when we had the object against the reporter:
our first frame against their first point, and when the watcher decided
against when they posted. And, once each, every alert that faded or was
rejected with no report matched to it (FALSE?, 'paged' when it cost a
page), to look at by eye; one a later report matches is taken back
(NOT FALSE), since the report may come hours after the alert.

    kreutz.audit --loop [--digest HH:MM]

audits every instrument with a watcher's state on disk, every half hour,
each in its own process (the instrument is fixed per process), and
appends the lines to data/audit.log. With --digest, once a day at that
UTC time the day's lines, with how many alerts paged, go to the ntfy
topic (NTFY_TOPIC, as kreutz.notify) at low priority: what to read over
coffee instead of watching a log. Meant as a container alongside the
watchers:

    docker compose run --rm -d --name kreutz-audit kreutz kreutz.audit --loop --digest 13:00

A report its reporter retracted gets no verdict (one given before the
retraction is dropped, with a RETRACTED line), and a report and the
Corrections and Additional Observations that amend it
(evaluate.amended) share the best verdict among them, "as" the one that
earned it: the comet was caught if any of its reports' points were.

The searched span starts where this audit first ran on the watcher (the
watcher's own state only keeps its last ~11 hours), and never before its
first logged decision: nothing earlier was recorded. What's been said is
kept in <live dir>/audit.json, so a verdict is printed once. --since looks
back from a given time instead, printing every verdict and remembering
nothing. A failure to reach Sungrazer is only reported after three in a row.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import DATA
from .evaluate import FRAME, amended, report_points, retracted
from .instrument import CURRENT as INSTRUMENT, INSTRUMENTS
from .lane import field_or_none
from .live import LIVE
from .reports import _fetch, parse
from .shortlist import FRAME_COLS, FRAME_ROWS, _known_by_position

STATE = LIVE / "audit.json"
MISS_AFTER = timedelta(hours=2)  # the watcher needs ~45 min of frames to decide; give it margin
FETCH_FAILURES = 3
LOG = DATA / "audit.log"            # --loop: every line said, with when
DIGEST_STATE = DATA / "audit-digest.json"  # --digest: when the last one went
LOOP_S = 1800
DIGEST_LINES = 25                   # an ntfy message is a header: keep it short


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
        paged = any(rec.get("paged") for k, rec in hits if k == "alert")
        return ("LOST" if not paged and fates and all(f in ("faded", "rejected") for f in fates) else "OK"), hits
    if kinds & {"following", "open chain"}:
        return None, hits
    if kinds:
        return "VETO", hits
    if newest - pts[-1][0] > MISS_AFTER:
        return "MISS", hits
    return None, hits


RANK = {"OK": 0, None: 1, "LOST": 2, "VETO": 3, "MISS": 4}  # best first; None is still open


def timing(hits, r, pts):
    """When we had the object against the reporter, for a verdict's line:
    our first frame against their first point, and the watcher's decision
    against their posting. '' when no record of ours has a first frame."""
    recs = [rec for _, rec in hits if rec and rec.get("first") and rec.get("decided_utc")]
    if not recs:
        return ""
    first = min(datetime.fromisoformat(rec["first"]) for rec in recs)
    decided = min(datetime.fromisoformat(rec["decided_utc"]) for rec in recs)
    lead = (pts[0][0] - first).total_seconds() / 3600
    out = f"ours from {first:%H:%M}, theirs {pts[0][0]:%H:%M} ({lead:+.1f} h)"
    posted = r.get("posted_utc")
    if posted:
        told = datetime.fromisoformat(posted.rstrip("Z"))
        out += f"; decided {decided:%H:%M}, posted {told:%H:%M} ({(told - decided).total_seconds() / 3600:+.1f} h)"
    return out


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


def revived_too(matched_by, alerts=None):
    """`matched_by` (alert stem -> report id) with the alerts the matched
    ones revived (live.py: the alert's "revived", the faded alerts whose
    object it is) matched to the same report: position can't carry a
    report back to an alert hours earlier, but the watcher already tied
    them. `alerts` is the directory of alert files."""
    alerts = alerts or LIVE / "alerts"
    for stem, rid in list(matched_by.items()):
        p = alerts / f"{stem}.json"
        if not p.exists():
            continue
        for old in json.loads(p.read_text()).get("revived") or []:
            matched_by.setdefault(old, rid)
    return matched_by


def pager(live=None):
    """Does this watcher page (status.json's "notify", run with --notify)?
    None when its status predates the flag."""
    p = (live or LIVE) / "status.json"
    return json.loads(p.read_text()).get("notify") if p.exists() else None


def unflag(flagged, matched_by):
    """Take back the FALSE? alerts (`flagged`, in place) that a report has
    since matched (`matched_by`: alert stem -> report id); the lines
    saying so. An alert is flagged when it fades, and the report that
    came of it may be posted hours later."""
    out = []
    for name in [n for n in flagged if n in matched_by]:
        flagged.remove(name)
        out.append(f"NOT FALSE {name}: matched {matched_by[name]}")
    return out


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
    field = field_or_none((FRAME_ROWS, FRAME_COLS), when=newest)
    reps = fetch_reports(start)
    rp = {}
    for d in sorted({(newest - timedelta(days=n)).date() for n in range((newest - start).days + 2)}):
        for r, pts, _ in report_points(reps, d):
            if len(pts) >= 2:
                rp[r["id"]] = (r, pts)
    gone = retracted(reps)
    rp = {k: v for k, v in rp.items() if k not in gone}
    judged, matched_by = {}, {}
    for rid, (r, pts) in rp.items():
        if pts[-1][0] < start or pts[0][0] > newest:
            continue  # seen while the watcher wasn't searching
        judged[rid] = judge(r, pts, cands, fate, newest, field)
        for _, rec in judged[rid][1]:
            if rec and rec.get("alert"):
                matched_by.setdefault(_stem(rec["alert"]), rid)
    revived_too(matched_by)
    matched = set(matched_by)
    by_stem = {}
    for rec in log:
        if rec.get("alert"):
            by_stem.setdefault(_stem(rec["alert"]), []).append(rec)
    root = amended(rp)
    best = {}
    for rid, (verdict, _) in judged.items():
        fam = root.get(rid, rid)
        if fam not in best or RANK[verdict] < RANK[judged[best[fam]][0]]:
            best[fam] = rid
    out = [f"RETRACTED {rid}: withdrawn by its reporter, was {said.pop(rid)}" for rid in sorted(gone & set(said))]
    for rid, (r, pts) in sorted(rp.items(), key=lambda kv: kv[1][1][0][0]):
        if rid not in judged:
            continue
        verdict, hits = judged[rid]
        by = best[root.get(rid, rid)]
        lifted = RANK[judged[by][0]] < RANK[verdict]
        if lifted:
            verdict, hits = judged[by]
        if verdict is None or said.get(rid) == verdict:
            continue
        said[rid] = verdict
        detail = ", ".join(f"{k} {rec.get('decided_utc', '')[11:16]} {rec.get('note') or ''}".strip()
                           for k, rec in hits if rec) or "nothing near its path"
        if lifted:
            detail = f"as {by}: {detail}"
        # the alerts the matched ones revived count for when we first had it
        earlier = [("alert", rec) for stem, who in matched_by.items() if who == by
                   for rec in by_stem.get(stem, []) if rec["kind"] == "alert"]
        when = timing(hits + earlier, rp[by][0] if lifted else r, rp[by][1] if lifted else pts)
        out.append(f"{verdict} {rid} {r['reporter']} {r.get('group') or 'no group'} {r['kind']}, "
                   f"points {pts[0][0]:%m-%d %H:%M}-{pts[-1][0]:%H:%M} "
                   f"at ({pts[0][1]:.0f},{pts[0][2]:.0f})->({pts[-1][1]:.0f},{pts[-1][2]:.0f}): {detail}"
                   + (f"; {when}" if when else ""))
    out += unflag(flagged, matched_by)
    pages = pager()
    for name, hist in false_alerts(log, matched).items():
        if name in flagged or datetime.fromisoformat(hist[-1]["decided_utc"]) < start:
            continue
        flagged.append(name)
        gated = any(h.get("paged") for h in hist)  # passed the page gates; a page only where the watcher notifies
        cost = (", paged" if pages else ", passed the page gates") if gated else ""
        out.append(f"FALSE? {name}: alert {hist[0]['decided_utc'][5:16]}{cost}, then {hist[-1]['kind']} "
                   f"{hist[-1]['decided_utc'][11:16]} {hist[-1].get('note') or ''}".rstrip())
    if not since:
        st["fetch_failures"] = 0
        STATE.write_text(json.dumps(st, indent=1))
    return out


def watched():
    """[(instrument key, its live dir)] for the instruments a watcher has
    left state for."""
    return [(k, DATA / f"live{i.tag}") for k, i in INSTRUMENTS.items()
            if (DATA / f"live{i.tag}" / "state.json").exists()]


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def run_all(log=LOG):
    """Audit every watched instrument, each in a process of its own (the
    instrument is fixed when kreutz imports), and append what was said to
    `log` with the time; the lines."""
    lines = []
    for key, _ in watched():
        p = subprocess.run([sys.executable, "-m", "kreutz.audit"], capture_output=True, text=True,
                           env={**os.environ, "KREUTZ_INSTRUMENT": key})
        lines += [ln for ln in p.stdout.splitlines() if ln.strip()]
        if p.returncode:
            err = p.stderr.strip().splitlines()
            lines.append(f"[{INSTRUMENTS[key].label}] audit failed: {err[-1] if err else p.returncode}")
    if lines:
        now = _utcnow()
        with log.open("a") as fh:
            fh.writelines(f"{now:%Y-%m-%dT%H:%M} {ln}\n" for ln in lines)
    return lines


def logged_since(since, log=LOG):
    """The lines run_all appended to `log` at `since` or later, without their times."""
    out = []
    for ln in (open(log) if log.exists() else []):
        try:
            t = datetime.strptime(ln[:16], "%Y-%m-%dT%H:%M")
        except ValueError:
            continue
        if t >= since:
            out.append(ln[17:].rstrip("\n"))
    return out


def pages_since(since, dirs=None):
    """{instrument label: (alerts that passed the page gates, whether the
    watcher pages)} decided at `since` or later, from each watched
    instrument's log.jsonl (`dirs`: [(key, live dir)])."""
    out = {}
    for key, d in (dirs if dirs is not None else watched()):
        p = d / "log.jsonl"
        n = 0
        for ln in (open(p) if p.exists() else []):
            r = json.loads(ln)
            if r.get("kind") == "alert" and r.get("paged") and datetime.fromisoformat(r["decided_utc"]) >= since:
                n += 1
        if n:
            out[INSTRUMENTS[key].label] = (n, pager(d))
    return out


def digest(since, now, log=LOG, dirs=None):
    """(title, lines) for the day's ntfy message: how many alerts paged
    (or, on a watcher that doesn't notify, passed the page gates), then
    every verdict and NOT FALSE line said since `since`, and the FALSE?
    ones that paged (the others cost nothing)."""
    pages = pages_since(since, dirs)
    head = ", ".join(f"{n} page{'s' if n != 1 else ''} {lab}" if notifies
                     else f"{n} passed the gates {lab} (no pager)" for lab, (n, notifies) in pages.items()) or "no pages"
    said = [ln for ln in logged_since(since, log) if "FALSE?" not in ln or ", paged" in ln]
    lines = [f"{since:%m-%d %H:%M} to {now:%m-%d %H:%M} UTC: {head}"] + said[:DIGEST_LINES]
    if len(said) > DIGEST_LINES:
        lines.append(f"... {len(said) - DIGEST_LINES} more in data/audit.log")
    if not said:
        lines.append("no reports to judge, nothing taken back")
    return "kreutz: the last day", lines


def loop(digest_at=None):
    """run_all every LOOP_S, printing the lines, and at `digest_at` (a UTC
    time) once a day the digest of the last day to the ntfy topic, or to
    stdout without one. The last digest's time is kept in DIGEST_STATE so
    a restart doesn't send another."""
    from .notify import _send
    while True:
        try:
            for ln in run_all():
                print(ln, flush=True)
        except Exception as e:  # keep going: the next run may have better luck
            print(f"audit loop: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        now = _utcnow()
        last = json.loads(DIGEST_STATE.read_text()).get("sent") if DIGEST_STATE.exists() else None
        last = datetime.fromisoformat(last) if last else None
        if digest_at and now.time() >= digest_at and (last is None or last.date() < now.date()):
            title, lines = digest(last or now - timedelta(days=1), now)
            topic = os.environ.get("NTFY_TOPIC")
            try:
                if topic:
                    _send(os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/"), topic,
                          title, lines, "low", "clipboard")
                else:
                    print("\n".join([title] + lines), flush=True)
                DIGEST_STATE.write_text(json.dumps({"sent": now.isoformat()}))
            except Exception as e:
                print(f"digest: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(LOOP_S)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", type=datetime.fromisoformat,
                    help="look back from this time, print every verdict, remember nothing")
    ap.add_argument("--loop", action="store_true",
                    help=f"audit every instrument with a watcher, every {LOOP_S // 60} min, into data/audit.log")
    ap.add_argument("--digest", type=lambda s: datetime.strptime(s, "%H:%M").time(), metavar="HH:MM",
                    help="with --loop: the day's lines to the ntfy topic once a day at this UTC time")
    args = ap.parse_args()
    if args.loop:
        loop(args.digest)
        return
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
