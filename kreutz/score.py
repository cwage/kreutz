"""How the live watcher did: each alert against what Sungrazer got.

    docker compose run --rm kreutz kreutz.reports 2026-10-01 2026-10-08   # first, to cover the dates
    docker compose run --rm kreutz kreutz.score

For every alert in data/live/log.jsonl (and the older alert files that
predate the log), the matching CCOR-1 report by position, if any has
appeared since, and the earliest Kreutz report from another instrument in
the day before; then our lead or lag in minutes: the report's posting
time minus when the watcher decided. Positive means the watcher was
first. Reports only say someone saw it; whether an other-instrument
report is the same object is for a person to judge.
"""

import json
from datetime import datetime, timezone

from . import DATA
from .evaluate import matches, report_points
from .live import LIVE, other_instruments


def alerts():
    seen = set()
    out = []
    confirmed = {}  # alert file name -> when the follow-up confirmed it
    p = LIVE / "log.jsonl"
    if p.exists():
        for ln in open(p):
            r = json.loads(ln)
            if r["kind"] == "confirmed":
                confirmed[r.get("alert", "").rsplit("/", 1)[-1]] = r["decided_utc"]
            if r["kind"] == "alert":
                out.append(r)
                seen.add(r.get("alert", "").rsplit("/", 1)[-1])
    for r in out:
        r["confirmed_utc"] = confirmed.get(r.get("alert", "").rsplit("/", 1)[-1])
    for f in sorted((LIVE / "alerts").glob("*.json")):
        if f.name in seen or f.name.endswith(".confirmed.json"):
            continue
        d = json.load(open(f))
        track = d["track"] if isinstance(d, dict) else d
        decided = d.get("decided_utc") if isinstance(d, dict) else None
        decided = decided or datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        out.append({"kind": "alert", "decided_utc": decided, "first": track[0]["time"],
                    "last": track[-1]["time"], "track": track, "alert": str(f),
                    "x": track[-1]["x"], "y": track[-1]["y"],
                    "vx": track[-1]["vx"], "vy": track[-1]["vy"]})
    return out


def main():
    reps = [json.loads(l) for p in sorted((DATA / "reports").glob("*.jsonl")) for l in open(p)]
    byid = {}
    for r in reps:
        byid[r["id"]] = r  # later files override earlier scrapes of the same report
    reps = list(byid.values())
    rows = []
    for a in alerts():
        track = a.get("track")
        if track is None:  # from the log: reconstruct a two-point track from first/last
            track = [{"time": a["first"], "x": a["x"] - a["vx"] * 2, "y": a["y"] - a["vy"] * 2,
                      "vx": a["vx"], "vy": a["vy"]},
                     {"time": a["last"], "x": a["x"], "y": a["y"], "vx": a["vx"], "vy": a["vy"]}]
        if (LIVE / "alerts" / a["alert"].rsplit("/", 1)[-1]).exists():
            d = json.load(open(LIVE / "alerts" / a["alert"].rsplit("/", 1)[-1]))
            track = d["track"] if isinstance(d, dict) else d
        tt = [dict(c, t=datetime.fromisoformat(c["time"])) for c in track]
        t = tt[-1]["t"]
        decided = datetime.fromisoformat(a["decided_utc"])
        ccor1 = None
        for r, pts, posted in report_points(reps, t.date()):
            if matches(tt, pts) and (ccor1 is None or posted < ccor1[1]):
                ccor1 = (f"{r['reporter']} {r['id']}", posted)
        others = other_instruments(track, reps)
        first_other = min(others, key=lambda o: o["posted_utc"]) if others else None
        rows.append((a, ccor1, first_other, decided))

    def lead(posted):
        return f"{(posted - decided).total_seconds() / 60:+.0f} min"

    def leads(posted, a):
        """lead at the candidate stage, and at the confirmed stage if there was one"""
        s = lead(posted)
        if a.get("confirmed_utc"):
            s += f" / {(posted - datetime.fromisoformat(a['confirmed_utc'])).total_seconds() / 60:+.0f} confirmed"
        return s

    print("lead: minutes between our alert and the report's posting (+ means we were first), "
          "candidate stage / confirmed stage")
    print(f"{'alert (last frame)':22} {'decided':17} {'confirmed':11} {'CCOR-1 report':40} {'lead':24} other instrument")
    for a, ccor1, other, decided in rows:
        conf = a["confirmed_utc"][5:16] if a.get("confirmed_utc") else "-"
        c1 = f"{ccor1[0][:38]:40} {leads(ccor1[1], a):24}" if ccor1 else f"{'-':40} {'':24}"
        if other:
            op = datetime.strptime(other["posted_utc"], "%Y-%m-%dT%H:%M:%SZ")
            ot = f"{other['reporter']} {other['telescope']} {leads(op, a)}"
        else:
            ot = "-"
        print(f"{a['last'][:16]:22} {a['decided_utc'][5:16]:17} {conf:11} {c1} {ot}")


if __name__ == "__main__":
    main()
