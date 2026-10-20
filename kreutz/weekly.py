"""Collate a week of searching into a shortlist, contact sheets and a summary.

    docker compose run --rm kreutz kreutz.weekly 2026-09-27 2026-10-04 [--known 6]

scripts/weekly.sh runs the whole week (fetch, reports, search, then this).
This step reads every run file under data/runs/n8/ and data/runs/lane16/,
keeps the windows in [start, end), shortlists both (shortlist.py), merges
them (a lane track that follows a full-search track within MERGE_PX is the
same object), and writes to data/weekly/<start>_<end>/:

    shortlist.json   every fired track, known or scored for persistence
    to-vet.json      what the sheets show: all candidates, then the
                     brightest --known reported comets (for the pretty ones)
    sheet-NN.png     contact sheets of vetting strips for to-vet.json
    summary.txt      the numbers, and the list to look at
"""

import argparse
import json
from datetime import datetime, timedelta

import numpy as np

from . import DATA
from .frames import INSTRUMENT
from .evaluate import FRAME
from .sheet import make_sheets
from .shortlist import MAYBE, STRONG, candidates, shortlist

SOURCES = {"n8": 6.0, "lane16": 5.0}  # run kind -> shortlist threshold
MERGE_PX = 20
MERGE_FRAMES = 8


def runs_overlapping(kind, start, end):
    out = []
    for p in sorted((DATA / f"runs{INSTRUMENT.tag}" / kind).glob("*.jsonl")):
        a, b = (datetime.strptime(s, "%Y%m%dT%H%M") for s in p.stem.split("_"))
        if a < end and b > start:
            out.append(p)
    return out


def same_object(a, b):
    """Two fired records that are one body: b's position, carried along
    its own motion to a's time, lands within MERGE_PX of a."""
    ta, tb = datetime.fromisoformat(a["time"]), datetime.fromisoformat(b["time"])
    k = (ta - tb) / FRAME
    if abs(k) > MERGE_FRAMES:
        return False
    return np.hypot(b["x"] + b["vx"] * k - a["x"], b["y"] + b["vy"] * k - a["y"]) <= MERGE_PX


def collate(start, end, n_known):
    records = []
    counts = {}
    for kind, th in SOURCES.items():
        runs = runs_overlapping(kind, start, end)
        recs = shortlist(runs, th, start=start, end=end) if runs else []
        counts[kind] = (len(runs), len(recs))
        for r in recs:
            r["source"] = kind
            if not any(same_object(r, q) for q in records):
                records.append(r)
    known = sorted((r for r in records if "known" in r), key=lambda r: -max(r["scores"]))
    cands = candidates(records)
    to_vet = cands + known[:n_known]
    return records, cands, known, to_vet, counts


def summary(start, end, records, cands, known, to_vet, counts, sheets):
    lines = [f"kreutz weekly: {start:%Y-%m-%d} to {end:%Y-%m-%d} (end exclusive)", ""]
    for kind, (nr, nt) in counts.items():
        lines.append(f"{kind}: {nr} run files, {nt} tracks fired")
    lines.append(f"{len(records)} distinct tracks after merging, {len(known)} on reported comets")
    lines.append(f"{sum(r['persistence'] >= STRONG for r in cands)} strong and "
                 f"{sum(r['persistence'] < STRONG for r in cands)} maybe candidates "
                 f"(persistence >= {STRONG} / >= {MAYBE})")
    lines += ["", "Reported comets seen (brightest first):"]
    for r in known:
        lines.append(f"  {r['time'][:16]} ({r['x']:.0f},{r['y']:.0f}) v=({r['vx']:+.1f},{r['vy']:+.1f}) "
                     f"peak {max(r['scores']):.1f} over {r['windows']} windows  {r['known']}"
                     + (" [by time+lane]" if r.get("known_how") == "time+lane" else ""))
    lines += ["", "To look at (sheet numbers match to-vet.json order):"]
    for n, r in enumerate(to_vet):
        tag = f"persistence {r['persistence']:+.1f}" if "persistence" in r else r["known"]
        lines.append(f"  #{n} {r['time'][:16]} ({r['x']:.0f},{r['y']:.0f}) "
                     f"v=({r['vx']:+.1f},{r['vy']:+.1f}) {r['source']} {tag}")
    lines += ["", "Sheets: " + ", ".join(p.name for p in sheets)]
    lines += ["", "Anything unexplained: check Sungrazer's reports for every instrument "
              "(CCOR-1, CCOR-2, LASCO) before thinking it's new."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("start", type=datetime.fromisoformat)
    ap.add_argument("end", type=datetime.fromisoformat, nargs="?")
    ap.add_argument("--known", type=int, default=6, help="brightest reported comets to sheet")
    ap.add_argument("--per", type=int, default=6)
    args = ap.parse_args()
    end = args.end or args.start + timedelta(days=7)
    out = DATA / f"weekly{INSTRUMENT.tag}" / f"{args.start:%Y-%m-%d}_{end:%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    records, cands, known, to_vet, counts = collate(args.start, end, args.known)
    (out / "shortlist.json").write_text(json.dumps(records, indent=1, default=str))
    (out / "to-vet.json").write_text(json.dumps(to_vet, indent=1, default=str))
    sheets = make_sheets(to_vet, out, args.per) if to_vet else []
    text = summary(args.start, end, records, cands, known, to_vet, counts, sheets)
    (out / "summary.txt").write_text(text)
    print(text)
    print(out)


if __name__ == "__main__":
    main()
