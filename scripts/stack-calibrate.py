"""Where does the line stack (kreutz.stackcheck) put real comets against
false alerts? Runs it over every live alert of this instrument since
--from, along the line through the alert's own track, from the alert's
stacking window to four hours past it, and writes per alert the z the
stack had reached at the alert, one, two and four hours on, with the
audit's label (an alert the audit matched to a Sungrazer report is real,
one it flags FALSE? is not).

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor1 -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro --entrypoint python kreutz scripts/stack-calibrate.py --from 2026-10-23

-> data/calib/<instrument>.json and a table on stdout. Then
scripts/stack-calibrate.py --table reads the files for all instruments.
"""
import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timedelta

warnings.simplefilter("ignore")

from kreutz import DATA  # noqa: E402

OUT = DATA / "calib"
HOURS = (0, 1, 2, 4)


def labels(since):
    """{alert stem: "false"} from the audit's FALSE? lines; the rest of the
    alerts it saw are matched to reports."""
    from kreutz.audit import audit
    flagged = {}
    for ln in audit(since=since):
        if ln.startswith("FALSE? "):
            flagged[ln.split()[1].rstrip(":")] = "false"
    return flagged


def run(args):
    from kreutz.audit import LIVE
    from kreutz.instrument import CURRENT as INSTRUMENT
    from kreutz.stackcheck import check
    flagged = labels(args.start)
    recs = [json.loads(l) for l in open(LIVE / "log.jsonl")]
    kinds = {}
    for r in recs:
        if r.get("alert"):
            kinds.setdefault(r["alert"].rsplit("/", 1)[-1].replace(".json", ""), []).append(r["kind"])
    out = []
    files = sorted(p for p in (LIVE / "alerts").glob("*.json") if not p.name.endswith(".confirmed.json"))
    for p in files:
        d = json.loads(p.read_text())
        if not isinstance(d, dict):  # the oldest alert files are the bare track
            d = {"track": d}
        track = d["track"]
        t_alert = datetime.fromisoformat(track[-1]["time"])
        if t_alert < args.start:
            continue
        t0 = time.time()
        st = check(track, end=t_alert + timedelta(hours=max(HOURS)))
        row = {"name": p.stem, "alert_utc": t_alert.isoformat(), "paged": bool(d.get("paged")),
               "kind": "reported" if d.get("known") else "alert",
               "fates": kinds.get(p.stem, []),
               "label": flagged.get(p.stem, "real"),
               "n": st.n, "z": {str(h): st.z_at(t_alert + timedelta(hours=h)) for h in HOURS},
               "z_max": max((z for z in (st.z_at(t) for t in st.times) if z is not None), default=None),
               "secs": round(time.time() - t0)}
        out.append(row)
        zs = " ".join(f"+{h}h {row['z'][str(h)]:5.1f}" if row["z"][str(h)] is not None else f"+{h}h   -  " for h in HOURS)
        print(f"{row['label']:5s} {'paged' if row['paged'] else 'held ':5s} {p.stem:28s} n {st.n:3d}  {zs}  "
              f"max {row['z_max'] if row['z_max'] is not None else float('nan'):5.1f}  {row['secs']}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{INSTRUMENT.key}.json").write_text(json.dumps(out, indent=1))
    print(f"{INSTRUMENT.label}: {len(out)} alerts -> {OUT / (INSTRUMENT.key + '.json')}")


# Alerts the audit labels FALSE? that are known to be real: the first alert
# of the CCOR-1 comet of 2026-11-02 (reported from its 18:00 alert; nothing
# links the two in the log, and position matching can't reach back 8 h).
KNOWN_REAL = {"20261102T1015_1775_1536"}


def table(args):
    import numpy as np
    rows = []
    for p in sorted(OUT.glob("*.json")):
        for r in json.loads(p.read_text()):
            r["inst"] = p.stem
            if r["name"] in KNOWN_REAL:
                r["label"] = "real"
            rows.append(r)
    if args.paged:
        rows = [r for r in rows if r["paged"]]
    for label in ("real", "false"):
        print(f"\n{label}:")
        for r in sorted((r for r in rows if r["label"] == label), key=lambda r: r["alert_utc"]):
            zs = " ".join(f"{r['z'][str(h)]:6.1f}" if r["z"][str(h)] is not None else "   -  " for h in HOURS)
            print(f"  {r['inst']:5s} {'paged' if r['paged'] else 'held ':5s} {r['name']:28s} {zs}  "
                  f"max {r['z_max'] if r['z_max'] is not None else float('nan'):5.1f}  {'/'.join(r['fates'][:3])}")
    print("\nz by hour after the alert (median [min, max]):")
    for label in ("real", "false"):
        for h in HOURS:
            v = [r["z"][str(h)] for r in rows if r["label"] == label and r["z"][str(h)] is not None]
            if v:
                print(f"  {label:5s} +{h}h  n {len(v):3d}  {np.median(v):5.1f} [{min(v):5.1f}, {max(v):5.1f}]")
    for thr in (4, 5, 6, 7, 8, 10):
        for h in HOURS:
            real = [r for r in rows if r["label"] == "real" and r["z"][str(h)] is not None]
            false = [r for r in rows if r["label"] == "false" and r["z"][str(h)] is not None]
            if real and false:
                print(f"  z >= {thr:2d} at +{h}h: catches {sum(r['z'][str(h)] >= thr for r in real):2d}/{len(real)} real, "
                      f"passes {sum(r['z'][str(h)] >= thr for r in false):2d}/{len(false)} false")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="start", type=datetime.fromisoformat, default=datetime(2026, 10, 23))
    ap.add_argument("--table", action="store_true", help="summarise data/calib/*.json for all instruments")
    ap.add_argument("--paged", action="store_true", help="with --table: only alerts that passed the page gates")
    args = ap.parse_args()
    (table if args.table else run)(args)


if __name__ == "__main__":
    sys.exit(main())
