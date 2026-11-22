"""The live watcher's past alerts against a direction gate: how far each
one's motion was off straight at the Sun (live.sunward, the page gate CCOR
and C3 use) and off the Kreutz lane's flow at its position (lane.flow_at),
for the comets (reported, or confirmed) and for the alerts that faded
unreported.

    docker compose run --rm -T -e KREUTZ_INSTRUMENT=ccor2 -w /srv -e PYTHONPATH=/srv -v $PWD/scripts:/srv/scripts:ro --entrypoint python kreutz scripts/flow-gate-check.py

The lane is rebuilt for each alert from the samples taken before it
(less --exclude-h hours), since the lane samples include the reports of
the very comets being tested: a comet's own report makes its flow match
by construction (the CCOR-2 comet of 2026-11-22 01:30 read 126 px off the
lane at its alert, 0 px once its reporter's points were samples).
"""
import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

from kreutz import lane, live  # noqa: E402
from kreutz.instrument import CURRENT as INSTRUMENT  # noqa: E402


def alerts(log):
    """{alert stem: {"rec": the alert or reported record, "kinds": set,
    "known": reporter or None}} from the watcher's log."""
    out = {}
    for line in open(log):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        a = r.get("alert")
        if not a:
            continue
        e = out.setdefault(Path(a).stem, {"kinds": set(), "known": None, "rec": None})
        e["kinds"].add(r["kind"])
        if r["kind"] in ("alert", "reported") and e["rec"] is None:
            e["rec"] = r
        k = (r.get("known") or {}).get("reporter") or r.get("report")
        if k:
            e["known"] = k
    # a comet still being followed is named in the saved state, not the log
    try:
        for f in json.load(open(live.LIVE / "state.json")).get("pending", []):
            k = (f.get("known") or {}).get("reporter")
            if k and Path(f["alert"]).stem in out:
                out[Path(f["alert"]).stem]["known"] = k
    except (OSError, ValueError):
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", type=datetime.fromisoformat, default=datetime(2026, 10, 20))
    ap.add_argument("--exclude-h", type=float, default=6.0,
                    help="leave out lane samples from this long before each alert on (default 6)")
    ap.add_argument("--cuts", type=float, nargs="*", default=[20, 25, 30, 35, 45])
    args = ap.parse_args()
    H, W = live.FRAME_ROWS, live.FRAME_COLS
    sun_xy = INSTRUMENT.sun_xy
    rows = []
    for stem, e in sorted(alerts(live.LIVE / "log.jsonl").items()):
        r = e["rec"]
        if r is None or not all(k in r for k in ("x", "y", "vx", "vy", "last")):
            continue
        t = datetime.fromisoformat(r["last"])
        if t < args.since:
            continue
        real = bool(e["known"]) or "confirmed" in e["kinds"]
        if not real and "faded" not in e["kinds"]:
            continue  # still open and unnamed, or ended some other way
        cut = t - timedelta(hours=args.exclude_h)
        samples = [s for s in lane.lane_samples(t) if len(s) < 6 or datetime.fromisoformat(s[5]) < cut]
        try:
            field = lane.build_field((H, W), samples=samples, when=t)
        except FileNotFoundError:
            continue
        fx, fy = lane.flow_at(field, r["x"], r["y"], H)
        cos = (r["vx"] * fx + r["vy"] * fy) / (np.hypot(r["vx"], r["vy"]) * np.hypot(fx, fy) + 1e-30)
        flow = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        off, speed = live.sunward(r, sun_xy)
        px = lane.lane_px(field, r["x"], r["y"], H)
        rows.append((stem, real, e["known"], off, flow, px, speed))
    print(f"{INSTRUMENT.label}: {sum(r[1] for r in rows)} comets, {sum(not r[1] for r in rows)} faded alerts since {args.since:%Y-%m-%d}")
    print(f"{'alert':26} {'':6} {'sunward':>7} {'flow':>5} {'lane px':>7} {'px/fr':>5}  reporter")
    for stem, real, known, off, flow, px, speed in rows:
        print(f"{stem:26} {'COMET' if real else 'faded':6} {off:7.0f} {flow:5.0f} {px:7.0f} {speed:5.1f}  {known or ''}")
    n_c, n_f = sum(r[1] for r in rows), sum(not r[1] for r in rows)
    for name, col in (("sunward", 3), ("flow", 4)):
        print(f"\n{name} within:")
        for c in args.cuts:
            comets = sum(1 for r in rows if r[1] and r[col] <= c)
            faded = sum(1 for r in rows if not r[1] and r[col] <= c)
            print(f"  {c:3.0f} deg: comets {comets}/{n_c}, faded alerts {faded}/{n_f}")
    print("\nflow within 20 deg where a lane sample is within N px, else sunward within 20:")
    for near in (150, 200, 250, 300):
        ok = [r for r in rows if (r[4] <= 20 if r[5] <= near else r[3] <= 20)]
        print(f"  N {near:3d}: comets {sum(r[1] for r in ok)}/{n_c}, faded alerts {sum(not r[1] for r in ok)}/{n_f}")


if __name__ == "__main__":
    sys.exit(main())
