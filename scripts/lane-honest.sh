#!/bin/bash
# Test the lane search honestly: for each day, build the instrument's lane
# only from the Kreutz reports posted before that day whose points fall in
# the 35 days before it (kreutz.lanesamples' from_reports, over the reports
# posted by then), run the lane search for that day, then evaluate with
# kreutz.evaluate. The shipped lane file is restored afterwards, also when
# the run is interrupted.
#   scripts/lane-honest.sh ccor1 "2026-10-13 2026-10-15"
# Runs land in data/runs<tag>/lane8-honest/. On 2026-10-21 this showed a
# report-built lane drifting faster than five weeks of samples can track:
# the honest lane missed what a lane refitted to the test comets found.
cd "$(dirname "$0")/.."
K=${1:?instrument key, e.g. ccor1}; DAYS=${2:?days, space-separated}
if [ "$K" = ccor1 ]; then TAG=; LANE=kreutz/lane-samples.json; else TAG=-$K; LANE=kreutz/lane-samples-$K.json; fi
# An interrupted earlier run leaves its copy of the shipped lane behind:
# keep that one rather than backing up a test subset over it.
[ -e "$LANE.shipped" ] || cp "$LANE" "$LANE.shipped"
SRC=$(mktemp)
trap 'mv -f "$LANE.shipped" "$LANE"; rm -f "$SRC"' EXIT
trap 'exit 130' INT TERM
mkdir -p data/runs$TAG/lane8-honest data/logs
for D in $DAYS; do
  # A sample's velocity is its report's first-to-last motion, so a report
  # posted on the test day or later could carry that day's motion even on
  # earlier points. Only reports posted before the day are used, and
  # retractions are those known by then.
  KREUTZ_INSTRUMENT=$K docker compose run --rm -T --entrypoint python kreutz -c '
import json, sys
import kreutz.lanesamples as L
cut = sys.argv[1]
posted = L.all_reports
L.all_reports = lambda: [r for r in posted() if r.get("posted_utc", "9999") < cut]
print(json.dumps(L.from_reports()))' "$D" 2>/dev/null > "$SRC"
  python3 - "$D" "$SRC" "$LANE" <<'PY'
import json, sys
from datetime import date, timedelta
D = date.fromisoformat(sys.argv[1]); lo = (D - timedelta(days=35)).isoformat(); hi = D.isoformat()
keep = [x for x in json.load(open(sys.argv[2])) if lo <= x[5][:10] < hi]
json.dump(keep, open(sys.argv[3], "w"))
print(f"{D}: lane from {len(keep)} points of reports posted before it, dated {lo}..{hi}", flush=True)
PY
  KREUTZ_INSTRUMENT=$K docker compose run --rm -T kreutz kreutz.run ${D}T00:00 --n 8 --lane > data/logs/run-$K-$D-lane8-honest.log 2>&1
  mv data/runs$TAG/lane8/${D//-/}T0000_*.jsonl data/runs$TAG/lane8-honest/
  F=$(ls data/runs$TAG/lane8-honest/${D//-/}T0000_*.jsonl | sed 's|^data|/data|')
  KREUTZ_INSTRUMENT=$K docker compose run --rm -T kreutz kreutz.evaluate $F --threshold 5.0 5.5 6.0 2>&1 | grep -E '^threshold|found|missed'
done
