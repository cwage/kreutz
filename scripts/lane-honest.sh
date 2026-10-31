#!/bin/bash
# Test the lane search honestly: for each day, build the instrument's lane
# only from the Kreutz reports of the 35 days before it (kreutz.lanesamples
# --from-reports), run the lane search for that day, then evaluate with
# kreutz.evaluate. The shipped lane file is restored afterwards.
#   scripts/lane-honest.sh ccor1 "2026-10-13 2026-10-15"
# Runs land in data/runs<tag>/lane8-honest/. On 2026-10-21 this showed a
# report-built lane drifting faster than five weeks of samples can track:
# the honest lane missed what a lane refitted to the test comets found.
cd "$(dirname "$0")/.."
K=${1:?instrument key, e.g. ccor1}; DAYS=${2:?days, space-separated}
if [ "$K" = ccor1 ]; then TAG=; LANE=kreutz/lane-samples.json; else TAG=-$K; LANE=kreutz/lane-samples-$K.json; fi
SRC=$(mktemp); KREUTZ_INSTRUMENT=$K docker compose run --rm -T kreutz kreutz.lanesamples --from-reports 2>/dev/null > "$SRC"
cp "$LANE" "$LANE.shipped"
mkdir -p data/runs$TAG/lane8-honest
for D in $DAYS; do
  python3 - "$D" "$SRC" "$LANE" <<'PY'
import json, sys
from datetime import date, timedelta
D = date.fromisoformat(sys.argv[1]); lo = (D - timedelta(days=35)).isoformat(); hi = D.isoformat()
keep = [x for x in json.load(open(sys.argv[2])) if lo <= x[5][:10] < hi]
json.dump(keep, open(sys.argv[3], "w"))
print(f"{D}: lane from {len(keep)} report points dated {lo}..{hi}", flush=True)
PY
  KREUTZ_INSTRUMENT=$K docker compose run --rm -T kreutz kreutz.run ${D}T00:00 --n 8 --lane > data/logs/run-$K-$D-lane8-honest.log 2>&1
  mv data/runs$TAG/lane8/${D//-/}T0000_*.jsonl data/runs$TAG/lane8-honest/
  F=$(ls data/runs$TAG/lane8-honest/${D//-/}T0000_*.jsonl | sed 's|^data|/data|')
  KREUTZ_INSTRUMENT=$K docker compose run --rm -T kreutz kreutz.evaluate $F --threshold 5.0 5.5 6.0 2>&1 | grep -E '^threshold|found|missed'
done
mv "$LANE.shipped" "$LANE"; rm -f "$SRC"
