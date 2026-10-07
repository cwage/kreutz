#!/bin/bash
# usage: scripts/lane.sh START END [N]      (ISO dates, END exclusive)
# Lane-constrained search (run.py --lane) over a span, in three parallel
# CPU-capped blocks, then shortlist the result.
cd "$(dirname "$0")/.."
S=$1; E=$2; N=${3:-16}
mkdir -p data/logs
m1=$(date -d "$S + $(( ($(date -d "$E" +%s) - $(date -d "$S" +%s)) / 86400 / 3 )) days" +%F)
m2=$(date -d "$S + $(( ($(date -d "$E" +%s) - $(date -d "$S" +%s)) * 2 / 86400 / 3 )) days" +%F)
for b in "$S $m1" "$m1 $m2" "$m2 $E"; do
  set -- $b
  docker compose run --rm -T kreutz kreutz.run "${1}T00:00" "${2}T00:00" --n $N --lane > "data/logs/lane-$1.log" 2>&1 &
done
wait
docker compose run --rm -T kreutz kreutz.shortlist $(ls data/runs/lane$N/*.jsonl | sed 's#^#/#') --threshold 5.0 > data/logs/lane-shortlist.log 2>&1
cat data/logs/lane-shortlist.log
