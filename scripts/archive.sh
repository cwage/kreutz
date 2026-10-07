#!/bin/bash
# usage: scripts/archive.sh NEWEST OLDEST [N]
# Search every day from NEWEST back to OLDEST with scripts/day.sh. A day
# counts as done when all three of its chunk logs (or a whole-day GPU log
# from archive-gpu.sh) end with the output path run.py prints on
# completion, so this can be stopped and rerun.
cd "$(dirname "$0")/.."
START=$1; END=$2; N=${3:-8}
d=$START
while [[ ! "$d" < "$END" ]]; do
  chunks=$(grep -l "^/data/runs/n$N/" data/logs/run-$d-n$N-0*.log data/logs/run-$d-n$N-16.log 2>/dev/null | wc -l)
  if [ "$chunks" -lt 3 ] && ! grep -q "^/data/runs/n$N/" data/logs/run-$d-n$N-gpu.log 2>/dev/null; then
    echo "$(date -u '+%F %T') starting $d"
    scripts/day.sh "$d" "$N"
  fi
  d=$(date -d "$d - 1 day" +%F)
done
echo "$(date -u '+%F %T') archive done"
