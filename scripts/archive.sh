#!/bin/bash
# usage: scripts/archive.sh NEWEST OLDEST [N]
# Search every day from NEWEST back to OLDEST with scripts/day.sh. A day
# counts as done when all three of its chunk logs end with the output path
# run.py prints on completion, so this can be stopped and rerun.
cd "$(dirname "$0")/.."
START=$1; END=$2; N=${3:-8}
d=$START
while [[ ! "$d" < "$END" ]]; do
  done=$(grep -l "^/data/runs/n$N/" data/logs/run-$d-n$N-*.log 2>/dev/null | wc -l)
  if [ "$done" -lt 3 ]; then
    echo "$(date '+%F %T') starting $d"
    scripts/day.sh "$d" "$N"
  fi
  d=$(date -d "$d - 1 day" +%F)
done
echo "$(date '+%F %T') archive done"
