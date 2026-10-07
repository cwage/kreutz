#!/bin/bash
# usage: scripts/archive-gpu.sh NEWEST OLDEST [N]
# Same as archive.sh but one GPU process doing a whole day at a time
# (docker-compose.gpu.yml). Keep it to a single GPU process.
cd "$(dirname "$0")/.."
START=$1; END=$2; N=${3:-8}
mkdir -p data/logs
d=$START
while [[ ! "$d" < "$END" ]]; do
  log=data/logs/run-$d-n$N-gpu.log
  if ! grep -q "^/data/runs/n$N/" "$log" 2>/dev/null; then
    echo "$(date '+%F %T') starting $d"
    next=$(date -d "$d + 1 day" +%F)
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm -T kreutz kreutz.run "${d}T00:00" "${next}T00:00" --n $N > "$log" 2>&1
  fi
  d=$(date -d "$d - 1 day" +%F)
done
echo "$(date '+%F %T') archive done"
