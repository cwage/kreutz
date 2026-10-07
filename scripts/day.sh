#!/bin/bash
# usage: scripts/day.sh YYYY-MM-DD [N]
# Search one day of archived frames as three 8-hour chunks in parallel
# (each a CPU-capped container, see docker-compose.yml). Logs go to
# data/logs/run-<day>-n<N>-<chunk>.log.
cd "$(dirname "$0")/.."
D=$1; N=${2:-8}
NEXT=$(date -d "$D + 1 day" +%F)
mkdir -p data/logs
for h in 00 08 16; do
  e=$((10#$h + 8)); if [ $e -eq 24 ]; then E="${NEXT}T00:00"; else E="${D}T$(printf %02d $e):00"; fi
  docker compose run --rm -T kreutz kreutz.run ${D}T$h:00 $E --n $N > data/logs/run-$D-n$N-$h.log 2>&1 &
done
wait
