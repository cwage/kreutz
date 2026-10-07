#!/bin/bash
# usage: scripts/weekly.sh [END_DATE] [--gpu]
# The week of UTC days ending before END_DATE (default: today, so the
# seven complete days through yesterday; NOAA's archive runs about a day
# behind). Fetches frames and Sungrazer reports, runs the full 8-frame
# search day by day (three CPU-capped chunks, or one GPU process with
# --gpu) and the lane search over the whole span alongside, then collates
# into data/weekly/<start>_<end>/ (kreutz.weekly). Days already searched
# are skipped, so it can be rerun.
set -e
cd "$(dirname "$0")/.."
GPU=0; END=""
for a in "$@"; do case $a in --gpu) GPU=1;; *) END=$a;; esac; done
END=${END:-$(date -u +%F)}
START=$(date -d "$END - 7 days" +%F)
LAST=$(date -d "$END - 1 day" +%F)
mkdir -p data/logs
log=data/logs/weekly-$START.log
exec > >(tee -a "$log") 2>&1
echo "$(date -u '+%F %T') weekly $START..$END"

docker compose run --rm -T kreutz kreutz.fetch "$(date -d "$START - 1 day" +%F)" "$LAST"
docker compose run --rm -T kreutz kreutz.reports "$(date -d "$START - 7 days" +%F)" "$END"

docker compose run --rm -T kreutz kreutz.run "${START}T00:00" "${END}T00:00" --n 16 --lane > data/logs/lane-$START.log 2>&1 &
LANE=$!
if [ $GPU = 1 ]; then scripts/archive-gpu.sh "$LAST" "$START" 8; else scripts/archive.sh "$LAST" "$START" 8; fi
wait $LANE

docker compose run --rm -T kreutz kreutz.weekly "$START" "$END"
echo "$(date -u '+%F %T') weekly done"
