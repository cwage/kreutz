#!/bin/bash
# usage: KREUTZ_INSTRUMENT=ccor1 scripts/orbit-archive.sh N FIRST LAST [JOBS]
# Run the orbit-constrained search (kreutz.run --orbit --n N) over every
# day from FIRST to LAST (YYYY-MM-DD, inclusive), JOBS days at a time
# (default 3, each a CPU-capped container), into data/runs<tag>/orbit<N>/.
# A day counts as done when its log, data/logs/orbit-<instrument>-<day>-n<N>.log
# (orbit-<instrument>-<day>.log for the n8 runs made before this script),
# ends with the output path run.py prints on completion, so this can be
# stopped and rerun. scripts/threshold-replay.py replays the runs.
cd "$(dirname "$0")/.."
N=$1; FIRST=$2; LAST=$3; JOBS=${4:-3}
INST=${KREUTZ_INSTRUMENT:-ccor1}
mkdir -p data/logs
day() {
  local d=$1 log=data/logs/orbit-$INST-$1-n$N.log
  for f in "$log" $([ "$N" = 8 ] && echo "data/logs/orbit-$INST-$d.log"); do
    grep -q "^/data/runs.*/orbit$N/" "$f" 2>/dev/null && return
  done
  docker compose run --rm -T -e KREUTZ_INSTRUMENT="$INST" kreutz kreutz.run "$d" --orbit --n "$N" > "$log" 2>&1
  echo "$(date -u '+%F %T') $INST orbit$N $d: $(tail -n 1 "$log")"
}
export -f day; export N INST
d=$FIRST
while [[ ! "$d" > "$LAST" ]]; do echo "$d"; d=$(date -d "$d + 1 day" +%F); done | xargs -P "$JOBS" -I{} bash -c 'day {}'
echo "$(date -u '+%F %T') $INST orbit$N $FIRST..$LAST done"
