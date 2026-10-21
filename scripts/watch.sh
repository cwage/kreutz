#!/bin/bash
# Keep an eye on the running watchers (kreutz.live, one per instrument):
# one line per thing worth a look, nothing for quiet windows.
#   - from their logs: alerts, confirmations, rejections, report matches,
#     vetoes (a chain decided as a star, off-lane, ...: once each, where
#     the chain itself goes on for hours), errors, a container that stops
#   - every 5 min: a status.json more than 45 min old (a stalled watcher,
#     or a feed that stays down; single feed errors are left out)
#   - every 30 min: kreutz.audit for each instrument (missed, vetoed and
#     lost reported comets; alerts that came to nothing)
# Runs until killed; meant for a terminal or a monitor that reads its lines.
cd "$(dirname "$0")/.."
follow() {  # $1 label, $2 container
  docker logs -f --since 1m "$2" 2>&1 | grep -v --line-buffered -E 'Warning|warnings.warn|feed error' \
    | grep -E --line-buffered 'ALERT|CONFIRMED|REJECTED|by position|track at .* is |Traceback|Error|error|failed|skipped' \
    | sed -u "s/^/[$1] /"
  echo "[$1] log stream ended: container $2 stopped?"
}
declare -A box  # instrument key -> watcher container
for n in $(docker ps --format '{{.Names}}'); do
  docker inspect -f '{{join .Args " "}}' "$n" | grep -q 'kreutz.live' || continue
  k=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$n" | sed -n 's/^KREUTZ_INSTRUMENT=//p')
  box[${k:-ccor1}]=$n
done
labels=$(python3 -c "import json; print(' '.join(f\"{i['key']}:{i['label']}:{i['live']}\" for i in json.load(open('data/instruments.json'))))")
for e in $labels; do
  IFS=: read -r key label dir <<< "$e"
  if [ -n "${box[$key]}" ]; then follow "$label" "${box[$key]}" &
  elif [ -d "data/$dir" ]; then echo "[$label] no watcher container running"; fi
done
i=0
while true; do
  for e in $labels; do
    IFS=: read -r key label dir <<< "$e"
    [ -n "${box[$key]}" ] || continue
    age=$(python3 -c "import json,datetime as D;s=json.load(open('data/$dir/status.json'));print(int((D.datetime.utcnow()-D.datetime.fromisoformat(s['updated_utc'])).total_seconds()//60))" 2>/dev/null || echo unreadable)
    if [ "$age" = unreadable ] || [ "$age" -gt 45 ]; then echo "[$label] status.json not updated for $age min"; fi
    if [ $((i % 6)) -eq 0 ]; then
      docker compose run --rm -T -e KREUTZ_INSTRUMENT="$key" kreutz kreutz.audit 2>&1 \
        | grep -v -E 'Warning|warnings.warn|Container .* (Creat|Start)'
    fi
  done
  i=$((i + 1))
  sleep 300
done
