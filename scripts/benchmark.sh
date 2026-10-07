#!/bin/bash
# The July 17-24 2025 benchmark (eight confirmed CCOR-1 comets) at window
# lengths 4 and 8, then the evaluation. Frames and reports first:
#   docker compose run --rm kreutz kreutz.fetch 2025-07-16 2025-07-24
#   docker compose run --rm kreutz kreutz.reports 2025-07-01 2025-08-31
cd "$(dirname "$0")/.."
for d in 2025-07-17 2025-07-18 2025-07-19 2025-07-20 2025-07-21 2025-07-22 2025-07-23 2025-07-24; do
  scripts/day.sh $d 4
  scripts/day.sh $d 8
done
docker compose run --rm kreutz kreutz.evaluate /data/runs/n8/202507*.jsonl
