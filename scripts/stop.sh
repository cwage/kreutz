#!/bin/bash
# Stop archive/benchmark runs and their containers (not the live watcher).
ps -eo pid,args | grep -E "scripts/([d]ay|[a]rchive(-gpu)?|[l]ane|[b]enchmark)\.sh" | awk '{print $1}' | xargs -r kill
sleep 1
docker ps --no-trunc --format '{{.Names}}|{{.Command}}' | awk -F'|' '$2 ~ /kreutz.run/ {print $1}' | xargs -r docker stop >/dev/null
echo "$(docker ps --no-trunc --format '{{.Command}}' | grep -c kreutz.run) kreutz.run containers still running"
