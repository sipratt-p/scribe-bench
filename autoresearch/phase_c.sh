#!/usr/bin/env bash
# Phase C on the beast: swap DeepSeek out, Qwen (judge 1) in, score everything Phase B produced plus the second baselines.
set -u
cd ~/projects/scribe-bench
echo "[phase C] $(date) stopping DeepSeek engine"
docker rm -f ds4-flash-dspark >/dev/null 2>&1 || true
sleep 5
docker start scribe-qwen38 >/dev/null 2>&1 && echo "[phase C] scribe-qwen38 starting"
for i in $(seq 1 60); do sleep 10; if curl -s -m 3 localhost:8004/v1/models | grep -q qwen; then echo "[phase C] qwen up after $((i*10)) s"; break; fi; done
curl -s -m 3 localhost:8004/v1/models | grep -q qwen || { echo "[phase C] qwen not up, abort"; exit 1; }
source .venv/bin/activate
echo "[phase C] $(date) running phase_c scoring"
python -m autoresearch.phase_c 2>&1 | grep -v -i warn
echo "[phase C] $(date) DONE_C"
