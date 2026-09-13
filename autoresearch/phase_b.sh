#!/usr/bin/env bash
# Phase B on the beast: swap the abliterated DeepSeek for the official DSpark build, regenerate the DeepSeek-dependent outputs.
# Usage (on the beast): bash ~/projects/scribe-bench/autoresearch/phase_b.sh
set -u
cd ~/projects/scribe-bench
echo "[phase B] $(date) stopping abliterated engine"
docker rm -f ds4-flash-dspark >/dev/null 2>&1 || true
sleep 5
cd ~/projects/dsv4-flash-nvfp4-sm120
PATH=$HOME/ml-env/bin:$PATH MODEL_DIR=/mnt/models/DeepSeek-V4-Flash-DSpark SERVED_NAME=DeepSeek-V4-Flash-DSpark GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=32768 nohup ./fraserprice_nop2p_scribe.sh > ~/ds4-flash-official.log 2>&1 < /dev/null &
echo "[phase B] launching official engine"
for i in $(seq 1 90); do sleep 20; if curl -s -m 3 localhost:8000/v1/models | grep -q DSpark; then echo "[phase B] official engine up after $((i*20)) s"; break; fi; if ! docker ps --format '{{.Names}}' | grep -q ds4-flash; then echo "[phase B] ENGINE DIED"; tail -n 5 ~/ds4-flash-official.log; exit 1; fi; done
curl -s -m 3 localhost:8000/v1/models | grep -q DSpark || { echo "[phase B] engine not up, abort"; exit 1; }
cd ~/projects/scribe-bench && source .venv/bin/activate
echo "[phase B] $(date) running official_rerun"
python -m autoresearch.official_rerun 2>&1 | grep -v -i warn
echo "[phase B] $(date) DONE_B"
