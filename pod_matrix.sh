#!/bin/bash
# Run the NVIDIA-vs-open matrix on a 2-GPU pod. Run from /workspace/scribe-bench after remote_setup.sh
# and model downloads. GPU 0: note-model track. GPU 1: ASR track. Each track is a tmux session.
set -uo pipefail
cd /workspace/scribe-bench
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
mkdir -p runs logs
QWEN=$(.venv/bin/hf download Qwen/Qwen3.8-27B 2>/dev/null | tail -n1 | awk '{print $NF}')
NANO=$(.venv/bin/hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 2>/dev/null | tail -n1 | awk '{print $NF}')
OMNI=$(.venv/bin/hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 2>/dev/null | tail -n1 | awk '{print $NF}')
ACI=data/aci-bench

echo "QWEN=$QWEN"; echo "NANO=$NANO"; echo "OMNI=$OMNI"; for d in "$QWEN" "$NANO" "$OMNI"; do [ -d "$d" ] || { echo "missing model dir $d"; exit 1; }; done

# ---------------- GPU 0: note models ----------------
cat > logs/gpu0.sh <<EOF
set -x
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
# 1. LoRA SFT of Qwen3.8-27B, then merge
.venv-nemo/bin/python -m scribe_bench.sft_train --model $QWEN --data data/sft --out runs/lora_qwen38 --epochs 1 --max_len 3072 --bs 4 --grad_acc 4 --merge
# 2. serve merged model, generate ACI-Bench notes (plain + cite), all splits
.venv/bin/vllm serve runs/lora_qwen38/merged --served-model-name qwen3.8-27b-sft --port 8020 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.85 > logs/vllm_sft.log 2>&1 &
VP=\$!
until curl -s -m 3 localhost:8020/v1/models | grep -q '"id"'; do sleep 10; done
for s in test1 test2 test3; do
  .venv-nemo/bin/python -m scribe_bench.notegen aci $ACI runs/notes_qwen38_sft --base_url http://localhost:8020/v1 --model qwen3.8-27b-sft --split \$s --variants humantrans,asr,asrcorr --workers 32
  .venv-nemo/bin/python -m scribe_bench.notegen aci $ACI runs/notes_qwen38_sft_cite --base_url http://localhost:8020/v1 --model qwen3.8-27b-sft --split \$s --variants humantrans --workers 32 --cite
done
kill \$VP; sleep 15
# 3. Nemotron 3 Nano 30B-A3B base as the note model (reasoning off)
.venv/bin/vllm serve $NANO --served-model-name nemotron-nano --port 8021 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.85 --trust-remote-code > logs/vllm_nano.log 2>&1 &
VP=\$!
until curl -s -m 3 localhost:8021/v1/models | grep -q '"id"'; do sleep 10; done
for s in test1 test2 test3; do
  .venv-nemo/bin/python -m scribe_bench.notegen aci $ACI runs/notes_nemotron_nano --base_url http://localhost:8021/v1 --model nemotron-nano --split \$s --variants humantrans,asr,asrcorr --workers 32
  .venv-nemo/bin/python -m scribe_bench.notegen aci $ACI runs/notes_nemotron_nano_cite --base_url http://localhost:8021/v1 --model nemotron-nano --split \$s --variants humantrans --workers 32 --cite
done
kill \$VP; sleep 15
# 4. LoRA SFT of Nemotron Nano on the same pairs (attention + MLP projections; MoE experts untouched)
.venv-nemo/bin/python -m scribe_bench.sft_train --model $NANO --data data/sft --out runs/lora_nano --epochs 1 --max_len 3072 --bs 4 --grad_acc 4 --merge --targets all-linear
.venv/bin/vllm serve runs/lora_nano/merged --served-model-name nemotron-nano-sft --port 8022 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.85 --trust-remote-code > logs/vllm_nano_sft.log 2>&1 &
VP=\$!
until curl -s -m 3 localhost:8022/v1/models | grep -q '"id"'; do sleep 10; done
for s in test1 test2 test3; do
  .venv-nemo/bin/python -m scribe_bench.notegen aci $ACI runs/notes_nemotron_nano_sft --base_url http://localhost:8022/v1 --model nemotron-nano-sft --split \$s --variants humantrans,asr,asrcorr --workers 32
done
kill \$VP
echo GPU0_DONE
EOF

# ---------------- GPU 1: ASR ----------------
cat > logs/gpu1.sh <<EOF
set -x
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
# 1. NVIDIA offline checkpoints
.venv-nemo/bin/python -m scribe_bench.asr_nemo data/primock_export runs/parakeet_v3 --model nvidia/parakeet-tdt-0.6b-v3 --batch 4
.venv-nemo/bin/python -m scribe_bench.asr_nemo data/primock_export runs/canary_qwen --model nvidia/canary-qwen-2.5b --batch 2
# 2. Sortformer + Parakeet two-model pipeline (NVIDIA pass-2 equivalent)
.venv-nemo/bin/python -m scribe_bench.diar_sortformer data/primock_export runs/sortformer_parakeet --asr nvidia/parakeet-tdt-0.6b-v3 --diar nvidia/diar_sortformer_4spk-v1
# 3. Nemotron 3 Nano Omni end-to-end (audio in, diarized transcript out)
.venv/bin/vllm serve $OMNI --served-model-name nano-omni --port 8010 --max-model-len 65536 --max-num-seqs 4 --gpu-memory-utilization 0.85 --trust-remote-code --limit-mm-per-prompt '{"audio":1}' > logs/vllm_omni.log 2>&1 &
VP=\$!
until curl -s -m 3 localhost:8010/v1/models | grep -q '"id"'; do sleep 10; done
.venv-nemo/bin/python -m scribe_bench.asr_omni data/primock_export runs/nano_omni --base_url http://localhost:8010/v1 --model nano-omni --chunk_s 600
kill \$VP
# 4. score everything
.venv-nemo/bin/python -m scribe_bench.asr_score data/primock_export --lexicon data/lexicon.txt --moss runs/sortformer_parakeet --moss runs/nano_omni --nemo_manifest runs/parakeet_v3/pred.json --nemo_manifest runs/canary_qwen/pred.json --out runs/asr_score_nvidia.json
echo GPU1_DONE
EOF

tmux kill-session -t gpu0 2>/dev/null; tmux kill-session -t gpu1 2>/dev/null
tmux new-session -d -s gpu0 "bash logs/gpu0.sh > logs/gpu0.log 2>&1"
tmux new-session -d -s gpu1 "CUDA_VISIBLE_DEVICES=1 bash logs/gpu1.sh > logs/gpu1.log 2>&1"
echo "launched tmux sessions gpu0, gpu1"
