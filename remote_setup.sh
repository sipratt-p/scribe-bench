#!/bin/bash
# Bootstrap a fresh CUDA pod for scribe-bench (run on the pod, from /workspace).
# Installs uv, a venv with NeMo + Transformers + MOSS-TD + vLLM, clones NeMo for the streaming script.
set -euo pipefail
cd /workspace
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq sox libsndfile1 ffmpeg git git-lfs rsync tmux > /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p scribe-bench && cd scribe-bench
uv venv --python 3.12 .venv -q
source .venv/bin/activate
uv pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "nemo_toolkit[asr]" "transformers>=5.6" accelerate soundfile librosa jiwer pyannote.metrics \
  huggingface_hub hf_transfer textgrid wordfreq openai trl peft datasets
[ -d moss-td ] || git clone -q --depth 1 https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git moss-td
uv pip install -q -e ./moss-td
[ -d nemo-src ] || git clone -q --depth 1 https://github.com/NVIDIA/NeMo.git nemo-src
uv pip install -q vllm
python -c "import torch, nemo, transformers, vllm; print('torch', torch.__version__, torch.cuda.device_count(), 'GPUs; nemo', nemo.__version__, 'vllm', vllm.__version__)"
echo "setup done"
