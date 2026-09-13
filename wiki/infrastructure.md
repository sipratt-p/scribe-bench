# Infrastructure

## beast (2× RTX PRO 6000 Blackwell, 96 GB each, 249 GB RAM)
- `~/projects/scribe-bench` mirror of the repo; `.venv` (uv; torch 2.14 cu130, transformers 5.17, peft 0.20, fla 0.5.2).
- **State from 12 Sep 20:00**: both GPUs serve DeepSeek V4 Flash (`ds4-flash-dspark`, :8000, TP2, util 0.80, non-thinking no-P2P script; per-request `chat_template_kwargs: {"thinking": false}`). `scribe-qwen38` (Qwen3.8-27B-FP8 vLLM :8004, GPU0) is *stopped*, restart with `docker start scribe-qwen38` after stopping DS Flash; it is what the loop, verif_eval, plan_gap and expert_iter judging expect. Flash-Next single-GPU (:8003) measured slower than dense Qwen and was removed.
- Earlier layout: GPU0 scribe-qwen38 (writer, judge 1, role map); GPU1 MOSS-TD variants, NeMo runs, RL policy (`expert_iter`).
- H3 Studio (:8300) auto-restores the DS engine when its render queue drains unless launched with `H3_NO_ENGINE_RESTORE=1`; that is what relaunched a hung `ds4-flash-dspark` at 19:22 on 12 Sep while GPU0 was busy. `~/scripts/free-gpus.sh --stop` frees everything.
- Models under `~/models` (== `~/Models`; HF snapshots are symlinks, use `stat -L`).
- Logs: `runs/autoresearch_loop.log`, `runs/expert_iter.log`, `runs/plan_recall.log`, `runs/verif_eval.log`.
- RunPod API key: `~/.runpod/api_key`.

## Mac Studio (cosmo, 100.90.251.52)
- oMLX :8600 (`v4-flash` DeepSeek V4 Flash 4-bit; `EigenLabs--Qwen3.8-27B-4bit` for the demo); llama-server gemma4-vision :8500; Nemotron Ultra in-process via mlx_lm.
- Demo server: FastAPI on :8700 (`SCRIBE_GPU=1 uv run uvicorn demo.server:app`), live mode ssh-runs `live_asr.py` on the beast (needs ≥12 GB free on a beast GPU).

## Cloud (ask before spending)
- RunPod balance $5.81; Vast and Hyperstack negative. Total spent ~$49: 2×H200 (~$36) for the Qwen SFT + NVIDIA matrix, 1×H200 (~$13) for the Nano LoRA.
- Pod lessons: cu130 wheels vs driver 570 → cuda-compat + separate cu128 venv + `--enforce-eager VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN`; HF cache to /workspace; NeMo 26.04 container; manifests need absolute paths.

## Gotchas collected
Sortformer+Parakeet same-process CUDA illegal address → 3 phases; Omni one-line output → finditer parser; Nemotron Nano NaN on Blackwell (fallback and NeMo container) → cloud; Flash-Next NVFP4 95 GiB alloc → needs `VLLM_PLE_CPU_OFFLOAD=1` script; mlx_lm.server wedged → in-process; Gemma judge saturation → beast judges + retries; `pkill` self-matching ssh sessions → anchored patterns; escaped quotes inside ssh heredocs → write a script file and scp it; uv venv has no pip → `uv pip install --python .venv/bin/python`.

Sources: memory `project_scribe_bench.md`, `reference_gpu_providers.md`, `feedback_local_first_compute.md`; `remote_setup.sh`, `pod_matrix.sh`, `demo/README.md`.
