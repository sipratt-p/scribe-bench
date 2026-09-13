# Models

## ASR
| Model | Role | Where run | Notes |
|---|---|---|---|
| Nemotron-3.5-ASR-Streaming 0.6B | pass 1 (vanilla transcript) | beast, NeMo | 11.8 WER / 12.7 term miss; offline identical; deletion-heavy |
| Parakeet-TDT 0.6B v3 | NVIDIA offline | beast | 11.2 / 15.1 |
| Sortformer diarization + Parakeet v3 | NVIDIA two-model pass 2 | beast, 3-phase script (same-process CUDA illegal address) | DER 11.1, 1.3% misattributed words |
| Canary-Qwen 2.5B (SALM) | NVIDIA LLM decoder | beast | 11.6 / 9.4–9.6; rejected by the loop as pass 2 (dev 42.8) |
| MOSS-Transcribe-Diarize 0.9B | open pass 2, one pass | beast GPU1 | 10.3 / 8.4 / DER 11.4; hotword variants via `热词提示` |
| Nemotron 3 Nano Omni 30B-A3B | NVIDIA end-to-end pass 2 | beast vLLM, eager | 27.2 WER, invented timestamps, DER 88 — unusable as prompted |
| Kokoro TTS | synthetic ACI audio | beast | am_michael / af_heart voices |

## Note writers / judges
| Model | Serving | Role | Notes |
|---|---|---|---|
| Qwen3.8-27B (FP8 on vLLM :8004; bf16 in `~/models` for HF) | beast GPU0 | default writer, judge 1, RL policy | dense; base ROUGE-L 34.2 ACI |
| Qwen3.8-27B + LoRA (2,804 pairs, 1 epoch, H200) | pod | SFT study | 43.2 ROUGE-L, attribution cost |
| DeepSeek V4 Flash (`v4-flash`, oMLX :8600) | Mac Studio | loop's best writer, judge 2 | MoE; base 35.8 / TP 70.3 / mis 0.06 / plan 93 on ACI; verdict-last prompt needed |
| Qwen3.8-Flash-Next (ABLITERATED NVFP4, `~/scripts/qwen38fn-ablit-nvfp4-1gpu.sh`, :8003) | beast GPU1 | tried as writer | dropped: 95 GiB alloc without PLE offload; oMLX 0.6.3 cache crashes |
| Nemotron 3 Nano 30B-A3B | pod H200 (NaN on Blackwell locally) | base + small LoRA | 27.3 base / 38.5 tuned (24 notes) |
| Nemotron 3 Ultra 550B-A55B, 4-bit | Mac Studio, in-process mlx_lm (~12 tok/s) | base | 33.1; cleanest attribution of any base |
| Gemma 4 26B-A4B (llama-server :8500) | Mac | original judge, demo live judge | saturates; not trusted |

Published leaderboard context (not ours): ACI+PriMock comparison (Jul 2026, 0–1 composite): GPT-5.4 reasoning-off 0.515, DeepSeek V4 Flash 0.506, Gemma 4 E4B 0.498; MedBench v5 record generation (0–100): Opus 4.7 81.0, GPT-5.5 79.7, Kimi K2.6 79.1, GLM-5.1 77.6, DeepSeek V4 Pro 73.0, MedGemma 1.5 59.1.


Sources: `RESULTS.md`, `runs/asr_score*.json`, artifact section 5, [[fine-tuning-findings]].
