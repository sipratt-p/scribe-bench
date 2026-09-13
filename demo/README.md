# Scribe Bench Live (interview demo)

Two ambient-scribe pipelines on the same recording, side by side, with the delta a clinician would feel.

- **Vanilla** (internal key `theirs`): unoptimized streaming ASR (Nemotron 3.5 0.6B) → note. No speakers, no citations, no verifier.
- **Proposed**: second pass after the visit (MOSS-TD 0.9B, diarized) → note with span citations → verifier.

## Start (cached mode, no GPU needed)

```bash
cd ~/src/scribe-bench
uv run python demo/build_data.py            # rebuild demo/data from runs/ (only after new runs)
uv run uvicorn demo.server:app --host 127.0.0.1 --port 8700
open http://127.0.0.1:8700
```

Pick a consultation, press **Show cached result**. Everything shown is a real output from the 11 Sep 2026 run.

## Live mode (any audio, ~2–3 min per recording)

Needs: the beast reachable as `beast` over ssh with ≥12 GB free on one GPU (stop `ds4-flash-dspark`
or set `SCRIBE_GPU=1`), the oMLX server on :8600 (Qwen3.8-27B-4bit for notes), the Gemma server on :8500 (judge).

```bash
SCRIBE_GPU=0 uv run uvicorn demo.server:app --host 127.0.0.1 --port 8700
```

Press **Run live** with a consultation selected, or choose a file first (wav/m4a/mp3, converted to 16 kHz mono).
Uploads land in `demo/uploads/`.

## Pre-demo checklist (Sunday)

1. `docker stop ds4-flash-dspark` on the beast (or point `SCRIBE_GPU` at a free GPU); `nvidia-smi` shows ≥12 GB free.
2. `ssh beast 'cd ~/projects/scribe-bench && source .venv/bin/activate && python -m scribe_bench.live_asr data/primock57/mixed/day1_consultation02.wav --out /tmp/t.json'` finishes in ~60 s.
3. `curl localhost:8600/v1/models` lists `EigenLabs--Qwen3.8-27B-4bit`; `curl localhost:8500/v1/models` lists `gemma4-vision`.
4. Start the server, load a cached consultation, then run one live to warm the models.
5. Good consultations to show: day1_consultation01 (diarrhoea; drug names), day1_consultation02 (skin; speaker swap in Omni row), any with a third-party history.

## Suggested narrative (10 minutes)

1. Play 20 s of audio. Two speakers, one drug name.
2. Their column: streaming transcript, red = medical terms the streaming decoder lost. Note written from it.
3. Proposed column: speaker turns, cited note, verifier flags. Point at one UNSUPPORTED or THIRD-PARTY claim.
4. Delta table: same recording, what changed for the doctor.
5. Learnings: decoder not latency; citations cheap; fine-tuning wins ROUGE and loses attribution; scale buys safety not fidelity; build evals from the edit stream.

## Live synthesis page (`/live`, separate experiment)
- Needs a fast local model: DeepSeek V4 Flash on the beast :8000 (`cd ~/projects/dsv4-flash-nvfp4-sm120 && PATH=$HOME/ml-env/bin:$PATH MODEL_DIR=$HOME/models/DeepSeek-V4-Flash-Abliterated SERVED_NAME=DeepSeek-V4-Flash-Abliterated GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=32768 nohup ./fraserprice_nop2p_scribe.sh > ~/ds4-flash-scribe.log 2>&1 &`; ~4 min; check `curl -s http://100.83.231.108:8000/v1/models`). Live-ASR pass 1 runs on CPU automatically while DS Flash holds the GPUs (~70 s for a 9-min file).. Fallback: `docker start scribe-qwen38` (:8004) and pick the Qwen model in the page.
- Lookups need the beast's SearXNG tunnelled: `ssh -N -L 8890:127.0.0.1:8890 beast &` (the page degrades to no lookups without it).
- Sessions log to `runs/live_synth/`. Nothing here touches the loop, the results tables, or the two main demo columns.
