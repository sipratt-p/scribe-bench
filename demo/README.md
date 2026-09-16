# Scribe Bench Live

Two ambient-scribe pipelines on the same recording, side by side, with the delta a clinician would feel, plus a
live microphone copilot that keeps a decision-support view current while a consultation is happening.

- **Vanilla** (internal key `theirs`): unoptimized streaming ASR (Nemotron 3.5 0.6B) → note. No speakers, no citations, no verifier.
- **Proposed**: second pass after the visit (MOSS-TD 0.9B, diarized) → note with span citations → verifier.

## Start (cached mode, no GPU needed)

```bash
cd scribe-bench
uv run python demo/build_data.py            # rebuild demo/data from runs/ (only after new runs)
uv run uvicorn demo.server:app --host 127.0.0.1 --port 8700
open http://127.0.0.1:8700
```

Pick a consultation, press **Show cached result**. Everything shown is a real output from the 11 Sep 2026 run.

## Live mode (any audio, about 2 to 3 minutes per recording)

Needs a GPU host reachable over ssh with at least 12 GB free on one GPU (`SCRIBE_GPU` selects it), a note model behind an
OpenAI-compatible endpoint (the runs used Qwen3.8-27B via oMLX on :8600) and a judge model on :8500. Set
`SCRIBE_BEAST_HOST` (or `SCRIBE_BEAST_IP`) to the GPU host; it defaults to `localhost`.

```bash
SCRIBE_GPU=0 uv run uvicorn demo.server:app --host 127.0.0.1 --port 8700
```

Press **Run live** with a consultation selected, or choose a file first (wav/m4a/mp3, converted to 16 kHz mono).
Uploads land in `demo/uploads/`.

## Live microphone copilot (`/mic`)

Browser microphone → 16 kHz PCM over a WebSocket → a self-hosted streaming recogniser (Voxtral Mini 4B Realtime behind a
vLLM realtime endpoint) → timestamped transcript → the clinical view from `demo/livesynth.py`, rebuilt every interval or
when a turn ends: differential with evidence, the next question to ask, tiered red flags, plan stated vs suggested,
safety-netting, assumptions under test, terms heard. Guideline lookups (NHS, patient.info through a local SearXNG) attach
to the next tick. Possible mishearings of medical terms are flagged from the lexicon.

Conversation types are `Pack`s in `demo/packs.py` (system prompt, panel layout, hold rules, optional context documents);
the repository ships the clinical pack. The page renders any pack from its panel description, so a new domain needs no UI code.

Environment:

| variable | meaning | default |
|---|---|---|
| `SCRIBE_ASR_WS` | realtime ASR WebSocket | `ws://<gpu-host>:8090/v1/realtime` |
| `SCRIBE_ASR_MODEL` | served ASR model name | `voxtral-realtime` |
| `SCRIBE_BEAST_HOST` | GPU host for the writer models in `demo/livesynth.py` (`MODELS`) | `localhost` |
| `SCRIBE_SEARX` | SearXNG endpoint for lookups | `http://127.0.0.1:8890` |
| `SCRIBE_LIVE_THINK` | `1` = low-effort thinking in the writer (about 0.5 s per tick) | `0` |

Test without a microphone by streaming a wav at real-time pace:

```bash
uv run python -m autoresearch.mic_client data/primock57/mixed/day1_consultation01.wav --seconds 120 --model gemma_beast --interval 15
```

Sessions are saved under `runs/live_mic/`. The page reminds you to tell everyone present that an assistant is
transcribing before you start; in twelve US states and under GDPR that consent is required.

## Live synthesis page (`/live`, replay of a recording)

Needs a fast local model on the GPU host (the runs used DeepSeek V4 Flash on :8000, fallback Qwen3.8-27B on :8004).
Live-ASR pass 1 runs on CPU automatically while the big model holds the GPUs. Lookups need SearXNG reachable at
`SCRIBE_SEARX` (for a remote host: `ssh -N -L 8890:127.0.0.1:8890 <gpu-host> &`; the page degrades to no lookups without it).
Sessions log to `runs/live_synth/`. Nothing here touches the loop, the results tables, or the two main demo columns.
