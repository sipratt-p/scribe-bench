# Live synthesis during the visit (separate experiment)

**Status: built 12 Sep evening, separate from the loop and the results tables.** Nothing here feeds the composite, the head-to-head, or the demo's two main columns. Code: `demo/livesynth.py`, `demo/live.html`; sessions logged to `runs/live_synth/`.

## What it does
Pretend the doctor is talking to the patient right now, with weights we run locally.
- **Replay on the visit timeline.** A PriMock consultation's audio plays in the browser (1×–8×) while the streaming-ASR transcript (Nemotron-3.5 streaming output, no speaker labels) arrives in ~4-word chunks placed on the human utterance timestamps. This is a replay of real streaming output, not a re-run of the streaming model.
- **Rolling synthesis.** Every N seconds of visit time (default 20) a local model gets the transcript so far plus its previous synthesis and returns JSON: complaint, history so far, findings, plan so far, safety-netting, red flags (max 4), gaps a GP would normally still ask (max 5), terms heard. Runs in a background thread so the transcript never stalls; newly added items are highlighted.
- **Possible mishearings.** Streaming words checked against the 1,333 "medical" lexicon terms (7+ letters, not an English lemma or inflection): a non-word within 2 edits, a real word 1 edit away that is not an inflection, or 2–3 short words that fuse into a term ("met form in" → metformin, "a moxie cillin" → amoxicillin). These are the errors pass 2's LLM decoder exists to fix ([[decoder-finding]]).
- **End of visit.** A final synthesis, then the best after-visit note (the demo's cached loop-best pass-2 note) is shown beside it, its plan items are extracted, and for each one the earliest live snapshot that already contained it is found. Readout: "N of M final plan items were already in a live synthesis before the visit ended", with the first-seen time per item.
- **Model selector**: Qwen3.8-27B dense FP8 (GPU 0) or Qwen3.8-Flash-Next NVFP4 MoE single-GPU (GPU 1, `~/scripts/qwen38fn-ablit-nvfp4-1gpu.sh`, port 8003).

## First run (day1 #01, 16×, Qwen 27B dense, 30 s interval)
7 syntheses. Complaint ("diarrhoea for three days") locked at 30 s. Gaps list evolved sensibly (stool character → hydration → medication history → allergies). Plan stayed empty until the GP stated it near the end, then filled with nine items (gastroenteritis, conservative management, no antibiotics, fluids, Dioralyte, paracetamol, time off work, follow-up in 3–4 days, stool sample if persistent). All 4 final plan items were caught live, first seen at the end because that is when GP plans are said. Latency grew 3.7 → 11.4 s across the visit.

## Measured model speed (beast, single stream, 12 Sep 19:50)
| Model | Prefill | Decode |
|---|---|---|
| Qwen3.8-27B dense FP8, vLLM GPU 0 | 3,780 tok in 0.16 s ≈ 23k tok/s | ≈ 47 tok/s |
| Qwen3.8-Flash-Next NVFP4, 1 GPU (PLE CPU offload) | 3,780 tok in 0.55 s ≈ 6.9k tok/s | ≈ 37 tok/s |
| DeepSeek V4 Flash NVFP4, TP2, DSpark draft, util 0.80 | pending | pending |

Prefill is not the bottleneck; a ~500-token JSON synthesis at 47 tok/s is. The single-GPU Flash-Next config is *slower* than dense Qwen (the PLE offload path is a fit-it-at-all config, not a throughput config), so it was torn down. Seth's fastest local model is DeepSeek V4 Flash on both GPUs with the DSpark draft; the evening of 12 Sep the loop was stopped early (nothing accepted since #47), scribe-qwen38 stopped, and DS Flash launched from `~/projects/dsv4-flash-nvfp4-sm120/fraserprice_nop2p.sh` (non-thinking, no-P2P variant; the P2P variant wedges after weight load on this kernel) with `GPU_MEM_UTIL=0.80` so ~19 GB per GPU stays free for the demo's live ASR pass. Launch needs `hf` on PATH (`~/ml-env/bin`), or the script's `set -e` dies on a pip install.

## Caveats
- The "live" transcript is real streaming output replayed, not streaming inference; timing is proportional word placement over utterance timestamps.
- Mishearing flags are a heuristic (first version flagged "feeling → peeling"; fixed by treating inflections as English). Sparse by design: the streaming model mostly substitutes real words, which no edit-distance check can see.
- The final comparison uses the same local model as a judge; no clinician validation ([[judges]]).

Sources: `demo/livesynth.py`, `demo/live.html`, `runs/live_synth/*.json`, `runs/demo_server.log`.
