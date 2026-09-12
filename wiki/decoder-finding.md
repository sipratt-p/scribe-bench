# The decoder decides medical-term loss, not latency

**Tag: transferred** (held on PriMock57 and on the synthetic ACI corpus; four checkpoints; no score was tuned to find it).

PriMock57, 57 files, WER after symmetric normalization, medical-term miss = share of the 1,741 lexicon tokens in the human transcript that the hypothesis lost, DER with 250 ms collar:

| System | Decoder | WER | Term miss | DER |
|---|---|---|---|---|
| Nemotron-3.5 streaming, 1.1 s chunks | TDT/RNN-T | 11.8 | 12.7 | no speakers |
| Nemotron-3.5 same model, offline | TDT/RNN-T | 11.8 | 12.5 | no speakers |
| Parakeet-TDT 0.6B v3 | TDT | 11.2 | 15.1 | no speakers |
| Sortformer + Parakeet v3 | TDT | 11.2 | 15.1 | 11.1 |
| Canary-Qwen 2.5B | LLM | 11.6 | 9.4–9.6 | no speakers |
| MOSS-TD 0.9B plain | LLM | 10.3 | 8.4 | 11.4 |
| MOSS-TD + 150 hotwords | LLM | 10.4 | 8.2 | 11.4 |
| MOSS-TD + complaint hotwords (loop's pick) | LLM | 10.6 | 8.6 | 11.7 |
| Nano Omni 30B-A3B, chat prompt | LLM (as prompted) | 27.2 | 15.7 | 88.3 |

Synthetic ACI-Bench audio (Kokoro TTS, clean studio speech): MOSS-TD WER 4.2 / term miss 6.6 / DER 10.7 vs Nemotron streaming 5.1 / 8.9. Same ordering.

Reading
- WER is within a point across seven systems; term miss differs by nearly 2×. WER hides the thing the doctor feels.
- Running the streaming model offline changes nothing → the gain of pass 2 is the decoder, not the extra time.
- Hotwords do not move the transcript (±0.2). The loop's preference for complaint-tuned hotwords is a note-score artifact ([[autoresearch-loop]]).
- Word-level speaker misattribution at the transcript level: MOSS-TD 1.0%, Sortformer+Parakeet 1.3%, Omni 32%.

Sources: `RESULTS.md` (ASR tables), `runs/asr_score.json`, `runs/asr_score_nvidia.json`, `runs/asr_score_variants.json`, `runs/asr_score_synth.json`. Metric definitions in [[metrics]].
