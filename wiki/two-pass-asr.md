# Two-pass ASR design

The frontier design keeps the live transcript and the final transcript as different models.

- **Pass 1, streaming**: Nemotron-3.5-ASR-Streaming 0.6B (VAD + 1.1 s chunks), for the clinician's live view. Fast, no speakers, drops rare words.
- **Pass 2, after the visit**: one LLM-decoder, speaker-attributed pass over the whole recording. Open pick: MOSS-Transcribe-Diarize 0.9B. NVIDIA-native pick: Canary-Qwen 2.5B + Sortformer diarization. Nano Omni prompted as a transcriber was not usable ([[models]]).
- **Role map**: S01/S02 → Doctor/Patient by an LLM before note writing. This is the step that actually changed the note ([[role-mapping]]).
- **Note model**: open-weight, reasoning off, span citations required ([[citations-and-verifiability]]).
- **Verifier** ahead of the clinician ([[verifier]]); **frontier router** only for coding/orders/chart questions.

Why pass 2 is a *decoder* argument and not a *latency* argument: [[decoder-finding]]. What the July workshop says Abridge actually runs for pass 1 (a third-party provider, evaluating Nemotron Speech via vLLM): [[abridge]].

Sources: artifact sections 1–3 ([[deliverables]]); `scribe_bench/asr_*.py`, `autoresearch/asr_variants.py`.
