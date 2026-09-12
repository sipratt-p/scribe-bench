# Datasets

- **PriMock57**: 57 mock UK GP consultations (Babylon), doctor and patient tracks mixed to mono, per-speaker TextGrid timestamps → RTTM for DER, human transcript, clinician note with presenting complaint. Export: `data/primock_export/<id>.json`. Loop split: dev = first 20 sorted ids, test = other 37. References are terse (~136 words), so absolute ROUGE is low by construction.
- **ACI-Bench**: 120 encounters across test1/2/3 with human transcript, raw ASR, human-corrected ASR (`humantrans` / `asr` / `asrcorr`) and a reference note; 22 encounters exist in all three transcript forms. Used for note generation, SFT eval, verifier, fairness (patient gender/age metadata).
- **Lexicon**: 7,092 primary-care terms built from MTS-Dialog + MedSynth (`data/lexicon.txt`), used for term miss / recall / precision and as hotwords.
- **SFT pairs**: 2,804 open transcript→note pairs (ACI train + MTS-Dialog + MedSynth) via `sft_prepare.py`.
- **Synthetic ACI audio**: Kokoro TTS two-voice renderings of ACI `humantrans` dialogues, 16 kHz mono, RTTM + PriMock-shaped JSON (`autoresearch/synth_audio.py`, `runs/synth_aci/`). Clean studio speech: use for trends, not absolute WER.

Sources: `scribe_bench/primock.py`, `acibench.py`, `lexicon.py`, `sft_prepare.py`, `autoresearch/synth_audio.py`.
