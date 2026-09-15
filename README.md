# scribe-bench

> Open-data, open-weight evaluation harness for ambient clinical documentation (speech-to-note): seven ASR systems, note generation, LoRA fine-tuning, a claim-level verifier, and clinician-defined scorers. All numbers in RESULTS.md are reproducible from the commands below. Endpoints are configured with SCRIBE_NOTE_URL, SCRIBE_JUDGE_URL and SCRIBE_MAC_URL.


Local evaluation harness for the ambient-scribe "frontier stack" described in the
ambient-scribe architecture study. Everything runs on open data and open weights:

| Step | What it measures | Data | Where it runs |
|---|---|---|---|
| 1 ASR | WER, medical-term error, DER for pass-1 streaming (Nemotron-3.5 0.6B) vs pass-2 diarized (MOSS-Transcribe-Diarize 0.9B), with and without lexicon hotwords | PriMock57 (57 mixed-channel consultations) | beast, GPU 0 |
| 2 Note generation | ROUGE / BERTScore of Qwen3.8-27B notes from human, ASR, and corrected-ASR transcripts; the gap is the ceiling on what pass 2 buys | ACI-Bench test1-3 (120 encounters) | beast vLLM :8004 |
| 3 SFT | LoRA fine-tune of the note model on transcript-note pairs, held out on ACI-Bench test | ACI-Bench train/valid, MTS-Dialog, MedSynth | beast or Mac (mlx_lm) |
| 4 Verifier | Recall of a claim-level judge on injected hallucinations (drug swap, negation flip, fabricated finding, third-party leak) and flag rate on clean claims | claims extracted from cited notes | Mac Gemma 4 :8500 |

Results from the 11 Sep 2026 run are in `RESULTS.md`. Additional scorers modelled on clinician-defined dimensions from the vendor evaluation literature: word-level speaker misattribution (`asr_score.py`), attribution and follow-up completeness LLM judges
(`note_judge.py`), and metadata stratification for fairness (`note_score.py --by patient_gender|age_band`).
NVIDIA comparison runners: `asr_nemo.py` (Parakeet), `asr_canary_qwen.py`, `diar_sortformer.py`, `asr_omni.py`;
`pod_matrix.sh` runs the whole matrix on a 2-GPU pod after `remote_setup.sh`.

## Layout

```
scribe_bench/
  primock.py       TextGrids -> utterances, reference text, RTTM, notes
  acibench.py      ACI-Bench splits with humantrans / asr / asrcorr variants
  lexicon.py       medical lexicon from MTS-Dialog + MedSynth (no PriMock leakage)
  textnorm.py      symmetric WER normalization (fillers, UK/US spelling, digits)
  asr_moss.py      pass-2: MOSS-TD over full recordings, optional hotword prompt
  asr_nemotron.py  pass-1: NeMo cache-aware streaming sim + offline reference
  asr_score.py     WER, med-term error rate, DER
  notegen.py       notes from transcripts via any OpenAI-compatible server, --cite for [[n]] spans
  note_score.py    ROUGE-1/2/L, BERTScore (deberta-xlarge-mnli), citation coverage
  sft_prepare.py   chat-format jsonl for LoRA SFT
  verifier.py      extract claims, inject perturbations, LLM judge, precision/recall
data/              cloned corpora (gitignored) + lexicon.txt
runs/              outputs (gitignored)
```

## Setup

```bash
uv sync
cd data && git clone https://github.com/wyim/aci-bench && git clone https://github.com/abachaa/MTS-Dialog \
  && git clone https://github.com/babylonhealth/primock57 && (cd primock57 && git lfs pull)
# mix doctor+patient tracks
cd primock57 && mkdir mixed && for f in audio/*_doctor.wav; do b=$(basename $f _doctor.wav); sox -m $f audio/${b}_patient.wav mixed/$b.wav; done
uv run python -m scribe_bench.primock data/primock57 data/primock_export
uv run python -m scribe_bench.lexicon data data/lexicon.txt
```

GPU box: `uv venv && uv pip install "nemo_toolkit[asr]" transformers torch soundfile jiwer pyannote.metrics textgrid`
plus `pip install -e` of the MOSS-Transcribe-Diarize repo, and a shallow clone of NeMo for the streaming script.

## Run

```bash
# ASR (GPU box)
python -m scribe_bench.asr_moss data/primock_export runs/moss_plain
python -m scribe_bench.asr_moss data/primock_export runs/moss_hot --hotwords data/lexicon.txt
python -m scribe_bench.asr_nemotron data/primock_export runs/nemotron --nemo_src nemo-src --ctx 70,13
python -m scribe_bench.asr_score data/primock_export --lexicon data/lexicon.txt \
   --moss runs/moss_plain --moss runs/moss_hot --nemo_manifest runs/nemotron/offline.json \
   --nemo_manifest runs/nemotron/streaming_70_13/*.json

# Notes
python -m scribe_bench.notegen aci data/aci-bench runs/notes --base_url http://host:8004/v1 --model qwen3.8-27b --split test1
python -m scribe_bench.note_score aci data/aci-bench runs/notes --split test1

# Verifier
python -m scribe_bench.verifier extract runs/notes_cite/test1/humantrans transcripts.json claims.jsonl
python -m scribe_bench.verifier perturb claims.jsonl perturbed.jsonl --rate 0.25
python -m scribe_bench.verifier judge perturbed.jsonl judged.jsonl --base_url http://localhost:8500/v1 --model gemma4-vision
```

## Wiki

A compiled, Karpathy-style knowledge base of everything measured here lives in [`wiki/index.md`](wiki/index.md). Raw outputs under `runs/` and `autoresearch/` are the sources; the wiki is rewritten from them.
