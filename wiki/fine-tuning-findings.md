# Fine-tuning findings

**Tag: transferred** (held across three ACI splits and two model families).

ACI-Bench, 3-split mean, notes from the human transcript, Qwen judge for the clinician dimensions:

| Model | ROUGE-L | BERTScore | Term recall | Term precision | Misattrib/note (t1/t2/t3) | Follow-up recall |
|---|---|---|---|---|---|---|
| Qwen3.8-27B base | 34.2 | 70.8 | 66.7 | 66.4 | 0.08 / 0.03 / 0.10 | 96.5 / 97.7 / 97.9 |
| Qwen3.8-27B base + citations | 33.4 (t1) | 70.3 | 64.4 | 70.9 | 0.08 / 0.18 / 0.05 | 92.5 / 97.3 / 95.0 |
| Qwen3.8-27B, 1-epoch LoRA on 2,804 open pairs | **43.2** | 73.6 | 72.5 | 78.8 | 0.23 / 0.13 / 0.18 | 94.6 / 93.1 / 88.4 |
| Nemotron 3 Nano 30B-A3B base | 27.3 | 66.6 | 58.7 | 50.5 | 0.28 / 0.25 / 0.25 | 95.1 / 95.0 / 95.9 |
| Nemotron 3 Nano, small-adapter LoRA on 1,200 pairs (24 notes) | 38.5 | 69.6 | 54.2 | 74.3 | 0.29 | 78.2 |
| Nemotron 3 Ultra 550B-A55B base, 4-bit on a Mac Studio | 33.1 | 69.6 | 63.0 | 58.9 | 0.03 / 0.10 / 0.13 | 96.0 / 98.8 / 96.3 |
| DeepSeek V4 Flash base (PriMock/ACI, 12 Sep) | 35.8 | | | 70.3 | 0.06 | 93 |

Same 24 encounters, ROUGE-L: Qwen base 36.8, Ultra base 35.7, Nano base 31.1, Nano tuned 38.5, Qwen tuned 43.8.

Reading
- **Tuning first, base quality second, size a distant third.** One epoch of LoRA (1 h 53 m on one H200) moves Qwen +9 ROUGE-L, +12 term precision.
- **The tune that wins ROUGE loses on clinician-defined dimensions.** Misattributions double or triple (mostly family history written as the patient's own); follow-up recall drops up to 9 points; the small Nano adapter falls to 78%. Overlap metrics reward fluent, complete-looking notes; clinician-defined dimensions catch what fluency hides → the case for the [[verifier]] and for training on clinician edit deltas.
- **Scale bought safety, not fidelity.** Untuned Ultra has the fewest misattributions of any base model and the worst term precision.
- **Citations must be in the SFT targets** or the tune erases them ([[citations-and-verifiability]]).
- Edit effort (chars to reach the reference per 100 ref words): Qwen base 428, +cite 435, SFT 465, Nano base 468, Ultra 445, Nano tuned 458.

Practical notes: TRL failed under device_map for the hybrid (warmup_ratio, chunked CE, NLL assert) → `scribe_bench/sft_train_min.py`; PEFT saved keys without `.default.` → explicit remap in `hf_notegen.py --adapter`; PEFT cannot touch Mamba output projections in Nano; Nano produced NaN on Blackwell locally and trained only on the H200 ([[infrastructure]]).

Sources: `RESULTS.md` note tables, `runs/effort_notes_*.json`, `runs/sft_*.log`, artifact section 6.
