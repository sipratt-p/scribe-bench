# Vanilla vs best: the head-to-head

Same 57 PriMock consultations, same scorer. **Vanilla** = Nemotron streaming transcript (no speakers) → untuned Qwen3.8-27B, plain prompt. **v1** = MOSS-TD diarized transcript, anonymous labels, same writer. **Best** = MOSS-TD + complaint hotwords, LLM role map, precision-first extra, DeepSeek V4 Flash writer (loop iteration 47). **Best cited** = same with double-bracket line citations required.

## Transcript layer (57)
| | Vanilla | Best |
|---|---|---|
| WER | 11.8 | 10.6 |
| Medical-term miss | 12.7 | 8.6 |
| DER | no speakers | 11.7 |

## Note layer, test (37)
| | Vanilla | v1 | Best uncited | Best cited |
|---|---|---|---|---|
| Composite, judge 1 (Qwen) | 40.82 | 39.62 | **45.01** | 44.10 |
| Composite + 0.2 grounded | 58.54 | 57.58 | 62.59 | **62.74** |
| Composite, judge 2 (DeepSeek) | 44.36 | – | 46.80 | – |
| Term recall | 52.09 | 54.35 | 52.70 | 51.90 |
| Term precision | 26.41 | 25.78 | 30.90 | **32.91** |
| ROUGE-L | 21.75 | 20.60 | **23.91** | 22.68 |
| Plan recall, all reference items | 80.79 | 78.95 | **81.17** | 75.97 |
| Plan recall, transcript-stated only | **96.5** | – | 92.9 | 89.4 |
| Misattributions per note | 0.08 | 0.11 | **0.03** | **0.03** |
| Grounded sentences | 88.62 | 89.76 | 87.87 | **93.21** |
| Linked to cited lines | 0 | 0 | 0 | **83.95** |

## Dev (20)
Composite: 44.17 / 44.26 / **48.44** / 44.48. Grounded: 92.45 / 92.94 / 93.46 / **96.03**.

## Reading
- Vanilla and v1 tie: diarization alone did nothing for the note ([[role-mapping]]).
- The optimized pipeline wins on term precision, attribution, ROUGE-L and (cited) grounding/checkability. **Vanilla wins on completeness** of spoken plan items ([[plan-recall-gap]]).
- The margin over vanilla is **2.4 points under judge 2 and 4.2 under judge 1**. Quote the range. DeepSeek is both writer and judge 2 for the best config, so judge 1's number is the safer one.
- Term recall barely moves anywhere: the gain is fewer invented terms and fewer wrong-speaker statements, not more captured terms.
- Paired bootstrap over the 37 test files (`runs/paper_stats.md`): composite best − vanilla +4.2 [−0.3, +8.8], not resolved; term precision +4.5 [+2.3, +6.7] and ROUGE-L +2.2 [+1.0, +3.2] resolved; cited − best grounding +5.3 [+3.2, +7.8] resolved, plan recall −6.5 [−13.3, +0.2] not.

Sources: `runs/verif_eval.md`, `autoresearch/state.json`, `autoresearch/notebook.md` iterations 1 and 47, `/tmp/theirs_eval.py` and `/tmp/vanilla_extra.py` outputs (12 Sep), `runs/asr_score*.json`.
