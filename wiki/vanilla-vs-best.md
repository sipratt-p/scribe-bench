# Vanilla vs best: the head-to-head

Same 57 PriMock consultations, same scorer. **Vanilla** = Nemotron streaming transcript (no speakers) → untuned Qwen3.8-27B, plain prompt. **v1** = MOSS-TD diarized transcript, anonymous labels, same writer. **Best** = MOSS-TD + complaint hotwords, LLM role map, precision-first extra, DeepSeek V4 Flash writer (loop iteration 47). **Best cited** = same with double-bracket line citations required.

## Official weights (13 Sep evening): the table that the paper reports
The search ran on a community-modified ("abliterated") build of DeepSeek V4 Flash. Every DeepSeek-dependent number was re-run with the official release weights as writer and as judge 2, and two second baselines were added. Judge 1 = Qwen, test n = 37, paired bootstrap in `runs/paper_stats.md` (script `autoresearch/paper_ci.py`).

| | Vanilla | Parakeet v3 → Qwen | Sortformer + Parakeet → role map → Qwen | v1 | Best (official writer) | Best cited (official writer) |
|---|---|---|---|---|---|---|
| Composite, judge 1 | 41.01 | 40.01 | **43.01** | 39.97 | 42.26 | 42.16 |
| Composite + 0.2 grounded | 58.74 | 57.73 | **61.02** | 57.93 | 59.30 | 60.43 |
| Composite, judge 2 (official DeepSeek) | 45.06 | – | – | – | 44.40 | 43.34 |
| Term recall | 52.09 | 51.56 | 52.88 | **54.35** | 49.58 | 49.78 |
| Term precision | 26.41 | 24.82 | 25.77 | 25.78 | 28.92 | **31.06** |
| ROUGE-L | 21.75 | 21.07 | 21.44 | 20.60 | **23.23** | 22.63 |
| Plan recall, all reference items | 80.8 | 79.2 | **81.7** | 78.9 | 72.7 | 76.8 |
| Plan recall, transcript-stated only | **96.5** | – | – | – | 87.0 | 89.6 |
| Misattributions per note | 0.08 | 0.08 | **0.03** | 0.11 | **0.03** | 0.05 |
| Grounded sentences | 88.62 | 88.58 | 90.05 | 89.76 | 85.20 | **91.33** |
| Cited / linked | 0 | 0 | 0 | 0 | 0 | **78.9 / 74.3** |

Best − vanilla: composite +1.25 [−3.76, +6.23] (judge 2: −0.66 [−4.50, +3.03]); term precision **+2.5 [+0.5, +4.5]** and ROUGE-L **+1.5 [+0.3, +2.6]** resolved; plan recall **−8.1 [−13.8, −1.7]** and grounded **−3.4 [−6.6, −0.2]** resolved *losses*. Cited − vanilla: precision **+4.7 [+2.4, +7.1]**, grounded **+2.7 [+0.4, +5.1]**, composite +1.2 [−3.8, +6.3]. Two-pass Qwen − vanilla: +2.0 [−1.0, +5.7], misattributions −0.05 [−0.14, 0.00], nothing else moves. Roughly three of the four composite points the search reported were the writer build (official − abliterated on the same config, model-free: term recall −3.1 [−6.3, −0.0]).

## Transcript layer (57)
| | Vanilla | Best |
|---|---|---|
| WER | 11.8 | 10.6 |
| Medical-term miss | 12.7 | 8.6 |
| DER | no speakers | 11.7 |

## Note layer, test (37), search-time build of the writer (superseded by the table above)
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
- On the search-time build the margin over vanilla was 2.4 points under judge 2 and 4.2 under judge 1; **on the official weights it is +1.3 under judge 1 and −0.7 under judge 2, neither resolved**, and the Sortformer + Parakeet + role map + untuned Qwen pipeline (no search, no DeepSeek) is numerically the best composite. Quote the official-weight numbers.
- Term recall barely moves anywhere: the gain is fewer invented terms and fewer wrong-speaker statements, not more captured terms.
- Paired bootstrap over the 37 test files (`runs/paper_stats.md`): composite best − vanilla +4.2 [−0.3, +8.8], not resolved; term precision +4.5 [+2.3, +6.7] and ROUGE-L +2.2 [+1.0, +3.2] resolved; cited − best grounding +5.3 [+3.2, +7.8] resolved, plan recall −6.5 [−13.3, +0.2] not.

Sources: `runs/verif_eval.md`, `autoresearch/state.json`, `autoresearch/notebook.md` iterations 1 and 47, `/tmp/theirs_eval.py` and `/tmp/vanilla_extra.py` outputs (12 Sep), `runs/asr_score*.json`.
