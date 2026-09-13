# Citations and verifiability

**Tag: transferred** (citation precision gain on ACI test1/2/3 and PriMock; grounding halving on PriMock held-out).

## What citations do
Requiring every note sentence to end with double-bracket line citations to the numbered transcript:
- ACI-Bench (Qwen 27B): ~−1 ROUGE-L, +4–5 term precision on every split. Points at evidence → invents fewer terms.
- PriMock held-out (DeepSeek V4 Flash, loop best config): term precision 30.9 → 32.9; **unsupported sentences 11.4% → 6.8%**; 88.8% of sentences cited; **84.0%** of sentences supported by the exact lines they cite. Cost: plan recall 81.2 → 76.0, ROUGE-L 23.9 → 22.7, term recall 52.7 → 51.9 (dev showed −8 recall; test −0.8, so the dev drop was partly small-sample).
- Fine-tuning erases citation behaviour unless the targets carry citations (99.6% → 9–16% cited after SFT) ([[fine-tuning-findings]]).

## The loop's mistake
The composite ([[metrics]]) had no verifiability term. Citations only showed up as a recall cost, so the [[autoresearch-loop]] rejected them eight times (e.g. iteration 64: best config + cite, dev 44.48 vs 47.59). A score without the trust dimension removes the trust feature. This is the metric-owner lesson for [[interview-narrative]].

## The re-score (12 Sep)
Two measures added: **grounded** = share of note sentences a judge finds supported by the *human* transcript (whole transcript as evidence, same for cited and uncited); **linked** = share of sentences whose own cited lines support them (uncited notes cannot score). Composite_v = composite + 0.2 × grounded.

Official writer weights (13 Sep evening, the numbers the paper reports): grounded vanilla 88.62 / best uncited 85.20 / best cited 91.33; cited 78.9%, linked 74.3%; cited − uncited grounded **+6.13 [+3.23, +8.91]**, precision +2.14 [+0.03, +4.39], ROUGE-L −0.60 [−1.73, +0.59], plan recall +4.09 [−1.12, +9.05]. The uncited best is now *less* grounded than vanilla (−3.4 [−6.6, −0.2]); the cited one more (+2.7 [+0.4, +5.1]). The table below is the search-time (abliterated) build.

| Test (37), search-time build | Vanilla | Best uncited | Best cited |
|---|---|---|---|
| composite | 40.82 | 45.01 | 44.10 |
| composite_v | 58.54 | 62.59 | 62.74 |
| grounded | 88.62 | 87.87 | 93.21 |
| linked | 0 | 0 | 83.95 |
| cited_frac | 0 | 0 | 88.82 |

Dev (20): composite 44.17 / 48.44 / 44.48; grounded 92.45 / 93.46 / 96.03; linked 0 / 0 / 88.06.

Reading: the uncited best is *no better grounded than vanilla*; its win was precision and attribution. With grounding in the score, cited and uncited tie on test and both lead vanilla by ~4. The cited config is the one to ship; it is the only one the [[verifier]] can check per span. Commercial scribes offer on-demand evidence linking; the vanilla pipeline here does not, so "0 linked" describes the baseline, not any product.

Sources: `runs/verif_eval.md`, `runs/verif_eval.json`, `autoresearch/verif_eval.py`, `autoresearch/notebook.md` (cite rows 6–11, 21, 49, 64, 85, 88), `RESULTS.md` note tables.
