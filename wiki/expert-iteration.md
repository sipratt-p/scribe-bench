# Expert iteration: the RL exercise

Run started 12 Sep 12:26 on beast GPU 1, queued behind the plan-recall experiment. Predictions were written into the notebook before the first sample.

## Setup
- **Policy**: Qwen3.8-27B bf16 + LoRA (r 16, α 32, q/k/v/o/gate/up/down), prompted exactly like the loop's best *cited* config (MOSS-TD hotcc transcript, LLM role map, precision-first extra, citations required). The only writer we can both train and serve locally.
- **Reward**: composite + 0.2 grounded + 0.1 linked − length penalty outside round-0 mean ± 0.6 ([[metrics]]).
- **Loop**: sample 8 notes per dev transcript (T 0.8, top-p 0.95) → reward all 160 → keep top 2 per transcript → LoRA on the union of everything kept so far (2 epochs, lr 1e-4) → greedy on dev + test, judge 1 and judge 2 → repeat, 4 rounds.
- **Readouts** per round: composite, grounded, linked, misattrib, plan recall, edit effort, len ratio, passive/100w, agentless share, judge gap ([[judges]]).
- Sampling ~2 min/transcript even with `flash-linear-attention` installed; ~75 min per round.

## Predictions (logged first)
1. Round-1 gains are real: +2–3 dev; about half survives on test.
2. By round 3 the reward is gamed: shorter notes (precision), transcript copying (grounded), plan section balloons, attribution disappears (passive voice) and the judge stops flagging.
3. Verifier recall on injected errors does not move (not trained).
4. Judge 2 scores tuned notes lower than judge 1 more each round; the judge gap is the reward-hacking meter.

## Results (run complete, 12 Sep 12:26–17:40)
| Round | Split | Reward | Composite j1 | Composite j2 | Grounded | Linked | Cited | TR | TP | RL | Plan | Mis | Len | Edits/100w | Judge gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dev | 74.11 | 46.22 | – | 97.26 | 86.07 | 87.44 | 51.51 | 33.20 | 25.91 | 78.12 | 0.00 | 2.06 | 1062 | – |
| 1 | dev | 77.74 | 49.47 | – | 97.84 | 93.64 | 95.66 | 55.24 | 36.92 | 28.89 | 80.21 | 0.00 | 1.81 | 865 | – |
| 2 | dev | 75.38 | 46.12 | – | 97.42 | 95.42 | 96.59 | 54.49 | 35.80 | 29.13 | 76.04 | 0.05 | 1.80 | 867 | – |
| 3 | dev | 80.50 | **51.47** | – | 97.11 | 98.35 | 99.32 | 58.87 | 39.63 | 29.40 | 80.21 | 0.00 | 1.75 | 822 | – |
| 4 | dev | 80.22 | 50.98 | – | 97.11 | 99.24 | 100.0 | 58.45 | 38.12 | 29.81 | 80.21 | 0.00 | 1.77 | 834 | – |
| 0 | test | 68.48 | **41.48** | 44.33 | 92.86 | 85.87 | 87.47 | 52.83 | 27.53 | 20.69 | 76.97 | 0.05 | 1.98 | 1047 | −2.85 |
| 1 | test | 65.32 | 38.91 | 44.56 | 95.21 | 97.71 | 99.29 | 48.70 | 28.46 | 23.03 | 77.42 | 0.11 | 1.58 | 774 | −5.65 |
| 2 | test | 67.07 | 40.35 | 43.94 | 94.26 | 97.26 | 99.15 | 49.40 | 27.98 | 23.45 | 78.43 | 0.08 | 1.61 | 791 | −3.60 |
| 3 | test | 66.81 | 40.81 | 45.25 | 94.22 | 97.50 | 99.46 | 50.36 | 29.02 | 23.14 | 78.06 | 0.08 | 1.63 | 800 | −4.44 |
| 4 | test | 66.89 | 40.47 | **45.93** | 95.09 | 98.11 | 99.52 | 50.81 | 29.39 | 23.14 | 80.52 | 0.11 | 1.60 | 785 | −5.47 |

Sampling per round (dev, 160 samples): mean reward 70.22 → 74.97 → … ; mean best-of-8 79.48 → 82.31 → …; kept 40 per round, cumulative 160 by round 4.

## Reading
- **Dev climbed (46.2 → 51.0), test did not (41.5 → 40.5 under judge 1; 44.3 → 45.9 under judge 2).** Four rounds of 20 transcripts memorize the training set; the held-out composite is net zero to slightly negative by the stricter judge, +1.6 by the lenient one. The judges disagree on the *sign* of the result.
- **What transferred to test**: citation coverage 87 → 99.5%, linked 86 → 98%, grounded 92.9 → 95.1, term precision 27.5 → 29.4, plan recall 77.0 → 80.5, ROUGE-L 20.7 → 23.1, notes 19% shorter, edit effort −25%.
- **What was lost on test**: term recall 52.8 → 50.8 (shorter notes carry fewer terms); misattributions 0.05 → 0.11 by judge 1 (2 → 4 flagged notes of 37; judge 2 flags 0 throughout).
- **Predictions scored**: (1) *falsified* — no half-survival on test after round 1. (2) *not observed* — no passive voice (0.0 every round), length stable at 1.6, no plan balloon, ROUGE did not collapse into transcript copying; either the guards worked or four rounds were too few. (3) trivially true. (4) *falsified in direction* — the gap widened the other way (−2.9 → −5.5): Qwen scores the tuned notes lower than DeepSeek, driven by attribution flags, so the policy did not learn judge 1's habits.
- **Against the loop's best**: the tuned cited Qwen (test 40.5 j1) is still ~3.6 below the cited DeepSeek best (44.1 j1). Writer choice beat four rounds of expert iteration on the local writer.
- **Verdict**: style transfers, content does not, at this data scale. Worth exactly what it demonstrates: the same loop on a real corpus or a clinician edit stream is a different experiment; before running it against a judge, validate the judge, because two judges disagreed on whether the result was positive.

Why not GRPO: DeepSeek (284B, the best writer) cannot be trained here; with 57 consultations and an unvalidated judge, RL learns the judge's blind spots faster than the search loop can. This exercise is the reward side of a post-training programme (eval suites, environments, RL), not the RL side.

Sources: `autoresearch/expert_iter.py`, `autoresearch/expert_iter.sh`, `runs/expert_iter/notebook.md`, `runs/expert_iter/state.json`, `runs/expert_iter.log`.
