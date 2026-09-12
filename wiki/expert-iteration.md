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

## Results so far
| Round | Split | Reward | Composite j1 | Composite j2 | Grounded | Linked | TR | TP | Plan | Mis | Len | Edits/100w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dev | 74.11 | 46.22 | – | 97.26 | 86.07 | 51.51 | 33.20 | 78.12 | 0.00 | 2.06 | 1062 |
| 0 | test | 68.48 | 41.48 | 44.33 | 92.86 | 85.87 | 52.83 | 27.53 | 76.97 | 0.05 | 1.98 | 1047 |
| 1 | dev | 77.74 | **49.47** | – | 97.84 | 93.64 | 55.24 | 36.92 | 80.21 | 0.00 | 1.81 | 865 |
| 1 | test | 65.32 | **38.91** | 44.56 | 95.21 | 97.71 | 48.70 | 28.46 | 77.42 | 0.11 | 1.58 | 774 |

Round 1 samples: mean reward 70.22, mean best-of-8 79.48 (greedy 74.11); 40 examples, loss 0.22 after 7 steps.

Reading after round 1
- **Prediction 1 falsified.** Dev +3.3; test −2.6 under judge 1, flat under judge 2. 40 examples from 20 transcripts memorizes dev.
- **Style transferred, content did not**: 99% of sentences cited, linked 86 → 98%, notes 20% shorter, edit effort −26%; term recall −4 (shorter notes drop terms), misattributions 2 → 4 notes of 37.
- **Judge gap moved the wrong way** for prediction 4 (−2.85 → −5.65): Qwen scores the tuned notes *lower* than DeepSeek, so the policy is not learning judge 1's habits yet.
- Agentless-sentence proxy is saturated at baseline (bullet-style notes); passive rate starts at 0 and stays 0.

Rounds 2–4 pending (ETA ~18:30 12 Sep). Update this table from `runs/expert_iter/notebook.md`.

Why not GRPO: DeepSeek (284B, the best writer) cannot be trained here; with 57 consultations and an unvalidated judge, RL learns the judge's blind spots faster than the search loop can. This exercise is the reward side of Abridge's "milestone three" (eval suites, environments, RL) ([[abridge]]).

Sources: `autoresearch/expert_iter.py`, `autoresearch/expert_iter.sh`, `runs/expert_iter/notebook.md`, `runs/expert_iter/state.json`, `runs/expert_iter.log`.
