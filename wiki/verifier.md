# Verifier

Claims = sentences of cited notes, paired with their cited transcript lines plus one line of context. A quarter are perturbed (negation flip, number change, drug swap, fabricated finding, third-party PHI leak) and a small judge is asked, reason first, verdict last (`parse_verdict`).

| ACI split | Claims | Injected | Recall | Flags on clean | Negation | Number | Drug swap | Fabricated | Leak |
|---|---|---|---|---|---|---|---|---|---|
| test1 | 905 | 232 | 98.7% | 5.5% | 85% | 88% | n/a | 100% | 100% |
| test2 | 968 | 247 | 99.6% | 3.5% | 93% | 100% | 100% | 100% | 100% |
| test3 | 891 | 229 | 98.7% | 4.4% | 90% | 100% | 100% | 99% | 100% |

Judge benchmark on the same task ([[judges]]): Qwen3.8-27B 99.1% recall / 6.5% false positives; DeepSeek V4 Flash 95.7% / 4.5%; Gemma 4 26B-A4B 98.7% / 5.5%.

Reading: fabricated findings and third-party leaks are caught every time; the misses are subtle edits (a dropped "no", a doubled number), which is what a classifier trained on clinician edit deltas should learn. Some "clean" flags are the generator's own hallucinations. Nothing required a frontier model. Every model tested leaked third-party personal information into notes; prompt instructions reduced but did not fix it → verifier ahead of the draft is the cheapest win ([[nvidia-engagement]]).

Gotchas: DeepSeek gave verdict-before-reasoning until the prompt was made verdict-last; the demo's verifier claims were re-judged with Qwen after Gemma saturated.

Sources: `scribe_bench/verifier.py`, `RESULTS.md` verifier table, demo `claims` panel ([[deliverables]]).
