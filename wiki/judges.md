# Judges

Every clinician-dimension number comes from an LLM judge with thinking disabled (`chat_template_kwargs.enable_thinking=false`).

- **Judge 1 / screening**: Qwen3.8-27B-FP8 on the beast vLLM (:8004). Used for attribution, plan-item recall, grounded, linked, the verifier gate, and the loop's dev/test scoring.
- **Judge 2 / confirming**: DeepSeek V4 Flash on the Mac (oMLX :8600, `v4-flash`). Re-scores accepted configs on test; tolerance band 0.5. **Caveat**: DeepSeek is also the note *writer* in the best config, so its judge-2 numbers are self-scored; quote judge 1.
- **Gemma 4 26B-A4B** (llama-server :8500) was the original judge; user judged it untrustworthy; it saturated at ~500 s and OOM'd at 0.5 util. Kept only for the demo's live mode.

## Judge benchmark (injected-error claims, same task as the [[verifier]])
| Judge | Recall | False positives |
|---|---|---|
| Qwen3.8-27B | 99.1% | 6.5% |
| DeepSeek V4 Flash | 95.7% | 4.5% |
| Gemma 4 26B-A4B | 98.7% | 5.5% |

## Three-judge attribution check (13 Sep)
Gemma 4 26B-A4B re-judged the same 74 held-out notes (vanilla + best) for misattribution. Flags: Qwen 4, DeepSeek 1, Gemma 13. Pairwise κ: Qwen–DeepSeek −0.02, Qwen–Gemma +0.29, DeepSeek–Gemma +0.12; no note flagged by all three. Gemma reverses the pipeline direction (vanilla 0.135 vs best 0.216 per note). The "misattributions 0.08 → 0.03" result therefore rests on one judge and should be described as unresolved. See `runs/paper_stats.md` and [[paper-draft]].

## Live view, cross-judged item by item (13 Sep)
The v5 live run was re-judged with one verdict per item (flag, suggestion, snapshot) by DeepSeek and Gemma. Coverage 56 → 51/57 under both item judges (κ 0.63 with each other, 0.26 with the timeline prompt); plan verdicts swing with the prompt (26/58/3 → 67/5/15 → 21/65/1), κ 0.08–0.28; all three flags were stated features, Gemma accepts 3, DeepSeek's and Qwen's item judges 1 (tier). Qwen as a fourth judging (13 Sep evening): coverage 47/57, plan 32/54/1; Gemma–Qwen plan κ +0.72 while each is ~0.1 with DeepSeek's item verdicts, i.e. the two judges that did not write the views agree with each other and the writer is lenient to itself. On the official-weight live run (L7) the same two judges give coverage 50 and 48/57 (κ +0.85), plan 21/96/1 and 40/76/2 (κ +0.59), one genuine flag; timeline prompt 56/57 and 29/89/0. Numbers in `runs/paper_stats.md`; scripts `autoresearch/live_itemjudge.py`.

## Judge gap
Judge 1 − judge 2 composite on the same notes is the reward-hacking meter in [[expert-iteration]]. Round 0 (base cited Qwen): −2.85. Round 1: −5.65 (DeepSeek scores tuned notes *higher* than Qwen does; prediction 4 expected the opposite).

Known judge habits: DeepSeek flags almost no misattributions for any config (0.00–0.03) and scores plan recall higher (84–88 vs 76–81). Qwen is stricter on attribution. Neither is validated against clinicians, which the vendor evaluation literature treats as the step that makes a metric real.

Prompt gotchas: verdict must come after reasoning (`JUDGE_SYSTEM`, `parse_verdict`); DeepSeek produced degenerate `begin_of_sentence` repeats on some notes (regenerated); `_llm` in the demo retries.

Sources: `scribe_bench/note_judge.py`, `verifier.py`, `autoresearch/loop.py`, `runs/expert_iter/notebook.md`.
