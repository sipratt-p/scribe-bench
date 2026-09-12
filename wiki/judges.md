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

## Judge gap
Judge 1 − judge 2 composite on the same notes is the reward-hacking meter in [[expert-iteration]]. Round 0 (base cited Qwen): −2.85. Round 1: −5.65 (DeepSeek scores tuned notes *higher* than Qwen does; prediction 4 expected the opposite).

Known judge habits: DeepSeek flags almost no misattributions for any config (0.00–0.03) and scores plan recall higher (84–88 vs 76–81). Qwen is stricter on attribution. Neither is validated against clinicians, which Abridge's whitepaper treats as the step that makes a metric real ([[abridge]]).

Prompt gotchas: verdict must come after reasoning (`JUDGE_SYSTEM`, `parse_verdict`); DeepSeek produced degenerate `begin_of_sentence` repeats on some notes (regenerated); `_llm` in the demo retries.

Sources: `scribe_bench/note_judge.py`, `verifier.py`, `autoresearch/loop.py`, `runs/expert_iter/notebook.md`.
