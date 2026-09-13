# The overnight autoresearch loop

Karpathy-style: a notebook of results, a proposer LLM that reads it and suggests the next config, a manual queue, a mutation fallback, and a hard gate. Started 11 Sep ~23:00; deadline 13 Sep 08:30. 96+ iterations by 12 Sep 15:00.

## Setup
- **Data**: PriMock57 split dev 20 / test 37 (sorted ids). Every LLM call cached by content hash under `runs/autoresearch/cache`.
- **Knobs**: `asr_variant` (moss_plain, moss_hot150, moss_hotcc, canary_qwen, nemotron_stream, human), `asr_correct` (LLM transcript correction with lexicon), `role_map` (none / llm), `prompt` (base / attrib / strict / attrib_strict), `extra` (free-text instruction), `cite`, `gate` (none / drop_flagged via verifier), `scaffold` (proposition extraction first), `note_model` (qwen27b on the beast, dsv4flash on the Mac; flashnext dropped).
- **Score**: the loop composite ([[metrics]]).
- **Gate**: dev ≥ best − 0.1 → run test (37) → must beat best on test → judge 2 (DeepSeek) must agree within 0.5. Near-ties on dev still earn a test run.

## What it accepted
| Iter | Change | Dev | Test | Judge 2 |
|---|---|---|---|---|
| 1 | baseline: moss_plain → Qwen, no role map | 44.26 | 39.62 | 44.76 |
| 4 | + LLM role map | 45.74 | 41.79 | 44.65 |
| 12 | + DeepSeek writer, attrib prompt | 47.30 | 44.35 | 46.22 |
| 31 | DeepSeek writer, base prompt | 47.61 | 44.57 | 46.42 |
| 47 | + complaint hotwords + precision-first extra | **48.44** | **45.01** | **46.80** |

Accepted best: `{asr_variant: moss_hotcc, role_map: llm, prompt: base, extra: "Prioritize high-precision terminology. Explicitly state 'no plan' if no plan items are mentioned. Do not hallucinate follow-up instructions or medications.", cite: 0, gate: none, scaffold: 0, note_model: dsv4flash}`.

Contributions (dev ablations): role map ≈ +0.9, DeepSeek writer ≈ +1.6, hotwords + extra ≈ +0.8 combined.

**Official weights (13 Sep evening).** The loop ran on an abliterated DeepSeek build. Re-run on the official release weights the accepted config scores test 42.26 under judge 1 (+1.25 [−3.76, +6.23] over vanilla) and 44.40 under judge 2 (−0.66 [−4.50, +3.03]); term precision and ROUGE-L gains survive at half size, plan recall and grounding become resolved losses. About three of the four reported points were the writer build. Full table in [[vanilla-vs-best]]; the held-out gate caught dev overfitting but cannot catch a gain that is constant across every candidate.

## What it rejected, and why that matters
- **Dev gains that failed test** (the proposer overfitting 20 files): scaffold (iter 30: dev +2.3, test 40.0), verifier gate (iter 40: test 44.57 vs 44.57 tie → rejected), scaffold on best (iter 50: test 42.75), transcript correction (iter 95: dev 48.78, test 42.51; fourth time), every proposer-written prompt rewording after iteration 47.
- **Citations**, eight times, because the score had no verifiability term ([[citations-and-verifiability]]).
- **Canary-Qwen** as pass 2 (iter 91: dev 42.82).
- **Hotwords** are a score artifact: transcript metrics got slightly worse ([[decoder-finding]]).

## Reading
The loop is a benchmark-maxer by construction ([[benchmark-maxing-vs-quality]]). The gate is the product. Its most useful output was a failure (citations). See `/experiments` in the demo for every row ([[deliverables]]).

Sources: `autoresearch/loop.py`, `autoresearch/notebook.md`, `autoresearch/state.json`, `autoresearch/queue.jsonl`, `runs/loop_best/`.
