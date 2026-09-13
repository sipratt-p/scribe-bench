# Benchmark-maxing vs quality-maxing

A common piece of advice in model development is not to benchmark-max. This project did both, and the split is explicit.

## Benchmark-maxing, by construction
- The [[autoresearch-loop]] itself: one composite on 57 mock UK GP consultations, an LLM proposer writing prompt wordings. Evidence: repeated dev gains that failed the 37-file test gate.
- Hotwords: transcript slightly worse, note score +1 ([[decoder-finding]]).
- Iterations 70+: rephrasings of one sentence.
- [[expert-iteration]] round 1: +3.3 dev, −2.6 test.

## Quality-maxing: survived a change of data, judge, or metric
- [[decoder-finding]] (two corpora, four checkpoints).
- [[role-mapping]] (mechanism; explains the vanilla/v1 tie).
- The ROUGE vs clinician-dimension divergence in [[fine-tuning-findings]].
- The citation correction in [[citations-and-verifiability]] (noticing the score gamed itself and re-scoring).
- The [[plan-recall-gap]] analysis (the metric was wrong, not the writer).

## Not yet quality-maxing, say so
- Judges are Qwen 27B with prompts written here, not validated against clinicians.
- Composite weights are mine; edit effort is measured but not optimized; no specialty or fairness term in the loop.

Interview line: "The search loop is a benchmark-maxer and I treat it as one; its job is to find candidates cheaply. The findings I'd defend survived a change of dataset, judge, or metric. Its most useful output was a failure: it optimized citations away because my score did not value verifiability." See [[interview-narrative]].
