# Interview narrative (Abridge, Sun 13 Sep, 2pm)

## The spoken version
"I built the eval before I built the pipeline. On 57 open consultations, a vanilla streaming-to-note pipeline and my two-pass design tied: diarization on its own did nothing for the note. Telling the note model which speaker is the doctor did. The decoder, not latency, is what loses medical terms, and that held on a second corpus. I let a search loop try ~100 variants overnight with a held-out gate and a second judge so it couldn't fool itself; it found two to four points, and most of what it tried looked good on dev and failed the gate. Its most useful output was a failure: it optimized citations away because my score had no verifiability term. When I added one, citations came back at no net cost and halved unsupported sentences. That is the metric-owner lesson, and it is exactly what your clinician-validated dimensions exist to prevent. The open problem is plan completeness under citations, which is the closest thing on open data to your order-generation task, and a quarter of it turned out to be the metric counting orders the doctor never said aloud."

## Numbers to have ready
- Term miss: 12.7% streaming vs 8.6% LLM-decoder at ~equal WER ([[decoder-finding]]).
- Vanilla vs best on held-out: composite +4.2 [95% CI −0.3, +8.8] under Qwen, +2.5 [−0.1, +5.3] under DeepSeek, so say "probably better, not resolved at 37 visits"; resolved: term precision +4.5 [+2.3, +6.7], ROUGE-L +2.2, grounding with citations +5.3 [+3.2, +7.8]; misattributions 0.08 → 0.03 (not resolved); 84% of sentences checkable ([[vanilla-vs-best]], `runs/paper_stats.md`).
- SFT: +9 ROUGE-L, misattributions ×2–3, follow-ups −9 ([[fine-tuning-findings]]).
- Verifier: 98.7–99.6% recall on injected errors ([[verifier]]).
- Plan items: 41 of 154 never spoken; vanilla 96.5% / best 92.9% / cited 89.4% on spoken ones ([[plan-recall-gap]]).
- RL round 1: +3.3 dev, −2.6 test ([[expert-iteration]]).

## What not to claim
- That the vanilla pipeline is Abridge's product (it is an unoptimized baseline; Abridge has Linked Evidence and tuned models).
- That hotwords help (transcript got slightly worse).
- Absolute PriMock scores (terse UK GP references; ordering is the finding).
- That the judges are validated against clinicians (they are not).
- That the composite gain is measured: its interval includes zero. Quote the resolved components (precision, ROUGE-L, grounding) and the transcript term-miss gap, whose intervals do not overlap.

## Their vocabulary to use
attribution, completeness, Linked Evidence, order generation, mid-training, "don't benchmark-max", milestone three (eval suites / environments / RL) — see [[abridge]].

## Framing of the RL work
"The harness is the reward side of your milestone three. Before you run RL against a judge, here is how to find out what the judge cannot see." ([[benchmark-maxing-vs-quality]])
