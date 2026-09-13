# Index

Ambient clinical scribe evaluation harness, built 11–12 Sep 2026 for the Abridge interview (Sun 13 Sep, 2pm). Everything measured locally on open data; ~$49 of rented H200 time in total.

## Start here
- [[glossary]] — acronyms and jargon in plain language, grouped by topic
- [[overview]] — what was built, the one-paragraph result, and how to read the rest
- [[interview-narrative]] — how to say it to an interviewer, with what not to claim
- [[log]] — dated timeline of the work

## Findings
- [[decoder-finding]] — **transferred**: the ASR decoder, not latency, decides medical-term loss
- [[role-mapping]] — **transferred**: anonymous speaker labels do nothing for the note; naming the clinician does
- [[citations-and-verifiability]] — **transferred**: citations halve unsupported sentences; the loop threw them away because the score did not value them
- [[fine-tuning-findings]] — **transferred**: tuning > base quality > size; the SFT that wins ROUGE loses on clinician dimensions
- [[verifier]] — a small judge with the citation in front of it catches ~99% of injected errors
- [[plan-recall-gap]] — a quarter of reference plan items are never spoken; the metric was wrong
- [[vanilla-vs-best]] — the head-to-head table with every metric, both judges
- [[fairness]] — gender gaps within a point; age is a lead, not a finding

## Methods
- [[metrics]] — every metric defined, with what it can and cannot see
- [[judges]] — which LLM judges, the judge benchmark, and the judge-gap meter
- [[autoresearch-loop]] — the overnight search: knobs, gates, what it found, what it got wrong
- [[expert-iteration]] — the RL exercise: predictions first, results per round
- [[benchmark-maxing-vs-quality]] — which results are which, and why the gate is the product
- [[two-pass-asr]] — the design: streaming pass 1, LLM-decoder diarized pass 2

## Entities
- [[models]] — every ASR and note model tried, with numbers
- [[datasets]] — PriMock57, ACI-Bench, the lexicon, the synthetic corpus
- [[infrastructure]] — beast, Mac, ports, pods, cost, gotchas
- [[abridge]] — what Abridge actually runs and measures (public + the July workshop doc)
- [[nvidia-engagement]] — what the numbers mean for the NVIDIA account

## Deliverables
- [[deliverables]] — the artifact, the demo, the repo, the experiments page
- [[live-synthesis]] — separate experiment: in-visit decision support on the live transcript (demo `/live`)
- [[live-experiments]] — the run log for it: every version L0–L5, results side by side, what each taught, the search story

## Paper
- [[paper-draft]] — workshop-paper draft: title, abstract, sections, tables, figure list, to-do; statistics in `runs/paper_stats.md`

## Meta
- [[open-questions]] — what is unresolved and what to run next
- [[sources]] — raw files behind every page
