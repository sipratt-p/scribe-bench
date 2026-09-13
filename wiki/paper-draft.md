# Paper draft (workshop): What an LLM-judged evaluation of ambient clinical scribes can and cannot see

**Status:** first full draft, 13 Sep 2026. Target: a health-ML workshop (ML4H, CHIL, or a NeurIPS/ICLR health workshop), 4–8 pages plus appendix. Every number below is in [[sources]]; intervals are in `runs/paper_stats.md`. Nothing here has been read by a clinician, and the paper says so in the title's spirit and the limitations' letter.

---

## Title
**What an LLM-judged evaluation of ambient clinical scribes can and cannot see: an open harness and two case studies**

Alternative: *Measuring the scribe, not the model: an open evaluation harness for ambient clinical documentation with LLM judges*

## Abstract (≈200 words)
Ambient clinical scribes turn a recorded consultation into a draft note, and vendors evaluate them on clinician-defined dimensions scored by LLM judges. We release an open harness that reproduces that style of evaluation on public data (PriMock57 audio, ACI-Bench notes, a 7,092-term primary-care lexicon) and use it for two studies. Study 1 evaluates the after-visit pipeline. Across seven open speech recognizers, word error rate varies within a point while medical-term loss varies twofold, separating language-model decoders (8.4% [7.1, 9.7]) from classic decoders (12.5–15.1%); the effect replicates on a synthetic second corpus. Speaker diarization alone does not change the note; naming the clinician before writing cuts misattributions by two thirds. An overnight search over 131 pipeline variants gains term precision (+4.5 [+2.3, +6.7]) but its composite gain (+4.2 [−0.3, +8.8]) is not resolved at n = 37, and the search removed span citations because its objective had no verifiability term; re-scored with one, citations halve unsupported sentences (+5.3 grounded [+3.2, +7.8]). A quarter of reference plan items were never spoken, exposing a ground-truth flaw in completeness metrics. Study 2 evaluates an in-visit decision-support view on the same visits: prompt rules reduce red flags from 61 to 3 (all judged genuine) without losing diagnosis coverage (56/57), while guideline retrieval adds volume, not agreement. Two judges agree on ranking (ρ = 0.80) but not on attribution (κ ≈ 0), and a reinforcement-learning pass they disagree on in sign. We argue these evaluations are useful for ordering pipelines and for catching objective failures, and unsafe as absolute quality claims without clinician calibration.

## 1. Introduction
- Ambient scribes are deployed at scale; the published evaluation method of the market leader is clinician-defined dimensions (attribution, completeness, fairness) scored by LLM judges validated against expert review, tracked by specialty and partner.
- Independent, reproducible versions of that evaluation do not exist on open data. We built one, ran it end to end on open models and open audio, and report what it found and what it could not see.
- Contributions:
  1. **scribe-bench**: an open harness over PriMock57 and ACI-Bench with seven ASR runners, note generation against any OpenAI-compatible server, a claim-level verifier with an injected-error benchmark, LLM-judge scorers for attribution, completeness and grounding, an autoresearch loop with a held-out gate and a second judge, and a replayed in-visit decision-support view with its own timeline scoring. Code, lexicon, splits, prompts and cached outputs released.
  2. **Findings that transferred** across a change of data, judge or metric: decoder type drives medical-term loss; role mapping, not diarization, fixes attribution; span citations trade completeness for grounding; overlap metrics and clinician dimensions diverge under fine-tuning.
  3. **Findings about the evaluation itself**: an objective without a verifiability term removes the verifiability feature; 27% of reference plan items are unspoken, so completeness against the note over-credits guessing; two capable judges rank pipelines alike (ρ = 0.80) but disagree on attribution (κ ≈ 0) and on the sign of an RL result; a model judging its own outputs cannot detect its own escalation errors.
  4. **An in-visit case study** showing that prompt-level rules, not retrieval, moved the safety numbers, with confidence intervals.

## 2. Setup
**Data.** PriMock57: 57 mock UK GP consultations with separate doctor and patient audio tracks (mixed to mono), human transcripts with per-speaker timestamps, and the clinician's note; split 20 dev / 37 test by sorted id. ACI-Bench: 120 encounters (three test splits) with human, ASR and corrected transcripts and reference notes; used for note generation, fine-tuning evaluation, the verifier benchmark and fairness stratification. Lexicon: 7,092 primary-care terms from MTS-Dialog and MedSynth; 1,741 term occurrences in the PriMock human transcripts. Synthetic second audio corpus: ACI-Bench dialogues rendered with a two-voice TTS.

**Pipelines.** Vanilla: streaming ASR (Nemotron-3.5 0.6B, 1.1 s chunks, no speaker labels) → untuned Qwen3.8-27B note. Two-pass: after-visit ASR with an LLM decoder and speaker labels (MOSS-Transcribe-Diarize 0.9B; NVIDIA equivalents Canary-Qwen 2.5B + Sortformer) → optional role mapping (S01/S02 → Doctor/Patient by a 60-token LLM call) → note model → optional span citations → verifier. Search knobs: ASR variant, transcript correction, role map, prompt, extra instruction, citations, verifier gate, proposition scaffold, note model (Qwen3.8-27B local, DeepSeek V4 Flash).

**Metrics.** Transcript: WER after symmetric normalization; medical-term miss rate; DER (250 ms collar); word-level speaker misattribution. Note: lexicon term recall/precision; ROUGE-L; plan-item recall (judge extracts items from the reference note, checks each in the draft; variant: transcript-stated items only); misattributions per note (judge, four types); grounded (judge, whole human transcript as evidence); cited fraction; linked (judge, only the cited lines as evidence); edit effort. Composite for the search: 0.3 TR + 0.3 TP + 0.2 ROUGE-L + 0.2 plan − 40 × misattrib. Verifier: recall on injected errors (negation, number, drug swap, fabricated finding, third-party leak) over 2,764 claims.

**Judges.** Judge 1 Qwen3.8-27B-FP8 (vLLM, thinking off) for all per-note judgments; judge 2 DeepSeek V4 Flash for confirmation of accepted configs. Judge benchmark on injected errors: Qwen 99.1% recall / 6.5% false positive, DeepSeek 95.7% / 4.5%, Gemma 4 98.7% / 5.5%.

**Statistics.** Paired bootstrap over visits (10,000 resamples) for differences; binomial or Wilson intervals for rates; Spearman ρ and Cohen's κ for judge agreement. All reported in `runs/paper_stats.md`.

**Compute.** One workstation with two RTX PRO 6000 (Blackwell) GPUs and a Mac Studio; about $49 of rented H200 time for fine-tuning. Everything else local.

## 3. Study 1: the after-visit note

### 3.1 The decoder, not latency, decides medical-term loss
Table 1 (seven systems, 57 recordings). WER spans 10.3–11.8%; medical-term miss spans 8.4–15.1%. Language-model decoders (MOSS-TD 8.4% [7.1, 9.7]; Canary-Qwen 9.4% [8.0, 10.8]) versus transducer decoders (Nemotron streaming 12.5% [10.9, 14.0]; Parakeet-TDT v3 15.1% [13.4, 16.8]); intervals do not overlap. Running the streaming model offline on the whole recording changes nothing (12.5 vs 12.7). Chart-lexicon hotwords move MOSS-TD by ±0.2. Replicates on the synthetic ACI corpus (MOSS 6.6% vs Nemotron 8.9%). Diarization: MOSS-TD DER 11.4%, Sortformer + Parakeet 11.1%, ≈1% of words misattributed for both; Nemotron 3 Nano Omni prompted as a transcriber was unusable (27% WER, DER 88%).

### 3.2 Diarization alone does not reach the note; role mapping does
Notes from the diarized transcript with anonymous labels carried the same misattribution rate as notes from the undiarized streaming transcript (0.19 per note on all 57); notes from the human transcript, 0.05. On the held-out set the vanilla pipeline and the anonymous-label two-pass pipeline tie on every note metric (composite −1.0 [−5.9, +4.1]). Mapping S01/S02 to Doctor/Patient before writing was the first change the search accepted (dev 44.3 → 45.7, test 39.6 → 41.8) and, in the final configuration, misattributions are 0.03 per note versus 0.08–0.11.

### 3.3 What an overnight search found, with intervals
131 iterations; gate = dev ≥ best − 0.1 → held-out test must beat best → judge 2 within 0.5. Accepted: role map (+0.9 dev), DeepSeek writer (+1.6), complaint-tuned hotwords + a precision-first instruction (+0.8). Held-out composite: vanilla 40.8, best 45.0; difference +4.2 [−0.3, +8.8] under judge 1 and +2.5 [−0.1, +5.3] under judge 2. Components that are resolved: term precision +4.5 [+2.3, +6.7], ROUGE-L +2.2 [+1.0, +3.2]. Not resolved: term recall, misattribution, plan recall, grounding. Every proposer-written prompt variant after the acceptance, transcript correction (six times), the scaffold and the verifier gate gained on dev and failed on test. The search is a benchmark-maximizer by construction; the gate is the useful part.

### 3.4 The objective removed citations
Citations cost ~1 ROUGE-L and buy 4–5 term-precision points on every ACI split, but the composite has no verifiability term, so the search rejected citations eight times (the recall cost is all it could see). Re-scored with grounded/linked: cited − best is +5.3 grounded [+3.2, +7.8] and +2.0 precision [−0.5, +4.6] against −1.2 ROUGE-L [−2.3, −0.2] and −6.5 plan recall [−13.3, +0.2]; 88.8% of sentences cited, 84.0% supported by their own cited lines; unsupported sentences 11.4% → 6.8%. The uncited best is no better grounded than vanilla (−0.8 [−4.0, +2.4]). Lesson stated as such: a score without the trust dimension removes the trust feature.

### 3.5 Completeness against the note over-credits guessing
Item-level audit of the held-out plan items: 41 of 154 reference plan items were never stated in the conversation (silent test orders, drugs written but not discussed). On spoken items the cited writer captures 101/113 vs 105/113 uncited vs 110/114 vanilla; on unspoken items 16 vs 19 vs 11. Half the cited writer's headline plan-recall gap is refusal to invent. Four prompt variants aimed at plan recall gained up to +1.3 on dev and none held on test. Recommendation: score completeness against transcript-stated items, which is also closer to the published "completeness of follow-ups and referrals" dimension.

### 3.6 Overlap metrics and clinician dimensions diverge under fine-tuning (ACI-Bench)
One epoch of LoRA on 2,804 open pairs: Qwen3.8-27B ROUGE-L 34.2 → 43.2, term precision 66 → 79; misattributions ×2–3 (0.08 → 0.23 on test1), follow-up recall −4 to −9 points; citation behaviour erased (99.6% → 9–16% cited) because targets carried none. Untuned Nemotron 3 Ultra 550B sits a point below untuned Qwen 27B on ROUGE-L with the fewest misattributions of any base model. Ordering that held: tuning > base quality > size.

### 3.7 A reinforcement-learning pass the judges disagree about
Expert iteration on the local writer (sample 8, keep 2, LoRA, four rounds, reward = composite + grounding + linking with a length band; predictions logged before the first sample). Dev 46.2 → 51.0. Held-out: 41.5 → 40.5 under judge 1, 44.3 → 45.9 under judge 2. Style transferred (99.5% cited, notes 19% shorter, edit effort −25%); content did not (term recall −2). None of the predicted reward-hacking signatures appeared in four rounds. At 20 training transcripts this is memorization, and the sign disagreement between judges is the point.

## 4. Study 2: in-visit decision support on the live transcript
**Setup.** The streaming transcript is replayed on the human utterance timeline; every 20 s of visit time the model (DeepSeek V4 Flash, 220 tok/s) updates a JSON view: differential with evidence and what is missing, the single discriminating question, red flags (tiered), plan stated vs suggested, safety-netting given vs to add, assumptions under test, announced revisions. Scoring against the clinician's note and the timed transcript: time to the reference diagnosis in the top-3 and lead over the moment the GP stated it; suggested questions later asked; red-flag precision; plan suggestions agree/extra/contradict; revisions toward/away; premature complaint; stability; latency. 57 visits, 1,566 ticks per version.

**Versions.** v2 first prompt; v4 danger-first + consult modality + simple causes; v5 five-slot differential with a time-critical tag, flags only for stated features with a tier, a 90 s hold enforced in code; v5+NHS with guideline pages retrieved per differential entry (NHS A-Z → patient.info → Wikipedia low-trust; 191 lookups).

**Results (Table 3).** Diagnosis ever in top-3: v2 52/57 [81, 96], v5 56/57 [91, 100]; v5 − v2 +0.07 [+0.02, +0.14]. Lead over the GP: median 5:36, in 54/54 visits where the GP stated a diagnosis. Red flags: v2 61 raised / 7 false; v5 3 raised / 0 false; per-visit −1.02 [−1.44, −0.63] flags and −0.12 [−0.23, −0.04] false alarms. Contradicting plan suggestions: 13 → 3, −0.18 [−0.33, −0.04] per visit. v4's danger-first rule raised agreeing suggestions (+0.79 [+0.39, +1.21] per visit) but evicted the likely diagnosis (top-3 −0.05 [−0.12, +0.02]) and produced hedged flags. Guideline retrieval (v5+NHS − v5): no resolved change on any measure; suggestions +31 mostly "extra"; the NHS anaphylaxis page re-imported emergency advice on an insect-sting visit the prompt rules had calmed.

**Reading.** The safety improvements are prompt rules (stated-feature-only flags, a hold, tiering), and they are resolved at n = 57. "Ahead of the GP" measures how fast the view reads the history, not diagnostic insight. Every flag and contradiction is a model judging a model; the three v5 flags and three contradictions are what a clinician should read first.

## 5. What the judges can and cannot see
- **Ranking, yes.** Per-note composites from the two judges correlate at ρ = 0.80 over 74 notes; both order vanilla < best.
- **Levels, no.** Judge 2 scores every note 2–4 points higher; plan recall means differ by 5 points (ρ = 0.54).
- **Attribution, not at all.** Judge 1 flags 4/74 notes, judge 2 1/74; raw agreement 0.93 is base rate, κ = −0.02. The misattribution penalty carries 40 points per event in the composite, so this is the term most exposed to judge choice.
- **Self-judging.** In Study 2 the writer and the judge are the same model; it cannot detect its own over-escalation (the anaphylaxis-vs-antihistamine disagreement is judged "contradicts" both ways).
- **Objective design.** Two failures were caused by the metric, not the model: citations removed (3.4) and unspoken plan items credited (3.5). Both were found by re-scoring, not by a new model.
- **RL sign disagreement** (3.7).
- Implication: use LLM judges to order pipelines and to audit objectives; calibrate at least the attribution judge against clinicians before any absolute claim.

## 6. Limitations
Mock consultations, UK primary care, 57 recordings; terse reference notes make absolute overlap scores low by construction; no chart context; judges uncalibrated; the vanilla pipeline is an open reconstruction, not any vendor's product; DeepSeek is both writer and judge 2 for the best configuration; the live view is a replay, and consult modality is inferred; the plan-recall audit and the live scoring use the same judge that generated the outputs.

## 7. Reproducibility
Code, lexicon, splits, prompts, run notebooks and cached model outputs are in the repository; every table's source file is listed in [[sources]]. All models are open weights; the two workstation GPUs and about $49 of cloud time suffice to rerun everything except the 550B base-model run.

## Figures and tables (list)
1. Table 1: ASR matrix (WER, term miss with CIs, DER, speakers) — from [[decoder-finding]].
2. Figure 1: term miss vs WER scatter, decoder family coloured.
3. Table 2: held-out note metrics for vanilla / v1 / best / cited with bootstrap CIs — from `runs/paper_stats.md`.
4. Figure 2: the search's dev-vs-test trajectory (accepted points, rejected dev gains) — from `autoresearch/notebook.md`.
5. Table 3: live-view versions with CIs — from `runs/paper_stats.md`.
6. Figure 3: per-visit red flags and contradictions, v2 vs v5, paired.
7. Table 4: judge agreement.
8. Appendix A: run log ([[live-experiments]], [[autoresearch-loop]]); Appendix B: prompts; Appendix C: verifier benchmark ([[verifier]]); Appendix D: fairness ([[fairness]]).

## To do before submission
- Bootstrap CIs for the ACI-Bench fine-tuning table (per-encounter data exists in `runs/effort_notes_*.json`).
- A third judge (Gemma 4 or a second open model) on the 74-note attribution set to see whether κ ≈ 0 is DeepSeek's under-flagging or a general problem.
- Figure 2 and Figure 3 drawn.
- Related work: Abridge evaluation whitepaper; ACI-Bench and PriMock57 papers; LLM-as-judge validity literature; note-generation benchmarks (MedBench record generation, the July 2026 ACI/PriMock comparison).
- A clinician read of 20 notes and the six live-view items as a "v2 paper" hook, or as a small appendix if one can be arranged before the deadline.
