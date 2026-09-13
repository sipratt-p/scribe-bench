# Paper draft (workshop): What an LLM-judged evaluation of ambient clinical scribes can and cannot see

**Status:** first full draft, 13 Sep 2026. Target: a health-ML workshop (ML4H, CHIL, or a NeurIPS/ICLR health workshop), 4–8 pages plus appendix. Every number below is in [[sources]]; intervals are in `runs/paper_stats.md`. Nothing here has been read by a clinician, and the paper says so in the title's spirit and the limitations' letter.

---

## Title
**What an LLM-judged evaluation of ambient clinical scribes can and cannot see: an open harness and two case studies**

Alternative: *Measuring the scribe, not the model: an open evaluation harness for ambient clinical documentation with LLM judges*

## Abstract (≈200 words)
Ambient clinical scribes turn a recorded consultation into a draft note, and are increasingly evaluated on clinician-defined dimensions scored by LLM judges. We release an open harness that reproduces that style of evaluation on public data (PriMock57 audio, ACI-Bench notes, a 7,092-term primary-care lexicon) and use it for two studies. Study 1 evaluates the after-visit pipeline. Across seven open speech recognizers, word error rate varies within a point while medical-term loss varies twofold, separating language-model decoders (8.4% [7.1, 9.7]) from classic decoders (12.5–15.1%); the effect replicates on a synthetic second corpus. Speaker diarization alone does not change the note; naming the clinician before writing cuts misattributions by two thirds under one judge, a result three judges then fail to agree on. An overnight search over 131 pipeline variants reported a +4.2 composite gain on the held-out set; re-run with the writer's official release weights instead of the community build the search had used, the gain is +1.3 [−3.8, +6.2], term precision (+2.5 [+0.5, +4.5]) and ROUGE-L survive at half size, and plan recall (−8.1 [−13.8, −1.7]) and grounding become resolved losses, while a two-pass open pipeline the search never tried scores as high without any search. The search had also removed span citations because its objective had no verifiability term; re-scored with one, citations recover grounding (+6.1 [+3.2, +8.9] over the uncited best). A quarter of reference plan items were never spoken, exposing a ground-truth flaw in completeness metrics. Study 2 evaluates an in-visit decision-support view on the same visits: prompt rules reduce red flags from 61 to 3 (all judged genuine) without losing diagnosis coverage (56/57), while guideline retrieval adds volume, not agreement. Two judges rank notes alike (ρ = 0.80) yet disagree on the sign of the best − vanilla difference on the official weights, and three judges do not agree on attribution: pairwise κ from −0.02 to +0.29, no note flagged by all three, and the third judge reverses the direction of the attribution difference between pipelines; a reinforcement-learning pass the first two disagree on in sign. We argue these evaluations are useful for ordering pipelines and for catching objective failures, and unsafe as absolute quality claims without clinician calibration.

## 1. Introduction
- Ambient scribes are deployed at scale, and the evaluation method described in the vendor and research literature is clinician-defined dimensions (attribution, completeness, fairness) scored by LLM judges validated against expert review.
- Independent, reproducible versions of that evaluation do not exist on open data. We built one, ran it end to end on open models and open audio, and report what it found and what it could not see.
- Contributions:
  1. **scribe-bench**: an open harness over PriMock57 and ACI-Bench with seven ASR runners, note generation against any OpenAI-compatible server, a claim-level verifier with an injected-error benchmark, LLM-judge scorers for attribution, completeness and grounding, an autoresearch loop with a held-out gate and a second judge, and a replayed in-visit decision-support view with its own timeline scoring. Code, lexicon, splits, prompts and cached outputs released.
  2. **Findings that transferred** across a change of data, judge or metric: decoder type drives medical-term loss; role mapping, not diarization, fixes attribution; span citations trade completeness for grounding; overlap metrics and clinician dimensions diverge under fine-tuning; a two-pass transcript with a role map and an untuned open writer is as good as anything the search found.
  3. **Findings about the evaluation itself**: an objective without a verifiability term removes the verifiability feature; about three of the four composite points an overnight search reported came from the writer build it happened to run on, and vanish on the official weights; 25–27% of reference plan items are unspoken, so completeness against the note over-credits guessing; capable judges rank notes alike (ρ = 0.80) but disagree on the sign of a small pipeline difference, and three of them disagree on attribution (pairwise κ −0.02 to +0.29, no note flagged by all three) and two disagree on the sign of an RL result; a model judging its own outputs cannot detect its own escalation errors.
  4. **An in-visit case study** showing that prompt-level rules, not retrieval, moved the safety numbers, with confidence intervals.

## 2. Setup
**Data.** PriMock57: 57 mock UK GP consultations with separate doctor and patient audio tracks (mixed to mono), human transcripts with per-speaker timestamps, and the clinician's note; split 20 dev / 37 test by sorted id. ACI-Bench: 120 encounters (three test splits) with human, ASR and corrected transcripts and reference notes; used for note generation, fine-tuning evaluation, the verifier benchmark and fairness stratification. Lexicon: 7,092 primary-care terms from MTS-Dialog and MedSynth; 1,741 term occurrences in the PriMock human transcripts. Synthetic second audio corpus: ACI-Bench dialogues rendered with a two-voice TTS.

**Pipelines.** Vanilla: streaming ASR (Nemotron-3.5 0.6B, 1.1 s chunks, no speaker labels) → untuned Qwen3.8-27B note. Two-pass: after-visit ASR with an LLM decoder and speaker labels (MOSS-Transcribe-Diarize 0.9B; NVIDIA equivalents Canary-Qwen 2.5B + Sortformer) → optional role mapping (S01/S02 → Doctor/Patient by a 60-token LLM call) → note model → optional span citations → verifier. Search knobs: ASR variant, transcript correction, role map, prompt, extra instruction, citations, verifier gate, proposition scaffold, note model (Qwen3.8-27B local, DeepSeek V4 Flash). **Writer builds.** The overnight search and the first live-view runs used a community-modified ("abliterated", refusal-ablated) NVFP4 build of DeepSeek V4 Flash because it was the build the GPU box could serve at the time; every DeepSeek-dependent number in the paper was then re-run with the official release weights (the DSpark NVFP4 build of the same checkpoint, thinking off) as writer and as judge 2, and the official-weight numbers are the ones reported unless a table says otherwise.

**Metrics.** Transcript: WER after symmetric normalization; medical-term miss rate; DER (250 ms collar); word-level speaker misattribution. Note: lexicon term recall/precision; ROUGE-L; plan-item recall (judge extracts items from the reference note, checks each in the draft; variant: transcript-stated items only); misattributions per note (judge, four types); grounded (judge, whole human transcript as evidence); cited fraction; linked (judge, only the cited lines as evidence); edit effort. Composite for the search: 0.3 TR + 0.3 TP + 0.2 ROUGE-L + 0.2 plan − 40 × misattrib. Verifier: recall on injected errors (negation, number, drug swap, fabricated finding, third-party leak) over 2,764 claims.

**Judges.** Judge 1 Qwen3.8-27B-FP8 (vLLM, thinking off) for all per-note judgments; judge 2 DeepSeek V4 Flash for confirmation of accepted configs. Judge benchmark on injected errors: Qwen 99.1% recall / 6.5% false positive, DeepSeek 95.7% / 4.5%, Gemma 4 98.7% / 5.5%.

**Statistics.** Paired bootstrap over visits (10,000 resamples) for differences, with the recording as the sampling unit for transcript metrics and the visit for note and live metrics; binomial or Wilson intervals for rates; Spearman ρ and Cohen's κ for judge agreement. We report every per-metric interval rather than a single headline; the composite is an engineering objective for the search, not a claim, and we do not correct for multiple comparisons because the point is the pattern across metrics, not any one test. All numbers are in `runs/paper_stats.md`.

**Compute.** One workstation with two RTX PRO 6000 (Blackwell) GPUs and a Mac Studio; about $49 of rented H200 time for fine-tuning. Everything else local.

## 2a. Related work (every reference below verified against the arXiv, ACL Anthology, PMC, Europe PMC or medRxiv record on 13 Sep 2026)
- **Datasets.** PriMock57 [1]: 57 mock primary-care consultations with audio, utterance-level transcripts and notes, released as an ASR and note-generation benchmark; we add per-speaker timing exports and a dev/test split. ACI-Bench [2]: 207 role-played encounters with human, ASR and corrected transcripts and notes across three test sets; our note-generation, fine-tuning, verifier and fairness studies use its test splits. MedMosaic [3] is a large medical-audio question-answering benchmark (46,701 pairs) for multimodal models; the Multicultural Medical Assistant study [4] asks whether LLMs can correct medical ASR errors across accents. Our medical-term miss rate is a narrower, lexicon-based view of the same problem.
- **Evaluation practice in deployed systems.** Published vendor methodology describes clinician-defined dimensions scored by LLM judges validated against expert review, hallucination-detection frameworks, and clinician-in-the-loop studies before deployment. Our attribution, completeness and grounding judges reproduce that shape on open data; our verifier's injected-error recall (98.7–99.6%) is an open analogue of a hallucination-detection rate, with the caveat that ours is measured on synthetic corruptions.
- **LLM-as-a-judge in healthcare.** Two 2026 scoping reviews catalogue the field: Li et al. [5] analyse 134 studies from January 2023 to February 2026 with a focus on human alignment, and the MedJUDGE review [6] covers 49 studies from 2020 to January 2026 and proposes a governance framework. Reported alignment ranges widely: on multi-document EHR summaries an LLM judge reached an intraclass correlation of 0.82 with clinicians on a validated instrument [7]; on physician-to-physician eConsult concordance the best judge reached κ = 0.75 against a physician majority vote while the three physicians agreed among themselves only moderately (Fleiss κ = 0.46) [8]. DeLucia et al. [9] show judges and clinicians reaching the same completeness verdicts for different reasons; Sun et al. [10] contrast similarity metrics with LLM judges on clinical dialogue. Our contribution to this line is narrower and mechanical: capable judges agree on ranking (ρ = 0.80) and disagree on a low-base-rate dimension (attribution, pairwise κ −0.02 to +0.29 across three judges), and an objective built from judged terms can remove a feature the judges never see.
- **Rubric-based clinical evaluation.** HealthBench [11] uses rubrics written by 262 physicians (48,562 criteria); DistillNote [12] evaluates note summaries by their usefulness for downstream clinical prediction. Our plan-item audit (27% of reference items unspoken) is an instance of the ground-truth problem such rubrics also face when the reference note contains decisions that were never verbalized.
- **Medical ASR beyond WER.** Afonja et al. [13] show that low overall WER coexists with high error on clinical entities and that fine-tuning on accented clinical speech closes much of the gap; Adedeji et al. [4, 14] use LLMs to repair medical transcription errors; Ellis et al. [15] show WER and a suite of standard metrics correlate poorly with clinician-assigned risk labels and train a judge to replicate the clinical assessment; MedWER [16] proposes a model-free term-list protocol. Our lexicon-based term-miss rate sits between WER and those judged-impact scores: model-free like MedWER, but reported with the decoder as the explanatory variable.
- **Diarization.** Sortformer [17] is the end-to-end diarizer we pair with Parakeet; the finding that anonymous speaker labels do not reach the note until roles are assigned is, to our knowledge, not reported elsewhere.
- **Note generation models.** MEDIQA-Chat 2023 systems [18] established GPT-4 in-context generation as competitive with human notes on ACI-Bench; Wang et al. [19] adapt an open 13B model with continued pretraining, SFT and RL from AI and human feedback; Ahmed and Sadeque [20] compare parameter-efficient fine-tunes on MTS-Dialog and report persistent hallucination. Our fine-tuning result (overlap up, clinician dimensions down, every interval excluding zero) is the same tension measured on both sides at once.
- **Judge bias and reward hacking.** Wataoka et al. [21] trace self-preference in LLM judges to a preference for low-perplexity text; Ye et al. [22] quantify twelve judge biases; Norman et al. [23] show exact-match agreement overstates judge discrimination and that κ deflates agreement by 33–41 points, which matches our attribution result. Khalaf et al. [24] show the rise-then-fall pattern of inference-time reward hacking is inevitable under common alignment methods; Nishimura-Gasparian et al. [25] show RL reasoning training increases specification gaming. Our search loop removing citations, and our expert-iteration pass learning style but not content, are small instances of the same dynamics under an LLM-judged objective.
- **Ambient documentation systems.** Commercial scribes publish latency and adoption figures but not reproducible evaluations; open-weight speech and note models (NVIDIA Nemotron/Parakeet/Canary, MOSS-Transcribe-Diarize, Qwen, DeepSeek) make an independent evaluation feasible, which is what this paper does.

### References (verified)
1. A. Papadopoulos Korfiatis, F. Moramarco, R. Sarac, A. Savkov. *PriMock57: A Dataset of Primary Care Mock Consultations.* Proc. ACL 2022 (Vol. 2: Short Papers), pp. 588–598. arXiv:2204.00333.
2. W. Yim, Y. Fu, A. Ben Abacha, N. Snider, T. Lin, M. Yetisgen. *Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation.* Scientific Data 10, 586 (2023). doi:10.1038/s41597-023-02487-3. arXiv:2306.02022.
3. H. Rajgarhia, S. Ojha, A. Shaik, A. Pothanapalli, R. Lokesh, A. Mukherji, P. Desikan. *MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio.* arXiv:2605.00969 (May 2026).
4. A. Adedeji, M. Sanni, E. Ayodele, S. Joshi, T. Olatunji. *The Multicultural Medical Assistant: Can LLMs Improve Medical ASR Errors Across Borders?* arXiv:2501.15310 (Jan 2025).
5. L. Li, D. Li, C. Chen, R. Ma, R. Yu, M. Lin, R. Yin, L. Fan, C. Shyr, S. Ma, M. Liu, S. Bethard. *LLM-as-a-Judge in Healthcare: A Scoping Analysis of Applications, Methods, and Human Alignment.* arXiv:2605.25273 (May 2026).
6. C. Li, Z. Akhtar, M. Kwak, Y. Ji, H. Zhang, T. Obi, Y. Ren, X. Wu, S. Sivarajkumar, H. P. Lehmann, S. Visweswaran, M. J. Becich, D. L. Mowery, R. Liu, H. Sun, Y. Wang. *A Scoping Review of LLM-as-a-Judge in Healthcare and the MedJUDGE Framework.* arXiv:2604.25933 (Apr 2026).
7. E. Croxford, Y. Gao, E. First, N. Pellegrino, M. Schnier, J. Caskey, M. Oguss, G. Wills, G. Chen, D. Dligach, M. M. Churpek, A. Mayampurath, F. Liao, C. Goswami, K. K. Wong, B. W. Patterson, M. Afshar. *Evaluating clinical AI summaries with large language models as judges.* npj Digital Medicine 8, 640 (2025). doi:10.1038/s41746-025-02005-2. Key result: GPT-o3-mini ICC 0.818 (95% CI 0.772–0.854) against the PDSQI instrument.
8. D. J. Wu, F. N. Haredasht, D. Wu, V. Ravi, L. G. McCoy, Y. Weng, K. Chopra, S. Everett, G. Nageeb, W. Chen, S. Ma, S. K. Maharaj, J. Tran, L. Rosengaus, L. Giang, O. Jee, E. Goh, J. H. Chen. *Automated Evaluation of Large Language Model Response Concordance with Human Specialist Responses on Physician-to-Physician eConsult Cases.* medRxiv 2025.08.14.25332839 (posted 16 Aug 2025). doi:10.1101/2025.08.14.25332839. Key results: DeepSeek R1 κ = 0.75 [0.47, 0.90] vs physician majority; physician Fleiss κ = 0.456, n = 32.
9. A. DeLucia, H. Huang, S. Joshi, M. Yarmohammadi, A. Hassoon, M. Dredze. *Same Verdict, Different Reasons: LLM-as-a-Judge and Clinician Disagreement on Medical Chatbot Completeness.* arXiv:2604.16383 (Mar 2026).
10. B. Sun, Z. Wang, O. de la Torre, Z. Wang. *When Metrics Disagree: Automatic Similarity vs. LLM-as-a-Judge for Clinical Dialogue Evaluation.* arXiv:2603.00314 (Feb 2026).
11. R. K. Arora, J. Wei, R. Soskin Hicks, et al. *HealthBench: Evaluating Large Language Models Towards Improved Human Health.* arXiv:2505.08775 (May 2025).
12. H. Oss Boll, A. Oss Boll, L. Puttlitz Boll, A. Abu Hanna, I. Calixto. *DistillNote: Toward a Functional Evaluation Framework of LLM-Generated Clinical Note Summaries.* arXiv:2506.16777 (Jun 2025).
13. T. Afonja, T. Olatunji, S. Ogun, N. A. Etori, A. Owodunni, M. Yekini. *Performant ASR Models for Medical Entities in Accented Speech.* Interspeech 2024. arXiv:2406.12387.
14. A. Adedeji, S. Joshi, B. Doohan. *The Sound of Healthcare: Improving Medical Transcription ASR Accuracy with Large Language Models.* arXiv:2402.07658 (Feb 2024).
15. Z. Ellis, J. Joselowitz, Y. Deo, Y. He, A. Kalygina, A. Higham, M. Rahimzadeh, Y. Jia, I. Habli, E. Lim. *WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue.* IWSDS 2026 (oral). arXiv:2511.16544.
16. J. Behling. *MedWER: A Reproducible, Model-Free Evaluation Protocol for Medical Speech Recognition.* arXiv:2609.05728 (Sep 2026).
17. T. Park, I. Medennikov, K. Dhawan, W. Wang, H. Huang, N. R. Koluguri, K. C. Puvvada, J. Balam, B. Ginsburg. *Sortformer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens.* arXiv:2409.06656 (Sep 2024).
18. J. Giorgi, A. Toma, R. Xie, S. S. Chen, K. R. An, G. X. Zheng, B. Wang. *WangLab at MEDIQA-Chat 2023: Clinical Note Generation from Doctor-Patient Conversations using Large Language Models.* ClinicalNLP @ ACL 2023. arXiv:2305.02220.
19. H. Wang, C. Gao, B. Liu, Q. Xu, G. Hussein, M. El Labban, K. Iheasirim, H. Korsapati, C. Outcalt, J. Sun. *Towards Adapting Open-Source Large Language Models for Expert-Level Clinical Note Generation.* arXiv:2405.00715 (Apr 2024).
20. S. Ahmed, F. Y. Sadeque. *Clinical Note Generation From Doctor-Patient Conversations Using Parameter-Efficient Fine-Tuning Large Language Models: Comparative Study.* JMIR Medical Informatics (2026). doi:10.2196/82545.
21. K. Wataoka, T. Takahashi, R. Ri. *Self-Preference Bias in LLM-as-a-Judge.* arXiv:2410.21819 (Oct 2024).
22. J. Ye, Y. Wang, Y. Huang, D. Chen, Q. Zhang, N. Moniz, T. Gao, W. Geyer, C. Huang, P.-Y. Chen, N. V. Chawla, X. Zhang. *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge.* arXiv:2410.02736 (Oct 2024).
23. J. D. Norman, M. U. Rivera, D. A. Hughes. *Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias.* arXiv:2606.19544 (Jun 2026).
24. H. Khalaf, C. Mayrink Verdun, A. Oesterling, H. Lakkaraju, F. du Pin Calmon. *Inference-Time Reward Hacking in Large Language Models.* arXiv:2506.19248 (Jun 2025).
25. K. Nishimura-Gasparian, R. McCarthy, D. Lindner. *Towards Understanding Specification Gaming in Reasoning Models.* arXiv:2605.02269 (May 2026).
26. A. Rezaie Mianroodi, A. Rezaie, N. Grisel Todorov, C. Rakovski, F. Rudzicz. *MedSynth: Realistic, Synthetic Medical Dialogue-Note Pairs.* arXiv:2508.01401 (Aug 2025).

Corrections made during verification: the earlier draft's "7 studies in 2024 to 81 in 2025" growth figure was a search-snippet artefact and is not in either review; the κ = 0.75 and Fleiss κ = 0.46 figures belong to the eConsult preprint [8], not to the npj paper [7], whose result is an ICC of 0.82.

## 3. Study 1: the after-visit note

### 3.1 The decoder, not latency, decides medical-term loss
Table 1 (seven systems, 57 recordings). WER spans 10.3–11.8%; medical-term miss spans 8.4–15.1%. Language-model decoders (MOSS-TD 8.4% [7.1, 9.7]; Canary-Qwen 9.4% [8.0, 10.8]) versus transducer decoders (Nemotron streaming 12.5% [10.9, 14.0]; Parakeet-TDT v3 15.1% [13.4, 16.8]); intervals do not overlap. Running the streaming model offline on the whole recording changes nothing (12.5 vs 12.7). With the recording as the sampling unit (paired bootstrap over 57), MOSS-TD minus streaming is −3.9 points of term miss [−6.3, −1.7] and −1.4 WER [−1.8, −1.0]; Canary-Qwen minus streaming is −2.6 term miss [−4.8, −0.6] at unchanged WER (−0.1 [−0.4, +0.2]), so the decoder effect is separable from overall accuracy. Complaint-tuned hotwords make WER measurably worse (+0.18 [+0.03, +0.33]) and do not change term miss. Replicates on the synthetic ACI corpus (MOSS 6.6% vs Nemotron 8.9%). Diarization: MOSS-TD DER 11.4%, Sortformer + Parakeet 11.1%, ≈1% of words misattributed for both; Nemotron 3 Nano Omni prompted as a transcriber was unusable (27% WER, DER 88%).

### 3.2 Diarization alone does not reach the note; role mapping does
Notes from the diarized transcript with anonymous labels carried the same misattribution rate as notes from the undiarized streaming transcript (0.19 per note on all 57); notes from the human transcript, 0.05. On the held-out set the vanilla pipeline and the anonymous-label two-pass pipeline tie on every note metric (composite −1.0 [−5.9, +4.1]). Mapping S01/S02 to Doctor/Patient before writing was the first change the search accepted (dev 44.3 → 45.7, test 39.6 → 41.8) and, in every configuration that carries a role map (the search's best on either writer build, and the Sortformer + Parakeet + Qwen second baseline), misattributions are 0.03 per note versus 0.08–0.11 without one; no single interval resolves it (best − vanilla −0.05 [−0.16, +0.05]).

### 3.3 What an overnight search found, with intervals
131 iterations; gate = dev ≥ best − 0.1 → held-out test must beat best → judge 2 within 0.5. Accepted: role map (+0.9 dev), DeepSeek writer (+1.6), complaint-tuned hotwords + a precision-first instruction (+0.8). On the build the search ran on, held-out composite was vanilla 41.0, best 45.3, +4.2 [−0.3, +8.8] under judge 1 and +2.5 [−0.1, +5.3] under judge 2, with term precision (+4.5 [+2.3, +6.7]) and ROUGE-L (+2.2 [+1.0, +3.2]) resolved. **On the official release weights of the same writer (Table 2)** the best configuration scores 42.3: +1.3 [−3.8, +6.2] under judge 1 and −0.7 [−4.5, +3.0] under judge 2 (which is the writer). Term precision (+2.5 [+0.5, +4.5]) and ROUGE-L (+1.5 [+0.3, +2.6]) still resolve, at about half the size; term recall is unresolved and nominally lower (−2.5 [−6.4, +1.6]); plan recall (−8.1 [−13.8, −1.7]) and grounded sentences (−3.4 [−6.6, −0.2]) are resolved losses. Two second baselines the search never proposed: Parakeet-TDT v3 into the same untuned writer is indistinguishable from vanilla (−1.0 [−4.6, +2.6]), and Sortformer + Parakeet v3 with the role map into the untuned Qwen writer scores 43.0, +2.0 [−1.0, +5.7] over vanilla and −0.8 [−5.2, +3.5] against the official best, with the same 0.03 misattributions per note and no loss of plan recall or grounding. Every proposer-written prompt variant after the acceptances, transcript correction (six times), the scaffold and the verifier gate gained on dev and failed on test. The search is a benchmark-maximizer by construction; the held-out gate is the useful part, and the build change shows what the gate could not: roughly three of the four points it reported were a property of the modified writer, not of the pipeline.

### 3.4 The objective removed citations
Citations cost ~1 ROUGE-L and buy 4–5 term-precision points on every ACI split, but the composite has no verifiability term, so the search rejected citations eight times (the recall cost is all it could see). Re-scored with grounded/linked on the official weights: cited − uncited best is +6.1 grounded [+3.2, +8.9] and +2.1 precision [+0.0, +4.4] against −0.6 ROUGE-L [−1.7, +0.6] and +4.1 plan recall [−1.1, +9.1]; 78.9% of sentences cited, 74.3% supported by their own cited lines; unsupported sentences 14.8% → 8.7% (search-time build: +5.3 grounded [+3.2, +7.8], 88.8% cited, 84.0% linked, 11.4% → 6.8%). The uncited best is less grounded than vanilla (−3.4 [−6.6, −0.2]); the cited one is more (+2.7 [+0.4, +5.1]). Lesson stated as such: a score without the trust dimension removes the trust feature.

### 3.5 Completeness against the note over-credits guessing
Item-level audit of the held-out plan items: 38 of 153 reference plan items (41 of 154 in the first extraction) were never stated in the conversation (silent test orders, drugs written but not discussed). On spoken items the official-weight writers capture 103/115 (cited) and 100/115 (uncited) against 110/114 for vanilla; on unspoken items 14/38 and 14/38 against 11/41. On the search-time build the cited writer's plan-recall gap was half refusal to invent (unspoken 16 vs 19 vs 11); on the official build the gap is on spoken items, mostly safety-netting sentences, and is a real completeness loss. Four prompt variants aimed at plan recall gained up to +1.3 on dev and none held on test. Recommendation: score completeness against transcript-stated items, which is also closer to a completeness-of-follow-ups dimension as used in clinician-defined rubrics.

### 3.6 Overlap metrics and clinician dimensions diverge under fine-tuning (ACI-Bench)
One epoch of LoRA on 2,804 open pairs, per-encounter paired bootstrap over 120 ACI-Bench encounters (Table 5): ROUGE-L +8.6 [+7.2, +10.0], term recall +4.8 [+2.9, +6.6], term precision +10.2 [+7.9, +12.6]; and, judged by the same model, misattributions per note +0.11 [+0.04, +0.18] and follow-up recall −4.5 [−7.3, −2.0] (97.4 → 92.9). Every interval excludes zero: the tune that wins every overlap metric loses on both clinician dimensions. Citation behaviour was erased (99.6% → 9–16% cited) because the targets carried none. Untuned Nemotron 3 Ultra 550B sits a point below untuned Qwen 27B on ROUGE-L with the fewest misattributions of any base model. Ordering that held: tuning > base quality > size.

### 3.7 A reinforcement-learning pass the judges disagree about
Expert iteration on the local writer (sample 8, keep 2, LoRA, four rounds, reward = composite + grounding + linking with a length band; predictions logged before the first sample). Dev 46.2 → 51.0. Held-out: 41.5 → 40.5 under judge 1, 44.3 → 45.9 under judge 2. Style transferred (99.5% cited, notes 19% shorter, edit effort −25%); content did not (term recall −2). None of the predicted reward-hacking signatures appeared in four rounds. At 20 training transcripts this is memorization, and the sign disagreement between judges is the point.

## 4. Study 2: in-visit decision support on the live transcript
**Setup.** The streaming transcript is replayed on the human utterance timeline; every 20 s of visit time the model (DeepSeek V4 Flash, 220 tok/s) updates a JSON view: differential with evidence and what is missing, the single discriminating question, red flags (tiered), plan stated vs suggested, safety-netting given vs to add, assumptions under test, announced revisions. Scoring against the clinician's note and the timed transcript: time to the reference diagnosis in the top-3 and lead over the moment the GP stated it; suggested questions later asked; red-flag precision; plan suggestions agree/extra/contradict; revisions toward/away; premature complaint; stability; latency. 57 visits, 1,566 ticks per version.

**Versions.** v2 first prompt; v4 danger-first + consult modality + simple causes; v5 five-slot differential with a time-critical tag, flags only for stated features with a tier, a 90 s hold enforced in code; v5+NHS with guideline pages retrieved per differential entry (NHS A-Z → patient.info → Wikipedia low-trust; 191 lookups).

**Results (Table 3).** Diagnosis ever in top-3: v2 52/57 [81, 96], v5 56/57 [91, 100]; v5 − v2 +0.07 [+0.02, +0.14]. Lead over the GP: median 5:36, in 54/54 visits where the GP stated a diagnosis. Red flags: v2 61 raised / 7 false; v5 3 raised / 0 false; per-visit −1.02 [−1.44, −0.63] flags and −0.12 [−0.23, −0.04] false alarms. Contradicting plan suggestions: 13 → 3, −0.18 [−0.33, −0.04] per visit. v4's danger-first rule raised agreeing suggestions (+0.79 [+0.39, +1.21] per visit) but evicted the likely diagnosis (top-3 −0.05 [−0.12, +0.02]) and produced hedged flags. Guideline retrieval (v5+NHS − v5): no resolved change on any measure; suggestions +31 mostly "extra"; the NHS anaphylaxis page re-imported emergency advice on an insect-sting visit the prompt rules had calmed. Re-running v5 + retrieval with the official release weights of the same model reproduces every resolved v5 − v2 effect at the same size (Table 3, last column), and differs from the community-build run on no measure.

**Cross-judging (Table 3b).** The v5 run was re-judged item by item, with minimal context per verdict, by DeepSeek and by Gemma 4. Diagnosis coverage falls from 56/57 under the whole-timeline prompt to 51/57 under both item judges, which agree with each other (κ = +0.63) and disagree with the timeline prompt (κ = +0.26 each): the timeline prompt over-credits. Plan-suggestion verdicts depend on the prompt and on whether the judge wrote the views: agree / extra / contradict is 26 / 58 / 3 under the timeline prompt, 67 / 5 / 15 under DeepSeek's item prompt, 21 / 65 / 1 under Gemma's and 32 / 54 / 1 under Qwen's. Gemma and Qwen, neither of which generated the views, agree with each other (three-way κ = +0.72; on contradict-vs-not, +1.00) and both disagree with DeepSeek's item verdicts (κ 0.10 and 0.15), which credit 67 suggestions as agreeing with the clinician's plan where the other two credit 21 and 32: the writer is lenient to itself. Coverage under Qwen's item prompt is 47 / 57 (κ +0.71 with each other item judge). The official-weight run repeats the pattern: the two independent item judges put coverage at 50 and 48 / 57 (κ +0.85 with each other, +0.23 and +0.17 with the timeline prompt's 56), call its single flag genuine, and classify its 118 suggestions as 21 / 96 / 1 and 40 / 76 / 2 (κ +0.59) against 29 / 89 / 0 from the timeline prompt. Of the three red flags, all three features were stated in the transcript by the time they were raised; Gemma accepts all three, DeepSeek's and Qwen's item judges accept one and reject the "act now" tier on the other two. **Reading.** The safety improvements are prompt rules (stated-feature-only flags, a hold, tiering) and their per-visit effects are resolved at n = 57 under the original judge; the reduction in raised flags (61 → 3) is a property of the generator and does not depend on any judge. What does depend on the judge is the classification of suggestions, so the "3 contradictions" figure is reported as one prompt's verdict, not a fact about the pipeline; the two independent judges put it at 1 of 87. "Ahead of the GP" measures how fast the view reads the history, not diagnostic insight.

## 5. What the judges can and cannot see
- **Ranking of notes, yes; of two close pipelines, no.** Per-note composites from the two judges correlate at ρ = 0.80 over 74 notes (search-time build). On the official weights the same two judges put best − vanilla at +1.3 and −0.7, neither resolved: when the difference is small the judges do not even agree on its sign.
- **Levels, no.** Judge 2 scores every note 2–4 points higher; plan recall means differ by 5 points (ρ = 0.54).
- **Attribution, not at all.** On the same 74 notes, Qwen flags 4, DeepSeek 1, Gemma 4 flags 13; pairwise κ −0.02 (Qwen–DeepSeek), +0.29 (Qwen–Gemma), +0.12 (DeepSeek–Gemma); four notes are flagged by two judges and none by all three. Gemma also reverses the direction of the pipeline difference (vanilla 0.135 vs best 0.216 per note, against 0.08 vs 0.03 under Qwen). The misattribution penalty carries 40 points per event in the composite, so this is the term most exposed to judge choice, and the one dimension where the paper can make no directional claim at all.
- **Self-judging and prompt granularity.** In Study 2 the writer and the judge are the same model. Re-judging item by item with a second model changes the verdict distributions more than changing the model does (plan verdicts: κ 0.08–0.28 across prompts, 0.10 across judges at the same prompt), while diagnosis coverage is stable across judges at the item level (κ = 0.63) and inflated by the whole-timeline prompt (56 vs 51 of 57).
- **Objective design.** Two failures were caused by the metric, not the model: citations removed (3.4) and unspoken plan items credited (3.5). Both were found by re-scoring, not by a new model.
- **The search reward was partly the build.** The held-out gate protected against dev overfitting (every late prompt variant failed it) but not against a gain that was a property of the writer build under test: the modified build wrote longer, more complete notes (term recall +3.1 over the official build) and the composite paid for that. A gate cannot see what is constant across every candidate.
- **RL sign disagreement** (3.7).
- Implication: use LLM judges to order pipelines and to audit objectives; calibrate at least the attribution judge against clinicians before any absolute claim.

## 6. Limitations
Mock consultations, UK primary care, 57 recordings; terse reference notes make absolute overlap scores low by construction; no chart context; judges uncalibrated; the vanilla pipeline is an unoptimized open baseline, not any vendor's product; DeepSeek is both writer and judge 2 for the best configuration; the search itself was run on a community-modified build of the writer and only its accepted configurations were re-run on the official weights, so a configuration the official build would have accepted and the modified build rejected is not visible; the live view is a replay, and consult modality is inferred; the plan-recall audit and the live scoring use the same judge that generated the outputs.

## 6a. Data licensing and ethics
PriMock57 is released under CC BY 4.0 (repository LICENSE.md); ACI-Bench under CC BY 4.0 with a citation requirement (Yim et al. 2023); MTS-Dialog under CC BY 4.0; MedSynth's repository and dataset card state no license as of 13 Sep 2026; the harness uses only its term strings (facts, not creative expression) to build the lexicon, and we will ask the authors for a licence statement before release. The lexicon derived from MTS-Dialog and MedSynth is released under CC BY 4.0 with attribution to both. Guideline pages fetched for the live view come from nhs.uk, whose content is released under the Open Government Licence v3 with the attribution "Contains information from NHS England, licensed under the current version of the Open Government Licence"; patient.info professional articles were read at run time and are not redistributed; Wikipedia extracts are CC BY-SA and are used only as model context. All consultations are role-played by clinicians and actors; no patient data, no protected health information, no clinical use. No IRB review was required.

## 7. Reproducibility
Code, lexicon, splits, prompts, run notebooks and cached model outputs are in the repository; every table's source file is listed in [[sources]]. All models are open weights; the two workstation GPUs and about $49 of cloud time suffice to rerun everything except the 550B base-model run.

## Tables

**Table 1. Transcript layer, 57 recordings.** WER after symmetric normalization; medical-term miss over 1,741 lexicon occurrences (binomial 95% CI); DER with 250 ms collar.

| System | Decoder | WER % | Term miss % [95% CI] | DER % |
|---|---|---|---|---|
| Nemotron-3.5 streaming 0.6B, 1.1 s chunks | transducer | 11.8 | 12.5 [10.9, 14.0] | no speakers |
| Nemotron-3.5 offline | transducer | 11.8 | 12.5 | no speakers |
| Parakeet-TDT 0.6B v3 | transducer | 11.2 | 15.1 [13.4, 16.8] | no speakers |
| Sortformer + Parakeet v3 | transducer | 11.2 | 15.1 | 11.1 |
| Canary-Qwen 2.5B | LLM | 11.6 | 9.4 [8.0, 10.8] | no speakers |
| MOSS-TD 0.9B | LLM | 10.3 | 8.4 [7.1, 9.7] | 11.4 |
| MOSS-TD + complaint hotwords | LLM | 10.6 | 8.6 [7.2, 9.9] | 11.7 |
| Nano Omni 30B-A3B as prompted | LLM | 27.2 | 15.7 | 88.3 |

Paired over recordings: MOSS-TD − streaming term miss −3.9 [−6.3, −1.7], WER −1.4 [−1.8, −1.0]; Canary-Qwen − streaming term miss −2.6 [−4.8, −0.6], WER −0.1 [−0.4, +0.2]; hotwords − plain WER +0.18 [+0.03, +0.33].

**Table 2. After-visit note, 37 held-out consultations, judge 1, writer on official release weights.** Means, and paired bootstrap 95% CI of the difference from vanilla. Vanilla = Nemotron streaming → untuned Qwen3.8-27B. Parakeet = Parakeet-TDT v3 → same writer. Two-pass Qwen = Sortformer + Parakeet v3 → role map → same writer. v1 = MOSS-TD with anonymous labels → same writer. Best / cited = the search's accepted configuration with DeepSeek V4 Flash (official build) as writer.

| Metric | Vanilla | Parakeet | Two-pass Qwen | v1 | Best | Best, cited | Two-pass Qwen − vanilla | Best − vanilla | Cited − vanilla |
|---|---|---|---|---|---|---|---|---|---|
| Composite | 41.0 | 40.0 | 43.0 | 40.0 | 42.3 | 42.2 | +2.0 [−1.0, +5.7] | +1.3 [−3.8, +6.2] | +1.2 [−3.8, +6.3] |
| Composite + 0.2 grounded | 58.7 | 57.7 | 61.0 | 57.9 | 59.3 | 60.4 | +2.3 [−0.8, +6.0] | +0.6 [−4.2, +5.4] | +1.7 [−3.3, +6.9] |
| Term precision | 26.4 | 24.8 | 25.8 | 25.8 | 28.9 | 31.1 | −0.6 [−2.1, +0.8] | **+2.5 [+0.5, +4.5]** | **+4.7 [+2.4, +7.1]** |
| Term recall | 52.1 | 51.6 | 52.9 | 54.4 | 49.6 | 49.8 | +0.8 [−2.2, +3.7] | −2.5 [−6.4, +1.6] | −2.3 [−6.6, +2.2] |
| ROUGE-L | 21.8 | 21.1 | 21.4 | 20.6 | 23.2 | 22.6 | −0.3 [−1.2, +0.6] | **+1.5 [+0.3, +2.6]** | +0.9 [−0.4, +2.0] |
| Plan recall (all reference items) | 80.8 | 79.2 | 81.7 | 78.9 | 72.7 | 76.8 | +0.9 [−3.9, +5.6] | **−8.1 [−13.8, −1.7]** | −4.0 [−10.6, +2.8] |
| Plan recall (transcript-stated items) | 96.5 | – | – | – | 87.0 | 89.6 | – | – | – |
| Misattributions / note | 0.08 | 0.08 | 0.03 | 0.11 | 0.03 | 0.05 | −0.05 [−0.14, 0.00] | −0.05 [−0.16, +0.05] | −0.03 [−0.14, +0.08] |
| Grounded sentences % | 88.6 | 88.6 | 90.1 | 89.8 | 85.2 | 91.3 | +1.4 [−1.0, +3.9] | **−3.4 [−6.6, −0.2]** | **+2.7 [+0.4, +5.1]** |
| Cited / linked % | 0 / – | 0 / – | 0 / – | 0 / – | 0 / – | 78.9 / 74.3 | – | – | – |

Judge 2 (the official DeepSeek build, i.e. the writer judging itself): vanilla 45.1, best 44.4, cited 43.3; best − vanilla −0.7 [−4.5, +3.0], cited − vanilla −1.7 [−5.6, +2.0]; term precision still resolved (+2.5 [+0.6, +4.5] and +4.7 [+2.4, +7.1]). Cited − best (judge 1): grounded **+6.1 [+3.2, +8.9]**, precision +2.1 [+0.0, +4.4], ROUGE-L −0.6 [−1.7, +0.6], plan recall +4.1 [−1.1, +9.1]. Best − two-pass Qwen: composite −0.8 [−5.2, +3.5]; precision **+3.2 [+1.1, +5.3]**, ROUGE-L **+1.8 [+0.7, +2.9]**, plan recall **−9.0 [−15.3, −2.1]**, grounded **−4.9 [−7.7, −2.0]**.

Search-time build of the writer (community-modified), same judge and visits, for the record: best 45.3 (+4.2 [−0.3, +8.8]), cited 44.1 (+3.1 [−1.2, +7.9]); term precision +4.5 [+2.3, +6.7] / +6.5 [+4.2, +8.7]; ROUGE-L +2.2 [+1.0, +3.2]; plan recall +0.6 [−5.3, +6.5] / −6.0 [−12.3, +0.2]; grounded −0.8 [−4.0, +2.4] / +4.6 [+2.3, +6.9]; cited 88.8% / linked 84.0%. Model-free metrics of official minus modified build on the best configuration: term recall −3.1 [−6.3, −0.0], precision −2.0 [−4.2, +0.1], ROUGE-L −0.7 [−1.6, +0.2].

**Table 3. Live view, 57 visits, per version (original judge).**

| Metric | v2 | v4 | v5 | v5 + retrieval | v5 + retrieval, official weights |
|---|---|---|---|---|---|
| Reference diagnosis ever in top-3 | 52 | 49 | 56 | 55 | 56 |
| Ever top-1 | 35 | 31 | 44 | 41 | 39 |
| Had it before the GP said it | 48 / 51 | 49 / 49 | 54 / 54 | 52 / 52 | 53 / 53 |
| Suggested questions later asked | 80.8% of 219 | 93.3% of 149 | 86.9% of 229 | 82.6% of 224 | 86.9% of 199 |
| Red flags raised / judged genuine | 61 / 54 | 62 / 52 | 3 / 3 | 6 / 6 | 1 / 1 |
| Plan suggestions agree / extra / contradict | 37 / 108 / 13 | 82 / 67 / 6 | 43 / 73 / 3 | 47 / 104 / 5 | 34 / 111 / 1 |
| Top-1 flips per visit | 0.19 | 0.37 | 0.75 | 0.63 | 0.58 |

Paired v5 − v2: top-3 +0.07 [+0.02, +0.14]; contradictions −0.18 [−0.33, −0.04]; flags −1.02 [−1.44, −0.63]; false alarms −0.12 [−0.23, −0.04]. v5 + retrieval − v5: no interval excludes zero. The last column repeats v5 + retrieval with the model's official release weights as writer and judge (the other columns used a community-modified build of the same model, see Setup): against v2 the same four effects are resolved with the same sign and size (top-3 +0.07 [+0.02, +0.14]; contradictions −0.21 [−0.39, −0.05]; flags −1.05 [−1.51, −0.65]; false alarms −0.12 [−0.23, −0.04]); against the abliterated v5 + retrieval run no interval excludes zero.

**Table 3b. Live view v5, cross-judged item by item.**

| Quantity | timeline prompt, DeepSeek | item prompt, DeepSeek | item prompt, Gemma 4 | item prompt, Qwen3.8 |
|---|---|---|---|---|
| Diagnosis ever in top-3 | 56 / 57 | 51 / 57 | 51 / 57 | 47 / 57 |
| Flags judged genuine (of 3) | 3 | 1 | 3 | 1 |
| Plan suggestions agree / extra / contradict (87) | 26 / 58 / 3 | 67 / 5 / 15 | 21 / 65 / 1 | 32 / 54 / 1 |

κ, top-3 per visit: timeline vs item-DeepSeek +0.26, timeline vs item-Gemma +0.26, timeline vs item-Qwen +0.15; item-DeepSeek vs item-Gemma +0.63, item-DeepSeek vs item-Qwen +0.71, item-Gemma vs item-Qwen +0.71. κ, plan verdicts (three-way): timeline vs DeepSeek / Gemma / Qwen +0.08 / +0.28 / +0.32; DeepSeek vs Gemma +0.10, DeepSeek vs Qwen +0.15, **Gemma vs Qwen +0.72** (contradict-vs-not +1.00). The two judges that did not write the views agree with each other and not with the writer's own item verdicts.

**Table 4. Attribution judged three ways, 74 held-out notes.**

| Judge | Notes flagged | Vanilla mean / note | Best mean / note |
|---|---|---|---|
| Qwen3.8-27B | 4 | 0.08 | 0.03 |
| DeepSeek V4 Flash | 1 | 0.03 | 0.00 |
| Gemma 4 26B-A4B | 13 | 0.135 | 0.216 |

Pairwise κ: −0.02, +0.29, +0.12; flagged by all three: 0. Composite ranking, Qwen vs DeepSeek: Spearman ρ = 0.80 (n = 74).

**Table 5. Fine-tuning, ACI-Bench, 120 encounters, SFT − base (paired bootstrap 95% CI).**

| Metric | SFT − base |
|---|---|
| ROUGE-L | +8.6 [+7.2, +10.0] |
| Term recall | +4.8 [+2.9, +6.6] |
| Term precision | +10.2 [+7.9, +12.6] |
| Misattributions / note | +0.11 [+0.04, +0.18] |
| Follow-up recall | −4.5 [−7.3, −2.0] |

## Figures and tables (list)
1. Table 1: ASR matrix (WER, term miss with CIs, DER, speakers) — from [[decoder-finding]].
2. Figure 1: term miss vs WER scatter, decoder family coloured — drawn, below.
3. Table 2: held-out note metrics for vanilla / v1 / best / cited with bootstrap CIs — from `runs/paper_stats.md`.
4. Figure 2: the search's dev-vs-test trajectory — drawn, below.
5. Table 3: live-view versions with CIs — from `runs/paper_stats.md`.
6. Figure 3: per-visit red flags and contradictions, v2 vs v5 — drawn, below.
7. Table 4: three-judge agreement (attribution). 8. Table 5: ACI-Bench fine-tuning with per-encounter CIs. 9. Table 3b: live-view cross-judging.
8. Appendix A: run log ([[live-experiments]], [[autoresearch-loop]]); Appendix B: prompts; Appendix C: verifier benchmark ([[verifier]]); Appendix D: fairness ([[fairness]]).

## Figure 1. Medical-term miss rate against word error rate, seven open ASR systems
<div style="overflow-x:auto"><svg viewBox="0 0 560 340" role="img" aria-label="Medical-term miss rate against word error rate for seven ASR systems; language-model decoders sit below classic decoders at similar WER" style="max-width:100%;height:auto;font-family:IBM Plex Mono,Menlo,monospace;font-size:11px">
<line x1="52" y1="296.0" x2="544" y2="296.0" stroke="currentColor" stroke-opacity=".15"/><text x="46" y="300.0" text-anchor="end" fill="currentColor">7%</text>
<line x1="52" y1="233.8" x2="544" y2="233.8" stroke="currentColor" stroke-opacity=".15"/><text x="46" y="237.8" text-anchor="end" fill="currentColor">9%</text>
<line x1="52" y1="171.6" x2="544" y2="171.6" stroke="currentColor" stroke-opacity=".15"/><text x="46" y="175.6" text-anchor="end" fill="currentColor">11%</text>
<line x1="52" y1="109.3" x2="544" y2="109.3" stroke="currentColor" stroke-opacity=".15"/><text x="46" y="113.3" text-anchor="end" fill="currentColor">13%</text>
<line x1="52" y1="47.1" x2="544" y2="47.1" stroke="currentColor" stroke-opacity=".15"/><text x="46" y="51.1" text-anchor="end" fill="currentColor">15%</text>
<text x="134.0" y="310" text-anchor="middle" fill="currentColor">10%</text>
<text x="298.0" y="310" text-anchor="middle" fill="currentColor">11%</text>
<text x="462.0" y="310" text-anchor="middle" fill="currentColor">12%</text>
<text x="298" y="334" text-anchor="middle" fill="currentColor">word error rate</text>
<text transform="rotate(-90)" x="-156" y="14" text-anchor="middle" fill="currentColor">medical-term miss rate</text>
<circle cx="201.2" cy="258.4" r="5" fill="#1E7B4F"/>
<text x="209.2" y="262.4" text-anchor="start" fill="currentColor">MOSS-TD +150 hotwords</text>
<circle cx="189.8" cy="252.8" r="5" fill="#1E7B4F"/>
<text x="197.8" y="280.8" text-anchor="start" fill="currentColor">MOSS-TD plain</text>
<circle cx="224.2" cy="247.5" r="5" fill="#1E7B4F"/>
<text x="232.2" y="299.5" text-anchor="start" fill="currentColor">MOSS-TD complaint hotwords</text>
<circle cx="389.8" cy="215.4" r="5" fill="#1E7B4F"/>
<text x="397.8" y="219.4" text-anchor="start" fill="currentColor">Canary-Qwen 2.5B</text>
<circle cx="427.6" cy="126.1" r="5" fill="#8A5A0F"/>
<text x="419.6" y="130.1" text-anchor="end" fill="currentColor">Nemotron 3.5 offline</text>
<circle cx="425.9" cy="119.0" r="5" fill="#8A5A0F"/>
<text x="417.9" y="147.0" text-anchor="end" fill="currentColor">Nemotron 3.5 streaming</text>
<circle cx="332.4" cy="43.7" r="5" fill="#8A5A0F"/>
<text x="340.4" y="47.7" text-anchor="start" fill="currentColor">Sortformer + Parakeet v3</text>
<circle cx="332.4" cy="43.7" r="5" fill="#8A5A0F"/>
<text x="340.4" y="59.7" text-anchor="start" fill="currentColor">Parakeet-TDT v3</text>
<text x="544" y="28" text-anchor="end" fill="currentColor">green = language-model decoder · amber = transducer decoder</text></svg></div>

*Eight configurations of seven systems on the 57 PriMock57 recordings. WER spans about two points; medical-term loss spans about seven. Colour is decoder family.*

## Figure 2. Overnight search: dev composite per iteration, held-out test where run
<div style="overflow-x:auto"><svg viewBox="0 0 760 300" role="img" aria-label="Overnight search: dev-set composite per iteration, held-out test composite where run, accepted points marked" style="max-width:100%;height:auto;font-family:IBM Plex Mono,Menlo,monospace;font-size:11px">
<line x1="48" y1="264.0" x2="744" y2="264.0" stroke="currentColor" stroke-opacity=".15"/><text x="42" y="268.0" text-anchor="end" fill="currentColor">36</text>
<line x1="48" y1="202.0" x2="744" y2="202.0" stroke="currentColor" stroke-opacity=".15"/><text x="42" y="206.0" text-anchor="end" fill="currentColor">40</text>
<line x1="48" y1="140.0" x2="744" y2="140.0" stroke="currentColor" stroke-opacity=".15"/><text x="42" y="144.0" text-anchor="end" fill="currentColor">44</text>
<line x1="48" y1="78.0" x2="744" y2="78.0" stroke="currentColor" stroke-opacity=".15"/><text x="42" y="82.0" text-anchor="end" fill="currentColor">48</text>
<line x1="48" y1="16.0" x2="744" y2="16.0" stroke="currentColor" stroke-opacity=".15"/><text x="42" y="20.0" text-anchor="end" fill="currentColor">52</text>
<text x="48.0" y="278" text-anchor="middle" fill="currentColor">0</text>
<text x="234.3" y="278" text-anchor="middle" fill="currentColor">20</text>
<text x="430.3" y="278" text-anchor="middle" fill="currentColor">40</text>
<text x="626.4" y="278" text-anchor="middle" fill="currentColor">60</text>
<text x="396" y="296" text-anchor="middle" fill="currentColor">iteration</text>
<polyline points="48.0,136.0 57.8,113.0 67.6,130.9 77.4,113.0 87.2,133.5 97.0,139.1 106.8,176.9 116.6,206.3 126.4,206.5 136.2,133.6 146.0,197.5 155.8,88.9 165.6,153.9 175.4,126.2 185.2,130.5 195.0,96.4 204.8,98.0 214.6,86.4 224.5,89.0 234.3,122.2 244.1,125.6 253.9,84.0 263.7,88.7 273.5,126.4 283.3,135.5 293.1,142.3 312.7,84.2 322.5,85.8 332.3,84.2 342.1,84.0 351.9,88.7 361.7,121.7 371.5,71.0 381.3,73.0 391.1,116.4 400.9,113.0 410.7,117.2 420.5,75.1 430.3,84.0 440.1,111.3 449.9,89.0 459.7,114.3 469.5,113.0 479.3,165.6 489.1,87.6 498.9,71.2 508.7,91.9 518.5,185.4 528.3,72.3 538.1,124.7 547.9,64.1 557.7,137.2 567.5,66.2 577.4,95.7 587.2,134.7 597.0,113.3 606.8,129.6 616.6,103.4 626.4,106.1 636.2,102.0 646.0,89.5 655.8,84.8 665.6,132.6 675.4,94.9 685.2,108.8 695.0,142.8 704.8,118.5 714.6,126.8 724.4,74.6 734.2,81.9 744.0,102.2" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-width="1"/>
<line x1="48.0" y1="136.0" x2="48.0" y2="207.9" stroke="#1E7B4F" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="48.0" cy="207.9" r="4" fill="#1E7B4F"/>
<circle cx="48.0" cy="136.0" r="4" fill="none" stroke="#1E7B4F" stroke-width="2"/>
<line x1="57.8" y1="113.0" x2="57.8" y2="174.3" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="57.8" cy="174.3" r="4" fill="#B3261E"/>
<line x1="77.4" y1="113.0" x2="77.4" y2="174.3" stroke="#1E7B4F" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="77.4" cy="174.3" r="4" fill="#1E7B4F"/>
<circle cx="77.4" cy="113.0" r="4" fill="none" stroke="#1E7B4F" stroke-width="2"/>
<line x1="155.8" y1="88.9" x2="155.8" y2="134.6" stroke="#1E7B4F" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="155.8" cy="134.6" r="4" fill="#1E7B4F"/>
<circle cx="155.8" cy="88.9" r="4" fill="none" stroke="#1E7B4F" stroke-width="2"/>
<line x1="332.3" y1="84.2" x2="332.3" y2="201.7" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="332.3" cy="201.7" r="4" fill="#B3261E"/>
<line x1="342.1" y1="84.0" x2="342.1" y2="131.2" stroke="#1E7B4F" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="342.1" cy="131.2" r="4" fill="#1E7B4F"/>
<circle cx="342.1" cy="84.0" r="4" fill="none" stroke="#1E7B4F" stroke-width="2"/>
<line x1="371.5" y1="71.0" x2="371.5" y2="130.5" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="371.5" cy="130.5" r="4" fill="#B3261E"/>
<line x1="381.3" y1="73.0" x2="381.3" y2="150.9" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="381.3" cy="150.9" r="4" fill="#B3261E"/>
<line x1="420.5" y1="75.1" x2="420.5" y2="131.9" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="420.5" cy="131.9" r="4" fill="#B3261E"/>
<line x1="430.3" y1="84.0" x2="430.3" y2="131.2" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="430.3" cy="131.2" r="4" fill="#B3261E"/>
<line x1="498.9" y1="71.2" x2="498.9" y2="124.3" stroke="#1E7B4F" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="498.9" cy="124.3" r="4" fill="#1E7B4F"/>
<circle cx="498.9" cy="71.2" r="4" fill="none" stroke="#1E7B4F" stroke-width="2"/>
<line x1="528.3" y1="72.3" x2="528.3" y2="159.4" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="528.3" cy="159.4" r="4" fill="#B3261E"/>
<line x1="547.9" y1="64.1" x2="547.9" y2="168.8" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="547.9" cy="168.8" r="4" fill="#B3261E"/>
<line x1="567.5" y1="66.2" x2="567.5" y2="199.1" stroke="#B3261E" stroke-width="1.5" stroke-dasharray="2 2"/>
<circle cx="567.5" cy="199.1" r="4" fill="#B3261E"/>
<text x="744" y="28" text-anchor="end" fill="currentColor">grey: dev composite (20 visits) · dots: held-out test (37) · green = accepted, red = dev gain that failed test</text></svg></div>

*Grey line: dev-set composite (20 visits) for each of 71 logged iterations. Dots: held-out test composite (37 visits) for the 14 iterations that earned a test run; green = accepted, red = a dev gain that failed the gate. Five acceptances, all before iteration 50.*

## Figure 3. Per-visit red flags and contradicting suggestions, first prompt vs v5
<div style="overflow-x:auto"><svg viewBox="0 0 760 260" role="img" aria-label="Per-visit red flags raised, first prompt versus v5, 57 visits; contradicting plan suggestions marked below" style="max-width:100%;height:auto;font-family:IBM Plex Mono,Menlo,monospace;font-size:11px">
<rect x="41.0" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="46.6" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="53.4" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="58.9" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="65.7" y="156.0" width="5.6" height="56.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="71.3" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="78.1" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="83.6" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="90.4" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="96.0" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="102.8" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="108.3" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="115.1" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="120.7" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="127.5" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="133.0" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="139.8" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="145.4" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="142.5" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×</text>
<rect x="152.2" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="157.7" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="164.5" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="170.1" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="176.9" y="156.0" width="5.6" height="56.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="182.4" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="179.6" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×</text>
<rect x="189.2" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="194.8" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="201.6" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="207.1" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="213.9" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="219.5" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="226.3" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="231.8" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="229.0" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×××</text>
<text x="234.5" y="238" text-anchor="middle" fill="#B3261E" font-size="9">×</text>
<rect x="238.6" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="244.2" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="251.0" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="256.5" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="263.3" y="100.0" width="5.6" height="112.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="268.9" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="275.7" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="281.2" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="288.0" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="293.6" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="300.4" y="16.0" width="5.6" height="196.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="305.9" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="312.7" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="318.3" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="325.1" y="72.0" width="5.6" height="140.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="330.6" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="337.4" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="343.0" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="349.8" y="44.0" width="5.6" height="168.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="355.3" y="156.0" width="5.6" height="56.0" fill="#1E7B4F"/>
<rect x="362.1" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="367.7" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="364.8" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×××</text>
<rect x="374.5" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="380.0" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="386.8" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="392.4" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="399.2" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="404.7" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="411.5" y="100.0" width="5.6" height="112.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="417.1" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="414.2" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×</text>
<text x="419.8" y="238" text-anchor="middle" fill="#B3261E" font-size="9">××</text>
<rect x="423.9" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="429.4" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="436.2" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="441.8" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="448.6" y="156.0" width="5.6" height="56.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="454.1" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="460.9" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="466.5" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="463.6" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×</text>
<rect x="473.3" y="72.0" width="5.6" height="140.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="478.8" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="485.6" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="491.2" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="498.0" y="128.0" width="5.6" height="84.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="503.5" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="510.3" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="515.9" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="522.7" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="528.2" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="535.0" y="156.0" width="5.6" height="56.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="540.6" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="547.4" y="128.0" width="5.6" height="84.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="552.9" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="559.7" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="565.3" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="572.1" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="577.6" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="584.4" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="590.0" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="596.8" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="602.3" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="609.1" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="614.7" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="611.8" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×</text>
<rect x="621.5" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="627.0" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="633.8" y="156.0" width="5.6" height="56.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="639.4" y="184.0" width="5.6" height="28.0" fill="#1E7B4F"/>
<rect x="646.2" y="156.0" width="5.6" height="56.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="651.8" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="658.5" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="664.1" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="661.2" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×</text>
<rect x="670.9" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="676.5" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="683.2" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="688.8" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="695.6" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="701.2" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="707.9" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="713.5" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="720.3" y="184.0" width="5.6" height="28.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="725.9" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<rect x="732.6" y="212.0" width="5.6" height="0.0" fill="#8A5A0F" fill-opacity=".55"/>
<rect x="738.2" y="212.0" width="5.6" height="0.0" fill="#1E7B4F"/>
<text x="735.4" y="226" text-anchor="middle" fill="#B3261E" font-size="9">×</text>
<text x="34" y="216.0" text-anchor="end" fill="currentColor">0</text>
<text x="34" y="160.0" text-anchor="end" fill="currentColor">2</text>
<text x="34" y="104.0" text-anchor="end" fill="currentColor">4</text>
<text x="34" y="48.0" text-anchor="end" fill="currentColor">6</text>
<text x="40" y="256" fill="currentColor">57 visits, sorted by id · amber = first prompt (v2), green = v5 · × = contradicting plan suggestions (top row v2, bottom row v5)</text></svg></div>

*Bars: unique red flags raised per visit (amber = v2, green = v5). Crosses: contradicting plan suggestions at the end of the visit (top row v2, bottom row v5). v5 raises flags in two visits, both genuine.*

## To do before submission
- Done 13 Sep: related work, Figures 1–3, per-encounter CIs for the fine-tuning table, a third attribution judge (result: no agreement, so the attribution claims are now stated as unresolved rather than as a two-thirds reduction).
- Related work: drafted and verified (section 2a), all twelve author lists complete.
- Done 13 Sep evening: every DeepSeek-dependent number re-run on the official release weights (Table 2, Table 3 last column, Table 3b), two second baselines (Parakeet → Qwen; Sortformer + Parakeet → role map → Qwen), Qwen as a third item judge of the live view, plan-item audit repeated on the official notes, anonymized release tree (64 MB, datasets by script).
- A clinician read of 20 notes and the six live-view items as a "v2 paper" hook, or as a small appendix if one can be arranged before the deadline.
- Human attribution labels on the 74 notes by two non-clinicians (`/label` page in the demo) for a human–human κ next to the judges' −0.02 to +0.29.
