# Open questions and next runs

1. **Plan completeness under citations.** Switch the plan metric to transcript-stated items; re-score the four-config table; then try a plan-specific reward or a "cite the plan lines" second pass. Target: the seven spoken safety-netting items ([[plan-recall-gap]]).
2. **Expert iteration rounds 2–4.** Does the growing dataset pull test back, or keep trading term recall for brevity? Does the judge gap ever go positive? ([[expert-iteration]])
3. **Judge validation.** No clinician has scored any note. A 20-note blind comparison (Qwen judge vs DeepSeek judge vs a clinician) would tell us which judge to trust; Abridge's whitepaper treats this as the step that makes a metric real ([[judges]]).
4. **Nano Omni through its documented SGLang path** with a transcription prompt and real timestamps; the vLLM/eager result measures Omni-as-prompted only ([[models]]).
5. **Nemotron Speech streaming via vLLM**, the model Abridge is actually evaluating, as pass 1 in the demo ([[abridge]]).
6. **Edit effort in the loop.** Measured once (Qwen base 428/100w on ACI; 1047 on PriMock RL round 0) but never optimized; it is the number the doctor feels.
7. **Third-party PHI leakage**: every model leaked; only the verifier catches it. A dedicated leak classifier on edit deltas.
8. **Demo v2 column**: switch to the cited best config (currently uncited loop best) once the plan metric is fixed ([[deliverables]]).
9. **Loop stop line** 13 Sep 08:30: re-judge the accepted config on a sample, aggregate `RESULTS.md`, morning Telegram summary.
