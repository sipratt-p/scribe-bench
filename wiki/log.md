# Log

All times Pacific. Numbers on this page are pointers; the pages they link to carry the exact values.

## 11 Sep 2026
- Architecture brief: existing NVIDIA stack, frontier design, sequence diagram, fine-tuning map, open-weight SOTA. Published as the artifact ([[deliverables]]).
- Harness built (`~/src/scribe-bench`, mirror on the beast). PriMock57 export with per-speaker timestamps; ACI-Bench loader; 7,092-term lexicon ([[datasets]]).
- ASR matrix on PriMock57: MOSS-TD, Nemotron streaming (offline and 1.1 s chunks), Parakeet v3, Sortformer+Parakeet, Canary-Qwen, Nano Omni ([[models]], [[decoder-finding]]).
- Note generation on ACI-Bench and PriMock57 with Qwen3.8-27B; citation variant; verifier with injected errors ([[verifier]]).
- RunPod 2×H200 (~$36): Qwen 27B LoRA SFT on 2,804 pairs; Nemotron Nano attempts (NaN on Blackwell locally); Ultra 550B base on the Mac Studio ([[fine-tuning-findings]], [[infrastructure]]).
- Abridge-dimension judges (attribution, completeness, fairness) reproduced with Gemma 4 ([[judges]], [[fairness]]).
- Second pod (1×H200, ~$13): Nemotron Nano LoRA with small adapter.
- Demo UI built: cached + live modes, explain toggle, learnings section ([[deliverables]]).
- DeepSeek V4 Flash added as a note model; Flash-Next dropped (serving crashes) ([[models]]).
- Overnight autoresearch loop started (~23:00) with Qwen 27B screening judge and DeepSeek confirming judge ([[autoresearch-loop]]).

## 12 Sep 2026
- 08:30 target stop line for the loop was extended; loop keeps running to 13 Sep 08:30. Best accepted at iteration 47 (dev 48.44 / test 45.01 / judge-2 46.8); nothing accepted since.
- Synthetic ACI-Bench audio (Kokoro) confirms the decoder trend on a second corpus ([[decoder-finding]]).
- /experiments page added to the demo; contrast fixed; plain-language guide added.
- Vanilla pipeline scored on the loop composite (dev 44.17 / test 40.82) ([[vanilla-vs-best]]).
- Artifact updated with loop outcome and interviewer phrasing; then corrected from the 22 Jul Abridge×NVIDIA workshop summary ([[abridge]]).
- Verifiability re-score (grounded / linked) of vanilla, v1, best, best+cite ([[citations-and-verifiability]]).
- Four cited prompt variants for plan recall: none held on test. Item-level gap analysis: 41 of 154 reference plan items never spoken ([[plan-recall-gap]]).
- Expert-iteration RL exercise queued behind plan-recall, predictions logged first; round 0 and round 1 done by 14:30 ([[expert-iteration]]).
- Baseline renamed "vanilla pipeline" across artifact (v11) and demo.
- Vanilla judge-2 composite (44.36) and transcript-stated plan recall (96.5%) computed; margin over vanilla restated as 2.4–4.2 depending on judge ([[vanilla-vs-best]]).
- This wiki written.
