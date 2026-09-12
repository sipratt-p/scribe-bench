# Abridge: what they run and measure

## Public
- Ambient documentation inside Epic (Haiku capture, Hyperspace review); Linked Evidence (highlight a sentence → supporting transcript, on demand); a Contextual Reasoning Engine mixing frontier APIs with their own models; June 2026 announcement to distill/fine-tune/post-train Nemotron on their data.
- **Evaluation whitepaper**: clinician-defined dimensions scored by LLM judges validated against expert review, tracked by specialty and partner, fairness across patient groups. Named dimensions: attribution (four misattribution types), completeness of follow-ups and referrals, consistency across groups. Reproduced here with our own judges ([[judges]], [[fairness]]).

## The 22 Jul 2026 developer workshop (Abridge × NVIDIA Megatron/NeMo team) — corrects earlier assumptions
- **Streaming ASR is a third-party provider today**, not fine-tuned Parakeet on Riva. Riva's container deployment was too complex; **Nemotron Speech streaming via vLLM** is under evaluation as the drop-in. Real-time transcription is "a core dependency".
- **Foundation-model initiative**: mid-training (continued pretraining) of open-weight models on ~12B tokens of internal clinical conversations + ~40B curated external. Milestones: (1) recipe on sub-70B, (2) scale to the 300B+ tier, (3) post-training infra: eval suites, agentic task environments, RL pipelines. **Dense Qwen 3.6 27B responded better than Qwen/Nemotron MoE** to the recipe.
- **Eval**: ~10 internal downstream tasks — note generation from transcripts, order generation (meds, labs), CDI code prediction, pronoun/wellness classification, nursing flowsheet generation. Lightweight SFT+RL on checkpoints before judging them. Directional improvement is enough; **"don't benchmark-max"**.
- **Training infra**: Megatron-Bridge 3× slower than Axolotl for Qwen 3.5 122B full FT on B200 (container 26.06 fix pending; MTP-head export bug).
- **Nemotron Coalition**: NVIDIA asked for clinical data for Nemotron 4, kept confidential.
- **Agentic CDS**: voice-in/voice-out clinical decision support as a cascaded pipeline (ASR → agentic LLM → TTS); longitudinal patient modeling as a future direction.

## Other internal docs shared
Digital Health Adoption Tracker (P0 account; NeMo, Parakeet, Triton, TRT-LLM); Speech/Riva business update ("extensive feedback on Riva, mostly positive"); GTC session proposal (long-form clinical transcription + note summarization). None describe evaluation.

## Mapping to this work
- Plan-item recall ≈ their order-generation / completeness tasks ([[plan-recall-gap]]).
- Generation-time citations + verifier are *additions*; nothing public says they do them ([[citations-and-verifiability]], [[verifier]]).
- Their "don't benchmark-max" is the frame for [[benchmark-maxing-vs-quality]]; our harness is the reward side of their milestone three ([[expert-iteration]]).
- Their dense-beats-MoE result matches our untuned ordering (Qwen 27B > Nano, Ultra) ([[fine-tuning-findings]]).

Sources: `~/Downloads/Abridge_DeveloperWorkshop_7_22_26_Summary.docx` (text: scratchpad `abridge_workshop.txt`), memory `reference_abridge_workshop.md`, artifact sections 1 and 7.
