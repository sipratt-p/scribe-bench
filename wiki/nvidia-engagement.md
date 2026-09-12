# What this means for the NVIDIA engagement

- **Riva feedback was about runtime, confirmed.** Riva's container deployment was too complex to adopt; streaming ASR is still a third-party provider; Nemotron Speech through vLLM is the candidate. Megatron-Bridge ran 3× slower than Axolotl on the same B200s ([[abridge]]).
- **The decoder is the ASR argument.** Parakeet-TDT v3 and Nemotron streaming lose 12–15% of lexicon terms; Canary-Qwen and MOSS-TD lose 8–10% at the same WER. Canary-Qwen + Sortformer is the NVIDIA-native pass 2 ([[decoder-finding]]). Getting MOSS-TD-class models into NIM and shipping an Omni transcription profile with real timestamps closes the gap.
- **Their note model is becoming their own** (mid-training dense models). The open-weight delta is gone. Remaining pitch: streaming ASR for pass 1, LLM-decoder checkpoint for pass 2, the Nemotron Coalition data exchange, Megatron-Bridge throughput parity.
- **The verifier is the cheapest win.** A 4B classifier on TensorRT-LLM in front of every draft; every model tested leaked third-party PHI into notes ([[verifier]]).
- **Residency is the sales argument.** Health systems that won't sign an OpenAI BAA can run the whole stack minus the router inside their VPC on NVIDIA hardware.

Sources: artifact section 7.
