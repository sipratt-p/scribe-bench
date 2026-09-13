# Glossary: acronyms and jargon

Plain-language definitions for everything used in this wiki. Grouped by where the term shows up. Pages that go deeper are linked.

## Speech-to-text

| Term | Stands for | What it means here |
|---|---|---|
| ASR | Automatic speech recognition | Turning audio into text. "Pass 1" is the fast live version, "pass 2" is the slower after-visit version ([[two-pass-asr]]). |
| WER | Word error rate | Wrong words (substituted + deleted + inserted) divided by words in the human transcript. 10% ≈ one word in ten wrong, mostly fillers. Hides medical-term loss ([[metrics]]). |
| Medical-term miss rate | | Share of medical words in the human transcript that the ASR lost. The number the doctor feels ([[decoder-finding]]). |
| Diarization | | Working out who spoke when. Output is speaker labels like S01/S02 with timestamps. |
| DER | Diarization error rate | Share of speech time given to the wrong speaker, missed, or invented. |
| Misattribution (transcript level) | | Share of words assigned to the wrong person. |
| Decoder | | The part of an ASR model that turns audio features into words. Classic decoders (CTC, RNN-T, TDT) predict tokens frame by frame; an "LLM decoder" is a language model doing that job, so it knows rare words better. |
| CTC / RNN-T / TDT | Connectionist temporal classification / recurrent neural network transducer / token-and-duration transducer | Families of classic, fast ASR decoders. Parakeet and the Nemotron streaming model use TDT. |
| SALM | Speech-augmented language model | NVIDIA's name for ASR built on an LLM decoder (Canary-Qwen). |
| Streaming | | ASR that emits words as the audio arrives, in small chunks (here 1.1 s). Necessarily fast and context-poor. |
| Offline | | ASR run on the whole recording after the fact. |
| VAD | Voice activity detection | Detecting when someone is speaking at all. |
| Hotwords | | A list of words handed to the ASR model to bias it toward them (e.g. drug names). Did not help here. |
| Sortformer | | NVIDIA's neural diarization model. |
| MOSS-TD | MOSS-Transcribe-Diarize | An open 0.9B model that transcribes and diarizes in one pass with an LLM decoder. Our pass-2 pick. |
| Parakeet / Canary / Nemotron ASR | | NVIDIA ASR model families ([[models]]). |
| Nano Omni | Nemotron 3 Nano Omni | NVIDIA's speech+text model; unusable as a transcriber when prompted through vLLM. |
| TTS / Kokoro | Text-to-speech | Used to make synthetic two-voice audio from ACI-Bench dialogues. |
| RTTM / TextGrid | | File formats for speaker timestamps, used to score DER. |

## Notes and note quality

| Term | Stands for | What it means here |
|---|---|---|
| Ambient scribe | | Software that listens to a visit and drafts the clinical note. |
| Reference note | | The note the real clinician wrote; what drafts are scored against. |
| ROUGE-L | Recall-Oriented Understudy for Gisting Evaluation, longest common subsequence | Word-overlap score with the reference note. Rewards fluent, similar wording; blind to who-said-what. |
| BERTScore | | Overlap in meaning rather than exact words, using an embedding model. |
| Term recall / term precision | | Of the medical terms in the reference note, how many we kept (recall); of the terms we wrote, how many are in the reference (precision). Precision is the hallucination guard. |
| Plan items / plan recall | | The actions in a note (prescriptions, tests, referrals, follow-ups, safety-netting). Recall = share a judge can find in the draft. Proxy for a completeness dimension ([[plan-recall-gap]]). |
| Safety-netting | | UK GP term: telling the patient when to come back or call (e.g. "go to A&E if …"). |
| Attribution / misattribution | | Whether a statement is pinned to the right person. Family history written as the patient's own is the classic misattribution. |
| Third-party leak | | Personal information about someone other than the patient (a relative, a colleague) ending up in the note. |
| Grounded | | A note sentence the judge finds supported by the transcript. |
| Citation / span citation | | A note sentence ending with the transcript line numbers that support it, written as double brackets. |
| Linked | | A cited sentence whose *own* cited lines actually support it ([[citations-and-verifiability]]). |
| on-demand evidence linking | | commercial products feature: highlight a note sentence, see the supporting transcript. On demand, not at generation time. |
| Verifier | | A small model that checks each cited claim against its lines and flags unsupported ones or leaks ([[verifier]]). |
| Injected errors | | Deliberate corruptions (flip a "no", change a number, swap a drug, invent a finding, add a leak) used to measure whether the verifier catches them. |
| Edit effort / edits per 100 words | | Character edits needed to turn the draft into the reference note, per 100 reference words. A proxy for how much work the doctor has to do. |
| Length ratio | | Draft words divided by reference words. |
| Role map | | Turning anonymous speaker labels (S01/S02) into Doctor/Patient before the note is written ([[role-mapping]]). |
| Scaffold | | Two-step writing: extract facts first, then write the note from them. Rejected by the loop. |
| Gate (drop_flagged) | | Running the verifier on the draft and deleting sentences it flags. Rejected by the loop. |
| Transcript correction | | Asking an LLM to fix misheard medical terms before note writing. Rejected by the loop four times. |
| Composite | | One blended number: 0.3 term recall + 0.3 term precision + 0.2 ROUGE-L + 0.2 plan recall − 40 × misattributions per note ([[metrics]]). |
| Clinician-defined dimensions | | the vendor evaluation literature term for note-quality measures doctors specified (attribution, completeness, …), scored by LLM judges validated against expert review. |
| Fairness stratification | | Splitting scores by patient gender or age to see if quality differs ([[fairness]]). |

## Models, training, and inference

| Term | Stands for | What it means here |
|---|---|---|
| Open-weight | | A model whose weights you can download and run yourself (Qwen, DeepSeek, Nemotron), as opposed to a frontier API (GPT, Claude). |
| Frontier API / frontier model | | The largest closed models reached over an API. In the design, used only for coding, orders and chart questions. |
| Dense vs MoE | Mixture of experts | Dense: every parameter is used for every token (Qwen 27B). MoE: only a subset of "expert" blocks fires per token (DeepSeek, Nemotron Nano/Ultra).  |
| 30B-A3B, 550B-A55B | | MoE size notation: total parameters, and how many are active per token. |
| Reasoning / thinking mode | | Models that write a hidden chain of thought before answering. Turned off everywhere here (`enable_thinking=false`); it hurt note quality in published tests. |
| Judge / LLM-as-judge | | Using a model to score another model's output against a rubric ([[judges]]). |
| Judge 1 / judge 2 | | Qwen 27B (screening) and DeepSeek V4 Flash (confirming). |
| Judge gap | | Judge 1 score minus judge 2 score on the same notes. Used as a reward-hacking meter. |
| SFT | Supervised fine-tuning | Training a model on example input→output pairs (transcript → note). |
| LoRA | Low-rank adaptation | A cheap way to fine-tune: train small added matrices instead of all weights. "r" is their rank (size). |
| PEFT | Parameter-efficient fine-tuning | The library that implements LoRA. |
| Adapter | | The saved LoRA weights; applied on top of the base model. |
| Merge (and unload) | | Folding the adapter into the base weights so the model runs as one. |
| Epoch | | One pass over the training data. |
| Mid-training / continued pretraining | | Keep pretraining an open model on a domain corpus before task-specific tuning. |
| Post-training | | Everything after pretraining: SFT, RL, preference tuning. |
| RL | Reinforcement learning | Updating a model from a reward signal rather than fixed targets. Not done here except the expert-iteration exercise. |
| Expert iteration / rejection-sampling fine-tuning | | Sample several outputs, keep the best by reward, fine-tune on those, repeat. The cheapest RL-shaped loop ([[expert-iteration]]). |
| GRPO | Group relative policy optimization | A proper RL algorithm for LLMs; the next step if expert iteration had shown signal. |
| Reward hacking | | The policy finding ways to raise the reward that do not raise real quality (shorter notes, passive voice, copying the transcript). |
| KL anchor | Kullback–Leibler | A penalty that keeps the tuned model close to the original so it does not drift. |
| Greedy vs sampling / temperature / top-p | | Greedy: always the most likely next token (deterministic). Sampling with temperature 0.8 and top-p 0.95: random but plausible, used to get eight different candidate notes. |
| Best-of-N | | Generate N candidates, keep the highest-scoring one. |
| Dev / test split | | 20 consultations to search on, 37 held out to check on. Gains that appear on dev and vanish on test are overfitting. |
| Ablation | | Turning one knob off to measure how much it contributed. |
| Benchmark-maxing | | Optimizing the score rather than the quality it is meant to represent ([[benchmark-maxing-vs-quality]]). |
| Autoresearch loop | | The overnight search: propose a config, score it, keep it only if it survives the test set and a second judge ([[autoresearch-loop]]). |
| Proposer | | The LLM that reads the results notebook and suggests the next experiment. |
| Cache by hash | | Every LLM call is keyed by its inputs, so reruns are free. |
| vLLM / SGLang / llama-server / oMLX / mlx_lm | | Inference servers. vLLM and SGLang on the beast's NVIDIA GPUs; llama-server, oMLX and mlx_lm on the Mac's Apple silicon. |
| NeMo / Riva / NIM / Triton / TensorRT-LLM | | NVIDIA's training framework, speech deployment product, packaged inference containers, serving server, and inference engine, respectively. |
| FP8 / NVFP4 / 4-bit | | Quantization levels: fewer bits per weight, smaller and faster, slight quality cost. |
| bf16 | bfloat16 | The standard 16-bit training precision. |
| PLE offload | Per-layer embedding CPU offload | A vLLM trick needed to fit Flash-Next on one GPU. |
| H200 / B200 / RTX PRO 6000 Blackwell | | NVIDIA GPUs: rented data-centre cards (H200, B200) vs the two workstation cards in the beast. |
| Eager mode | | Running a model without compiled graphs; slower but avoids compatibility bugs. |
| MTP heads | Multi-token prediction | Extra prediction heads some models ship for faster decoding. |

## Clinical and data terms

| Term | Stands for | What it means here |
|---|---|---|
| PriMock57 | | 57 mock UK GP consultations with audio, transcripts and notes ([[datasets]]). |
| ACI-Bench | Ambient clinical intelligence benchmark | 120 encounters with transcripts and reference notes; three test splits. |
| MTS-Dialog / MedSynth | | Open dialogue-to-note datasets used for the lexicon and the SFT pairs. |
| Lexicon | | The 7,092 medical terms used to compute term miss, recall and precision. |
| Presenting complaint | | Why the patient came in; used to pick complaint-related hotwords. |
| GP | General practitioner | UK primary-care doctor. |
| A&E / LAS | Accident and emergency / London Ambulance Service | UK terms in the PriMock plan items. |
| FBC / TSH | Full blood count / thyroid-stimulating hormone | Common blood tests that appear as unspoken plan items. |
| CDI | Clinical documentation integrity | Coding the note for billing. |
| EHR / Epic / FHIR | Electronic health record / the dominant EHR vendor / the interoperability standard | Where the note lands and where chart context comes from. Haiku and Hyperspace are Epic's mobile and desktop apps. |
| PHI | Protected health information | Why residency (keeping data inside the health system) matters. |
| BAA | Business associate agreement | The contract a health system needs before sending PHI to a vendor like OpenAI. |
| VPC | Virtual private cloud | The health system's own isolated cloud environment. |
| CDS | Clinical decision support | Tools that answer clinician questions or suggest actions during a visit. |

## In-visit decision support (separate experiment)

| Term | Stands for | What it means here |
|---|---|---|
| Decision support / CDS | Clinical decision support | Output meant to change what the clinician does during the visit, as opposed to a note written afterwards. |
| Differential | Differential diagnosis | The ranked list of conditions that could explain the presentation; here top-3 with evidence and what is missing. |
| Discriminating question | | The one question whose answer best separates the top two diagnoses, or rules out the most dangerous one. |
| Red flag / act now | | A feature that needs urgent action (chest pain with sweating, sudden weakness, anaphylaxis signs). Precision matters more than recall because false alarms cause alert fatigue. |
| Plan suggested vs plan stated | | Guideline-based actions the model proposes, kept apart from actions the GP has actually said. Scored as agrees / extra / contradicts against the GP's note. |
| Assumptions under test | | What the top diagnosis is leaning on and the finding that would overturn it. |
| Revision | | An announced change of belief at a tick, with the transcript evidence that caused it; scored as toward or away from the GP's final diagnosis. |
| Lookup | | A background NHS / NICE CKS search for a new differential entry, distilled into key questions, red flags, first-line management and safety-netting and fed to the next tick. |
| NICE CKS | National Institute for Health and Care Excellence, Clinical Knowledge Summaries | UK primary-care guideline summaries; blocks direct fetches, so only search snippets are used. |
| Tick / snapshot | | One re-synthesis of the live view, every 20 s of visit time by default. |
| Lead time | | How long before the GP stated the diagnosis the live view already had it in its top-3. |
| Activity log | | The page panel listing every task (synthesis, lookup, revision, scoring) with visit time and wall-clock duration. |
| SearXNG | | The self-hosted meta-search engine on the beast that the lookups go through. |
| Tailscale Serve / MagicDNS | | How the demo is exposed to the phone on the private tailnet (http://seth-cosmo-studio/live); HTTPS certificate issuance failed, so plain HTTP. |
