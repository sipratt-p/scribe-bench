# Live decision support during the visit (separate experiment)

**Status: built 12 Sep evening, separate from the loop and the results tables.** Nothing here feeds the composite, the head-to-head, or the demo's two main columns. Code: `demo/livesynth.py`, `demo/live.html`; sessions logged to `runs/live_synth/`.

## What it does
Pretend the doctor is talking to the patient right now, with weights we run locally.
- **Replay on the visit timeline.** A PriMock consultation's audio plays in the browser (1×–8×) while the streaming-ASR transcript (Nemotron-3.5 streaming output, no speaker labels) arrives in ~4-word chunks placed on the human utterance timestamps. This is a replay of real streaming output, not a re-run of the streaming model.
- **Rolling synthesis.** Every N seconds of visit time (default 20) a local model gets the transcript so far plus its previous synthesis and returns JSON: complaint, history so far, findings, plan so far, safety-netting, red flags (max 4), gaps a GP would normally still ask (max 5), terms heard. Runs in a background thread so the transcript never stalls; newly added items are highlighted.
- **Possible mishearings.** Streaming words checked against the 1,333 "medical" lexicon terms (7+ letters, not an English lemma or inflection): a non-word within 2 edits, a real word 1 edit away that is not an inflection, or 2–3 short words that fuse into a term ("met form in" → metformin, "a moxie cillin" → amoxicillin). These are the errors pass 2's LLM decoder exists to fix ([[decoder-finding]]).
- **End of visit.** A final synthesis, then the best after-visit note (the demo's cached loop-best pass-2 note) is shown beside it, its plan items are extracted, and for each one the earliest live snapshot that already contained it is found. Readout: "N of M final plan items were already in a live synthesis before the visit ended", with the first-seen time per item.
- **Model selector**: Qwen3.8-27B dense FP8 (GPU 0) or Qwen3.8-Flash-Next NVFP4 MoE single-GPU (GPU 1, `~/scripts/qwen38fn-ablit-nvfp4-1gpu.sh`, port 8003).

## First run (day1 #01, 16×, Qwen 27B dense, 30 s interval)
7 syntheses. Complaint ("diarrhoea for three days") locked at 30 s. Gaps list evolved sensibly (stool character → hydration → medication history → allergies). Plan stayed empty until the GP stated it near the end, then filled with nine items (gastroenteritis, conservative management, no antibiotics, fluids, Dioralyte, paracetamol, time off work, follow-up in 3–4 days, stool sample if persistent). All 4 final plan items were caught live, first seen at the end because that is when GP plans are said. Latency grew 3.7 → 11.4 s across the visit.

## Measured model speed (beast, single stream, 12 Sep 19:50)
| Model | Prefill | Decode |
|---|---|---|
| Qwen3.8-27B dense FP8, vLLM GPU 0 | 3,780 tok in 0.16 s ≈ 23k tok/s | ≈ 47 tok/s |
| Qwen3.8-Flash-Next NVFP4, 1 GPU (PLE CPU offload) | 3,780 tok in 0.55 s ≈ 6.9k tok/s | ≈ 37 tok/s |
| DeepSeek V4 Flash NVFP4, TP2, DSpark draft, util 0.90, 32k ctx | 3,516 tok in 0.41 s ≈ 8.6k tok/s | ≈ 220 tok/s sustained (600-token output in 2.7 s) |

Prefill is not the bottleneck; a ~500-token JSON synthesis at 47 tok/s is. The single-GPU Flash-Next config is *slower* than dense Qwen (the PLE offload path is a fit-it-at-all config, not a throughput config), so it was torn down. Seth's fastest local model is DeepSeek V4 Flash on both GPUs with the DSpark draft; the evening of 12 Sep the loop was stopped early (nothing accepted since #47), scribe-qwen38 stopped, and DS Flash launched from `~/projects/dsv4-flash-nvfp4-sm120/fraserprice_nop2p.sh` (non-thinking, no-P2P variant; the P2P variant wedges after weight load on this kernel) with `GPU_MEM_UTIL=0.80` so ~19 GB per GPU stays free for the demo's live ASR pass. Launch needs `hf` on PATH (`~/ml-env/bin`), or the script's `set -e` dies on a pip install.

## Second run (day1 #02, 16×, DeepSeek V4 Flash, 20 s interval)
25 syntheses over a 9.2-minute visit, latency 0.5–2.1 s (mean 1.2 s), never fell behind at 16×. Complaint moved from "sore throat" (first 20 s of small talk) to "itchy red skin" at 40 s and held. Plan items appeared at 471 s, 531 s and 554 s of a 554 s visit; **4 of 4 final plan items caught live**, two of them 80 s before the end. Two mishearing flags, both lexicon junk ("mention → mencion"), which led to the third heuristic revision: medical candidates within one edit of an ordinary English word are dropped (1,333 → 1,111 terms).

## Launching DeepSeek V4 Flash beside the demo (12 Sep 20:05, five attempts)
Weights alone are ≈ 83 GB per GPU and the KV cache needs a fixed ≈ 3.4 GiB (DSpark draft + sparse indexer, not context-length driven), so `GPU_MEM_UTIL` 0.80 and 0.88 both fail at profiling, and 0.91 OOM'd by 200 MB during CUDA-graph capture because GPU 0 also carries the desktop (≈ 6.6 GB). Working config: `fraserprice_nop2p_scribe.sh` (copy of the no-P2P script with `--max-num-seqs 16 --max-cudagraph-capture-size 64`), `GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=32768`, whisper_server stopped (it held 2.7 GB on GPU 1). Result: KV 3.55 GiB, 34k cached tokens, ≈ 8 GB free on GPU 1. The demo's audio-import mode then OOM'd (Nemotron's offline attention pad wants 2.8 GiB on top of 6.5 GB), fixed by running pass 1 on the CPU when the GPU has < 14 GB free: 71 s for a 9-minute file on 32 cores, pass 2 (MOSS-TD) 39 s on GPU 1.

## v2 → v3: from scribe-in-a-hurry to decision support (12 Sep 20:30–21:30)
Seth's challenge: "what's the value add of streaming it? shouldn't we be helping with diagnostics?" As first built, the live view was a scribe showing its work early; the after-visit note delivers the same content. The only in-visit outputs that change what the clinician does before the patient leaves are decision support, which is also where commercial scribes are heading . The page now produces, every tick:
- **Working differential** (top 3, likelihood, transcript evidence, what would confirm/exclude), **ask next** (the single question that best separates the top two), **act now** (red flags with the danger and the action), **plan the clinician has stated** vs **plan suggestions** (with a guideline basis), **safety-netting given** vs **to add**; plus the scribe view (history, findings, gaps, terms, mishearings).
- **Proactive lookups** (v3): every new differential entry triggers a background NHS + NICE CKS lookup through the beast's SearXNG (`ssh -N -L 8890:127.0.0.1:8890 beast`; CKS blocks direct fetches, NHS pages fetch fine, CKS contributes search snippets). DeepSeek extracts key questions, red flags, first-line management and safety-netting; cached per condition under `runs/live_synth/lookups/`; the next synthesis gets it as reference material and cites it in `basis`. ~8 s cold, 0 s cached.
- **Activity log**: every task (synthesis tick, lookup, scoring) streamed to the page with visit time, status, detail, wall duration and source links.
- **Scoring at the end**, against the clinician's own note: reference diagnosis and plan extracted from the note; when the GP first said the diagnosis and each plan item (timed transcript); earliest live snapshot with the diagnosis in the top-3 / top-1; whether each suggested question was later asked; red-flag precision; plan suggestions at the end as agrees / extra / contradicts; churn (top-1 flips, vanished items); latency mean and p95.

Example, visit #05 (GP: "?UTI, also need to exclude pregnancy"): live top-1 = UTI from 2:11, GP said it at 7:17; lookups for UTI and diverticulitis landed at 8 s each, PID and ectopic returned nothing usable; plan suggestions at the end: 1 agrees, 1 extra, 1 contradicts. Visit #02 (eczema flare): top-1 from 1:20 vs GP at 7:07; 3 of 4 suggestions agree, none contradict; the "new soaps/detergents?" question was never asked.

Reasoning cost measured: thinking off 214 tok/s, thinking on / low 244 tok/s with ~180 reasoning tokens (+0.5 s per tick, no throughput loss thanks to the DSpark draft). But on the long synthesis prompts thinking starved the JSON output at a 900-token budget, so synthesis runs with thinking off by default (`SCRIBE_LIVE_THINK=1` turns it on with a 2,200 budget).

## Results: 57 visits, v2 prompt (no lookups, no revisions), DeepSeek V4 Flash, 20 s ticks (12 Sep 22:20)
`autoresearch/live_eval.py` → `runs/live_synth/eval_summary.md`; 1,566 snapshots. Judge = the same DeepSeek; diagnosis and plan ground truth = the GP's note; timing ground truth = the human transcript.

| Metric | Value | Reading |
|---|---|---|
| Reference diagnosis ever in the live top-3 | 52 / 57 | misses: impacted ear wax, possible MS, hypothyroid (never said aloud), constipation/PID/STI/UTI, "likely migraine" (#5-09) |
| Ever top-1 | 35 / 57 | |
| Median time to top-3 | 1:20 | |
| Had it before the GP said it | 48 / 51 | median lead **4:54** |
| Suggested questions later asked by the GP | 219 unique, **80.8%** | |
| Red flags raised (unique) | 61 across 26 visits, **88.5%** judged genuine | 7 false alarms in 57 visits |
| Plan suggestions at the end | 158: 37 agree, 108 extra, **13 contradict** | the contradictions cluster in 4 visits (ear wax ×3, rash-with-travel ×3, and singles) |
| Premature complaint (first tick wrong) | 7.0% of visits | |
| Top-1 diagnosis flips per visit | 0.19 | vanished items per visit 1.63 |
| Mishearing flags | 26 total, 57.7% precise against the human transcript | sparse by design |
| Latency | mean 5.1 s, p95 7.3 s (4 visits in parallel) | ~2–3 s single-stream |

Reading
- **The differential is early and mostly right**: in 48 of 51 visits where the GP stated a diagnosis aloud, the view had it in its top-3 first, typically five minutes ahead. The GP's stated diagnosis is often just the working impression the model reaches from the same history, so "ahead" mostly means "the model reads the history faster than the GP finishes taking it"; it is still what an in-visit assistant should do.
- **Questions are useful**: four in five suggested questions were ones the GP went on to ask. That is a measure of relevance, not of adding anything the GP would not have thought of.
- **Red flags: 88% precision** but 7 false alarms over 57 visits is roughly one per eight visits, which is the alert-fatigue number a clinician would push back on first.
- **Plan suggestions are the weak spot**: 68% "extra" (reasonable, not in the note) and 8% contradicting the GP. The contradictions concentrate in the visits where the differential was wrong (ear wax, rash with travel), so plan suggestions inherit differential errors. A wrong differential plus confident guideline suggestions is the harm mode.
- **Stability is fine**: 0.19 top-1 flips per visit and 7% premature complaints; the "never guess from small talk" rule works most of the time.
- Not measured yet: the v3 lookups' and revisions' effect (run again with the v3 prompt), and any clinician's view of the red flags and contradictions.

## v2 → v4 on the same 57 visits (v4 = danger-first, modality-aware, simple-causes-first, re-derive every tick; lookups + revisions on; 12 Sep 23:50)
| Metric | v2 | v4 |
|---|---|---|
| Reference diagnosis ever in live top-3 | 52 / 57 | 49 / 57 |
| Ever top-1 | 35 / 57 | 31 / 57 |
| Median time to top-3 | 1:20 | **0:40** |
| Had it before the GP said it | 48 / 51 | **49 / 49**, median lead 5:39 |
| Suggested questions later asked | 80.8% of 219 | **93.3%** of 149 |
| Red flags: unique / genuine / precision | 61 / 54 / 88.5% | 62 / 52 / **83.9%** (10 false alarms) |
| Plan suggestions agree / extra / contradict | 37 / 108 / 13 | **82 / 67 / 6** |
| Announced revisions toward / away | – | 489 / 20 (of 1,258) |
| Wrong complaint on first tick | 7.0% | 5.3% |
| Top-1 flips per visit | 0.19 | 0.37 |
| Tick latency mean / p95 (4 visits in parallel) | 5.1 / 7.3 s | 10.0 / 10.0 s |
| Guideline lookups actually used | – | 8 (search engine returned empty for most conditions) |

What v4 fixed: the two dangerous cases. Impacted ear wax is now in the top-3 (simple-causes-first) and the rash-with-travel visit no longer suggests watch-and-wait (danger-first). Plan suggestions doubled their agreement with the GP and contradictions fell from 13 to 6, all six in two visits: an insect-bite reaction escalated to anaphylaxis (999, adrenaline), and a headache-with-fever visit chased meningitis and COVID where the GP's note says hypothyroid, a diagnosis never spoken aloud. Suggested questions got fewer and better (93% later asked).

What v4 cost: **danger-first crowds a three-slot differential.** Asthma exacerbation lost its slot to ACS / pericarditis, TIA to cervical radiculopathy / MS; heart failure was still listed second but the judge marked it absent ("exacerbation of heart failure" vs "heart failure"), a judge inconsistency. Red-flag precision fell because the model now raises hedged flags ("thunderclap not excluded", "onset not yet clarified"), which the judge rightly rejects. Top-1 flips doubled because the differential is re-derived every tick. Latency doubled with the longer prompt and output; single-stream on the demo page it is 4–6 s. The lookups barely contributed: SearXNG returned empty result sets for most conditions even serialized, so the v4 gains are the prompt rules, not the guideline material.

Next changes, not run: a five-slot differential so the dangerous entries stop evicting the likely one; red flags only for features actually present, with hedges going to `missing`; a working search path for the lookups (direct NHS site search); and a clinician reading the 62 red flags and 6 contradictions.

## v5 (13 Sep morning): 5-slot differential with a time-critical tag, stated-only tiered flags, 90 s hold, calmer wording
Seth's note after reading v4: "jumping to red flags too early is quite nerve racking for a patient." Three changes: the differential has five slots and time-critical possibilities carry a tag instead of displacing the likely diagnosis; a red flag must quote something actually said and carries a tier ("act now" vs "check today"), with unclarified possibilities going to the differential's `missing` list; and a hold, enforced in code as well as in the prompt, that allows no flags and no urgent suggestions in the first 90 s or before the complaint is established.

| Metric | v2 | v4 | v5 | v5 + NHS lookups |
|---|---|---|---|---|
| Reference diagnosis ever in live top-3 | 52 / 57 | 49 / 57 | **56 / 57** | 55 / 57 |
| Ever top-1 | 35 | 31 | **44** | 41 |
| Had it before the GP said it | 48 / 51 | 49 / 49 | **54 / 54**, lead 5:36 | 52 / 52, lead 5:35 |
| Suggested questions later asked | 80.8% | 93.3% | 86.9% (229) | 82.6% (224) |
| Red flags raised / genuine | 61 / 54 | 62 / 52 | **3 / 3** (2 visits) | 6 / 6 (3 visits) |
| Plan suggestions agree / extra / contradict | 37 / 108 / 13 | 82 / 67 / 6 | 43 / 73 / **3** | 47 / 104 / 5 |
| Revisions toward / away | – | 489 / 20 | 459 / 35 | 503 / 15 |
| Top-1 flips per visit | 0.19 | 0.37 | 0.75 | 0.63 |
| Latency mean / p95 (4 parallel) | 5.1 / 7.3 | 10.0 / 10.0 | 7.1 / 9.6 | 7.1 / 10.6 |

The three v5 flags: "feels like can't breathe that well right now" and "chest is tight" at 3:40–4:00 on the real anaphylaxis visit, and "vaginal bleeding with lower abdominal pain" at 2:00 on the abdominal-pain visit. All judged genuine. The three contradictions: "perform otoscopy" on the ear-wax visit, which is what the GP's plan says in other words (judge over-call), and two on the hives visit where the model calls it likely anaphylaxis and advises against relying on oral antihistamines while the GP gave antihistamines and called an ambulance, a real clinical disagreement rather than an error. The one missed diagnosis is #5-09, "likely migraine", where the GP never stated it aloud either. Cost: top-1 flips doubled, because a five-slot differential reorders more often. With the search path replaced and 191 NHS lookups feeding the ticks (last column), the metrics did not improve: more suggestions, same agreement, two more contradictions from the NHS anaphylaxis page re-importing urgency; see [[live-experiments]] L6.

Sunday demo picks (v5 on the page): #day3-01 (anaphylaxis: the only visit with 'act now' flags, both genuine, at 3:40), #day2-07 (acute cardiac event), #day4-03 (PE), #day2-09 (suspected stroke). Honest failure: #day5-09 (migraine never in the top-3; GP never said it aloud either) and the v4 history of ear wax and the rash-with-travel visit.

## Caveats
- The "live" transcript is real streaming output replayed, not streaming inference; timing is proportional word placement over utterance timestamps.
- Mishearing flags are a heuristic (first version flagged "feeling → peeling"; fixed by treating inflections as English). Sparse by design: the streaming model mostly substitutes real words, which no edit-distance check can see.
- The scoring uses the same local model as a judge; no clinician validation ([[judges]]). Red-flag false alarms and plan contradictions are the numbers a clinician would have to check first.
- Lookups are tiered (NHS page → patient.info professional article → Wikipedia, low trust), matched by name against cached indexes rather than a search engine, see [[live-experiments]]; NICE CKS is unreachable (403). The model's own knowledge still fills gaps and is distinguished from the lookup only by the `basis` tag and the activity-log tier badge.

Sources: `demo/livesynth.py`, `demo/live.html`, `runs/live_synth/*.json`, `runs/demo_server.log`.
