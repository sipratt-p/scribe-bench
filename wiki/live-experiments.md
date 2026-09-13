# Live decision support: every experiment, in order

Companion to [[live-synthesis]] (the design and the current numbers). This page is the run log: what each version changed, why, what it scored, and what it taught. All runs: 57 PriMock consultations, streaming-ASR transcript replayed on the human utterance timeline, one synthesis every 20 s of visit time (1,566 ticks per run), DeepSeek V4 Flash on the beast as writer and as judge, ground truth = the GP's note and the timed human transcript. Per-visit files under `runs/live_synth/eval*/`, summaries `runs/live_synth/eval_summary*.md`.

| # | When (12–13 Sep) | Version | What changed | Scored on |
|---|---|---|---|---|
| L0 | 19:30 | v1 scribe-in-a-hurry | first page: complaint / history / findings / plan / safety-netting / gaps / terms; mishearing flags; end-of-visit "which final plan items had the live view already caught" | 2 visits by hand |
| L1 | 20:30 | v2 decision support | differential (top-3, evidence, missing), ask-next, red flags, plan stated vs suggested, safety-netting given vs to add; timeline scoring vs the GP's note | 57 visits |
| L2 | 21:00 | v3 lookups + activity log | NHS + NICE CKS lookup per new differential entry, distilled and fed to the next tick; activity log panel; thinking-low tested | page only |
| L3 | 21:30 | v3 + assumptions/revisions | "assumptions under test" and announced revisions (was → now → because), scored toward/away the GP's diagnosis | page only |
| L4a | 23:00 | v4 (buggy) | danger-first, consult modality + examined, simple causes first; lookups on | 57 visits, discarded |
| L4b | 23:20 | v4 clean | + re-derive the differential every tick (fixes empty-differential anchoring); search calls serialized | 57 visits |
| L5 | 13 Sep 08:00 | v5 | 5-slot differential with time-critical tag; flags only for stated features, tiered act-now / check-today; 90 s hold enforced in code; calmer wording | 57 visits |

## Results side by side
| Metric | v2 (L1) | v4 (L4b) | v5 (L5) |
|---|---|---|---|
| GP's diagnosis ever in live top-3 | 52 / 57 | 49 / 57 | **56 / 57** |
| Ever top-1 | 35 | 31 | **44** |
| Median time to top-3 | 1:20 | 0:40 | 1:00 |
| Had it before the GP said it | 48 / 51 | 49 / 49 | **54 / 54** (lead 5:36) |
| Suggested questions later asked | 80.8% of 219 | **93.3%** of 149 | 86.9% of 229 |
| Red flags raised / genuine | 61 / 54 | 62 / 52 | **3 / 3** |
| Plan suggestions agree / extra / contradict | 37 / 108 / 13 | 82 / 67 / 6 | 43 / 73 / **3** |
| Revisions toward / away | – | 489 / 20 | 459 / 35 |
| Wrong complaint on first tick | 7.0% | 5.3% | 7.0% |
| Top-1 flips per visit | 0.19 | 0.37 | 0.75 |
| Mishearing flags precision (vs human transcript) | 57.7% of 26 | same | same |
| Latency mean / p95, 4 visits in parallel | 5.1 / 7.3 s | 10.0 / 10.0 s | 7.1 / 9.6 s |
| Guideline lookups actually used | – | 8 | 6 |

## What each run taught
- **L0**: the value of streaming a scribe is small; the after-visit note delivers the same content. Complaint declared from small talk at 20 s and corrected at 40 s: early wrong commitment is the failure class. Seth: "shouldn't we be helping with diagnostics?"
- **L1**: the differential is early and mostly right (52/57, five minutes before the GP), but "before the GP said it" mostly means reading the history faster than the GP finishes taking it. 13 contradicting plan suggestions clustered in the two visits where the differential was wrong (impacted ear wax never considered; rash-with-travel told to watch and wait despite dengue and meningitis in its own list). 7 red-flag false alarms in 57 visits.
- **L2**: DeepSeek at 220 tok/s makes 20 s ticks comfortable (1–3 s each single-stream). Thinking-low costs no throughput (DSpark draft) but starved the JSON at a 900-token budget, so off by default. Lookups cost ~8 s cold and are cached per condition.
- **L3**: revisions are announced (1,258 per run in v4, 459 toward vs 20 away), so the view does re-examine earlier beliefs each tick; the eval scores whether each revision moved toward the GP's answer.
- **L4a**: a prompt rule ("keep earlier items") made the model anchor on its own empty first differential for 10 visits. Caught by inspecting the lost visits before reporting; rule rewritten to re-derive the differential every tick. Lesson: check the per-visit files before the summary.
- **L4b**: danger-first fixed both dangerous cases and halved contradictions, but evicted the likely diagnosis from a three-slot differential (asthma → ACS/pericarditis, TIA → radiculopathy/MS) and produced hedged red flags ("thunderclap not excluded") that the judge rightly rejected. Seth: "jumping to red flags too early is quite nerve-racking for a patient."
- **L5**: five slots with a tag instead of eviction, flags only for stated features with a tier, and a code-enforced 90 s hold: flags 62 → 3 (all genuine), coverage 56/57, contradictions 3 and arguable (otoscopy on the ear-wax visit = the GP's own plan; "treat as anaphylaxis" vs the GP's antihistamine + ambulance). Cost: top-1 flips doubled.

## Search and the lookups: what actually happened
The v3/v4/v5 lookups ran through the beast's SearXNG (tunnelled to the Mac). It worked for the first ~20 conditions, then the upstream engines (Brave, DuckDuckGo, Google CSE) rate-limited and suspended under ~180 queries in an hour and returned empty result sets; serializing and retrying did not help. Net: **6 usable lookups of 179 attempted**, so none of the v4/v5 gains can be credited to guideline material; they are the prompt rules.

**Replacement (13 Sep 09:00–10:00): no search engine at all, three tiers, each fetched directly.**
1. **NHS conditions A-Z** (786 patient-facing entries, cached in `data/nhs_conditions_az.json`): the differential label is normalized (modifiers stripped, abbreviations expanded: TIA, URTI, ACS, PE…), fuzzy-matched by token, character and weighted ratio into a shortlist of 10, and DeepSeek picks the same condition or says none; a near-miss guard rejects picks that share no tokens. A small override table covers pages that exist but are not listed under that name (gastroenteritis → "Diarrhoea and vomiting", URTI → "Common cold", ACS → "Heart attack", urticaria → "Hives", patellofemoral → "Knee pain"). The condition page plus its symptoms/ and treatment/ subpages are fetched (~3 s cold).
2. **patient.info professional articles** (2,576 entries under specialty paths, cached in `data/patientinfo_doctor_az.json`), same matcher; preferred when its title matches at least as well as Wikipedia's.
3. **Wikipedia** (opensearch, then full-text search, then the plain-text extract API), clearly labelled **low trust**: the prompt is told it is background only and must not drive an urgent suggestion; the activity log shows a "low trust" badge. A relevance guard drops results whose title does not match the query (it had returned bronchiolitis obliterans for methotrexate pneumonitis).

NICE CKS stays out: it returns 403 to direct fetches. Coverage measured against the 181 distinct differential labels the model produced in the v5 run: **174 matched** after the aliases (the rest are non-conditions such as "medication side effect" or umbrella phrases such as "physical illness (e.g. anaemia, hypothyroidism, malignancy)"). Each cached lookup records its tier, and the eval reports how many lookups were used per visit. The `v5nhs` run (in progress) measures v5 with tier-1 lookups actually feeding the ticks; tiers 2–3 were added while it ran and will be in the next run.

## Still open
- A clinician reading the 3 v5 flags, the 3 contradictions, and a sample of the 459 "toward" revisions; every number here is the same model judging itself ([[judges]]).
- Consult modality is inferred from the transcript; in production it is known.
- Chart context (meds, problem list) is absent; the misses (hypothyroid, MS, migraine never spoken) are the visits where history alone was not enough.
- A working lookup path, then a re-run to measure what guideline material adds.

Sources: `demo/livesynth.py`, `demo/live.html`, `autoresearch/live_eval.py`, `runs/live_synth/eval/`, `runs/live_synth/eval_v4/`, `runs/live_synth/eval_v5/`, `runs/live_synth/eval_summary*.md`, `runs/live_synth/lookups/`, `runs/live_synth/*.json` (page sessions).
