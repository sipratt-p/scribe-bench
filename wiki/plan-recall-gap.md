# Plan recall under citations: where the cost comes from

Four cited prompt variants aimed at plan recall (enumerate every action / re-read the last third / forced Plan section / "a citable plan item must be included") gained up to +1.3 plan-recall points on dev and **none held on the 37 test files** (variant D: dev 78.95 → test 75.16 vs cited baseline 76.0; composite 41.76 vs 44.10).

Item-level gap analysis (test, 37 consultations, Qwen judge extracts items from the reference note, checks presence, and asks whether the transcript states the item):

| | Items | Uncited best found | Cited best found | Vanilla found |
|---|---|---|---|---|
| Stated in the transcript | 113–114 | 105 (92.9%) | 101 (89.4%) | 110 (96.5%) |
| Never spoken | 39–41 | 19 | 16 | 11 |

(Counts differ by one between runs because item extraction is an LLM call.)

Reading
- **The metric was wrong.** Of 154 reference plan items, ~41 are decisions the GP wrote but never said (a TSH, a full blood count, "prescribe naproxen"). A writer can only get those by guessing; the uncited writer sometimes guesses right and is rewarded. Plan recall should count transcript-stated items only, which is also closer to the completeness dimension in clinician-defined rubrics.
- **Half the cited writer's headline gap is refusing to invent.** On spoken items it is 3.5 points behind uncited, not 5.
- **Vanilla is best at spoken plan capture (96.5%).** The precision-first instruction and the citation rule both trade completeness for fewer invented terms. This is the most concrete open problem ([[open-questions]]).
- The seven real misses of the cited writer are safety-netting and follow-up actions said in the room (same-day face-to-face, blue-light ambulance, inhaler dose increase, A&E safety-net, pregnancy test, "contact the clinic if no better", paracetamol). A target for [[expert-iteration]].

Sources: `runs/plan_recall.md`, `runs/plan_recall.json`, `runs/plan_gap.json`, `autoresearch/plan_recall.py`, `autoresearch/plan_gap.py`, `/tmp/vanilla_extra.py` output (12 Sep).
