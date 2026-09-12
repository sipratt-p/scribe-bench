# scribe-bench wiki

A Karpathy-style LLM wiki: the raw material (run outputs, notebooks, scripts, the artifact, the workshop doc) stays
immutable under `runs/`, `autoresearch/`, `RESULTS.md` and the scratchpad; this directory is the *compiled* knowledge
base, rewritten by the agent whenever a raw source changes. Pages are markdown with `[[wikilinks]]`, one topic per page,
each with a summary at the top, the numbers that matter, and a **Sources** section pointing at the raw files so every
claim can be re-derived.

Conventions
- `index.md` is the map of content. Every page is linked from it. `log.md` is the dated timeline.
- Numbers are copied from raw outputs, never remembered. If a page and a raw file disagree, the raw file wins and the page is fixed.
- Each experiment page states the split (dev 20 / test 37 / all 57, or ACI test1/2/3), the judge, and the date.
- Findings are tagged **transferred** (held across a change of data, judge, or metric) or **candidate** (found by the search loop, not yet confirmed elsewhere). See [[benchmark-maxing-vs-quality]].
- To update: read the raw source, edit the relevant page(s), add a line to `log.md`, and keep `index.md` current.

Start at [[index]].
