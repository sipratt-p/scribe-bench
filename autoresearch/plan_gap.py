"""Which plan items does the cited writer drop, and are they in the transcript at all?

  python -m autoresearch.plan_gap

For each test consultation: extract plan items from the reference note (same judge prompt as the loop),
check presence in the uncited-best note and the cited-best note, and for items found only by the uncited
note ask the judge whether the transcript states the item. Writes runs/plan_gap.json and prints a summary."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoresearch import loop as L  # noqa: E402
from autoresearch.loop import COMPLETE_CHECK, COMPLETE_EXTRACT, cached, h, parse_json, strip_cites  # noqa: E402
from autoresearch.verif_eval import CONFIGS  # noqa: E402

IN_TRANSCRIPT = """Below is a consultation transcript and a list of plan items taken from the clinician's note. For each item decide whether the transcript itself states or clearly agrees that action (true), or whether the item is not said in the conversation (false). Answer with one JSON object mapping item numbers to true/false and nothing else."""


def main():
    recs, dev, test = L.load_recs()
    judge = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=8)
    unc, cit, out = (sys.argv[1:4] + ["best", "best_cite", "runs/plan_gap.json"][len(sys.argv) - 1:])[:3]
    ALL = {**CONFIGS, "best_official": {**CONFIGS["best"], "note_model": "dsv4flash_official"}, "best_cite_official": {**CONFIGS["best_cite"], "note_model": "dsv4flash_official"}}
    notes = {}
    for name in ("best", "best_cite"):
        cfg = ALL[{"best": unc, "best_cite": cit}[name]]
        nm = L.note_llm_for(cfg, judge)
        notes[name] = {c: L.note_for(cfg, recs[c], L.transcript_for(cfg, recs[c], judge), nm) for c in test}

    def one(cid):
        rec = recs[cid]
        ref = rec["note"]["note"]
        items_txt = judge("You extract plan items from clinical notes.", COMPLETE_EXTRACT + ref, 400)
        items = [l.strip("-• ").strip() for l in items_txt.splitlines() if l.strip() and l.strip().upper() != "NONE"]
        if not items:
            return cid, []
        listing = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items))
        found = {}
        for name in ("best", "best_cite"):
            f = parse_json(judge(COMPLETE_CHECK, f"ITEMS:\n{listing}\n\nDRAFT NOTE:\n{strip_cites(notes[name][cid])}", 300)) or {}
            found[name] = [str(f.get(str(i + 1))).lower() == "true" for i in range(len(items))]
        t = parse_json(judge(IN_TRANSCRIPT, f"TRANSCRIPT:\n{rec['dialogue']}\n\nPLAN ITEMS:\n{listing}", 300)) or {}
        in_tr = [str(t.get(str(i + 1))).lower() == "true" for i in range(len(items))]
        return cid, [{"item": it, "uncited": found["best"][i], "cited": found["best_cite"][i], "in_transcript": in_tr[i]} for i, it in enumerate(items)]
    with ThreadPoolExecutor(6) as ex:
        rows = dict(ex.map(one, test))
    allitems = [r for rs in rows.values() for r in rs]
    n = len(allitems)
    both = sum(1 for r in allitems if r["uncited"] and r["cited"])
    only_unc = [r for r in allitems if r["uncited"] and not r["cited"]]
    only_cit = [r for r in allitems if r["cited"] and not r["uncited"]]
    neither = [r for r in allitems if not r["uncited"] and not r["cited"]]
    summary = {
        "items": n, "found_by_both": both, "only_uncited": len(only_unc), "only_cited": len(only_cit), "neither": len(neither),
        "only_uncited_in_transcript": sum(1 for r in only_unc if r["in_transcript"]),
        "neither_in_transcript": sum(1 for r in neither if r["in_transcript"]),
        "all_items_in_transcript": sum(1 for r in allitems if r["in_transcript"]),
    }
    (ROOT / out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("\nItems only the uncited note captured:")
    for r in only_unc:
        print(f"  [{'in transcript' if r['in_transcript'] else 'NOT in transcript'}] {r['item']}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
