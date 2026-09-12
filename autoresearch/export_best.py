"""Materialise the loop's accepted best config for every PriMock consultation from the cache.

  python -m autoresearch.export_best <out_dir>

Writes <out_dir>/<id>.json = {"id", "config", "transcript" (role-mapped dialogue the note model saw),
"note", "metrics" (per-file rougeL/term_recall/term_precision/misattrib/plan)}. Uses the same cached
calls as the loop, so nothing is regenerated unless a consultation was never evaluated."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoresearch import loop as L  # noqa: E402


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads(L.STATE.read_text())
    cfg = state["best"]
    recs, dev, test = L.load_recs()
    note_llm = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=4)
    judge = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=4)
    nm = L.note_llm_for(cfg, note_llm)
    n = 0
    for cid in dev + test:
        rec = recs[cid]
        d = L.transcript_for(cfg, rec, note_llm)
        note = L.note_for(cfg, rec, d, nm)
        note = L.gate(cfg, rec, d, note, judge)
        m = L.score_note(rec, note, d, judge)
        (out / f"{cid}.json").write_text(json.dumps({"id": cid, "config": cfg, "transcript": d, "note": note,
                                                     "metrics": {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()},
                                                     "split": "dev" if cid in dev else "test"}))
        n += 1
    (out / "_config.json").write_text(json.dumps({"config": cfg, "best_dev": state["best_dev"], "best_test": state["best_test"],
                                                   "best_test2": state.get("best_test2")}, indent=1))
    print(n, "consultations exported for", cfg)


if __name__ == "__main__":
    main()
