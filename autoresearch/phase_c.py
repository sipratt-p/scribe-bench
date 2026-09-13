"""Phase C (beast on Qwen :8004, DeepSeek stopped): judge-1 scoring of everything Phase B produced, plus the second baseline.

  python -m autoresearch.phase_c

1. Held-out table with judge 1 + grounding for: vanilla (theirs), parakeet_qwen (second baseline), ours_v1, best_official, best_cite_official.
   Per-visit rows -> runs/paper_pervisit_test_official.json (same shape as runs/paper_pervisit_test.json), plus dev.
2. Summary markdown -> runs/verif_eval_official.md
Requires: runs/asr_variants/parakeet_v3/<id>.json (built by autoresearch/make_parakeet_variant.py) and Phase B's cached notes.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoresearch import loop as L  # noqa: E402
from autoresearch.verif_eval import CONFIGS, verifiability  # noqa: E402

CFGS = {
    "theirs": CONFIGS["theirs"],
    "parakeet_qwen": {**CONFIGS["theirs"], "asr_variant": "parakeet_v3"},
    "sortformer_parakeet_rolemap_qwen": {**CONFIGS["ours_v1"], "asr_variant": "sortformer_parakeet", "role_map": "llm"},
    "ours_v1": CONFIGS["ours_v1"],
    "best_official": {**CONFIGS["best"], "note_model": "dsv4flash_official"},
    "best_cite_official": {**CONFIGS["best_cite"], "note_model": "dsv4flash_official"},
}


def main():
    recs, dev, test = L.load_recs()
    j1 = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=6)
    out = {"test": {}, "dev": {}}
    for name, cfg in CFGS.items():
        nm = L.note_llm_for(cfg, j1)
        for split, ids in (("test", test), ("dev", dev)):
            def one(cid):
                rec = recs[cid]
                try:
                    d = L.transcript_for(cfg, rec, j1); n = L.note_for(cfg, rec, d, nm); n = L.gate(cfg, rec, d, n, j1)
                    m = L.score_note(rec, n, d, j1); m["composite"] = L.composite(m)
                    v = verifiability(rec, n, d, j1); m.update({k: v[k] for k in ("grounded", "linked", "cited_frac")})
                    return cid, m
                except Exception as e:  # noqa: BLE001
                    return cid, {"err": str(e)[:160]}
            with ThreadPoolExecutor(6) as ex:
                rows = dict(ex.map(one, ids))
            out[split][name] = rows
            ok = [r for r in rows.values() if "err" not in r]
            if ok:
                agg = {k: sum(r[k] for r in ok) / len(ok) for k in ("composite", "term_recall", "term_precision", "rougeL", "misattrib", "grounded", "linked", "cited_frac")}
                pi = sum(r["plan_items"] for r in ok); pf = sum(r["plan_found"] for r in ok)
                agg["plan_recall"] = 100 * pf / max(pi, 1)
                print(name, split, "n", len(ok), "err", len(rows) - len(ok), json.dumps({k: round(v, 2) for k, v in agg.items()}), flush=True)
            else:
                print(name, split, "ALL ERR", list(rows.values())[0], flush=True)
    json.dump(out, open(ROOT / "runs/paper_pervisit_official.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
