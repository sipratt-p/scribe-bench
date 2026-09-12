"""Re-score pipelines with a verifiability term (Abridge's Linked Evidence concern).

  python -m autoresearch.verif_eval [--ids dev|test|all]

Adds two per-note measures on top of the loop's metrics:
  grounded : share of note sentences a judge finds supported by the HUMAN transcript (what the clinician
             actually said), evidence = whole transcript. Same evidence for every config, cited or not.
  linked   : share of note sentences that carry a [[line]] citation whose cited lines (+/-1) support the
             sentence, judged against the transcript the writer saw. Uncited notes score 0 by construction.
Composite with verifiability: loop composite + 0.2 * grounded.
Writes runs/verif_eval.json and runs/verif_eval.md."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoresearch import loop as L  # noqa: E402
from autoresearch.loop import CITE, SENT_SPLIT, cached, h, parse_json, strip_cites  # noqa: E402

BEST_EXTRA = ("Prioritize high-precision terminology. Explicitly state 'no plan' if no plan items are mentioned. "
              "Do not hallucinate follow-up instructions or medications.")
CONFIGS = {
    "theirs": {"asr_variant": "nemotron_stream", "asr_correct": 0, "role_map": "none", "prompt": "base", "extra": "",
               "cite": 0, "gate": "none", "scaffold": 0, "note_model": "qwen27b"},
    "ours_v1": {"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "none", "prompt": "base", "extra": "",
                "cite": 0, "gate": "none", "scaffold": 0, "note_model": "qwen27b"},
    "best": {"asr_variant": "moss_hotcc", "asr_correct": 0, "role_map": "llm", "prompt": "base", "extra": BEST_EXTRA,
             "cite": 0, "gate": "none", "scaffold": 0, "note_model": "dsv4flash"},
    "best_cite": {"asr_variant": "moss_hotcc", "asr_correct": 0, "role_map": "llm", "prompt": "base", "extra": BEST_EXTRA,
                  "cite": 1, "gate": "none", "scaffold": 0, "note_model": "dsv4flash"},
}

GROUND_SYSTEM = """You verify a draft clinical note against the transcript of the consultation. For each numbered sentence of the note decide whether it is SUPPORTED by the transcript: the transcript states it, or it follows directly from what was said (a heading or section label counts as supported). A sentence that adds a detail the transcript does not contain (a drug, dose, duration, finding, instruction, history item) or contradicts the transcript is NOT supported. Answer with one JSON object mapping each sentence number to true or false and nothing else."""

LINK_SYSTEM = """Each numbered note sentence below is followed by the transcript lines its author cited as evidence. Decide, for each sentence, whether the cited lines alone support it. A sentence supported only by lines that were not cited counts as false. Answer with one JSON object mapping each sentence number to true or false and nothing else."""


def sentences(note: str) -> list[str]:
    out = []
    for s in SENT_SPLIT.split(note):
        s = s.strip()
        if s and not (len(s.split()) < 3 or s.isupper() or s.endswith(":")):
            out.append(s)
    return out


def verifiability(rec, note: str, dialogue: str, judge) -> dict:
    jtag = getattr(judge, "model", "j")
    sents = sentences(note)
    if not sents:
        return {"sents": 0, "grounded": 0.0, "linked": 0.0, "cited_frac": 0.0}

    def _ground():
        listing = "\n".join(f"{i + 1}. {strip_cites(s).strip()}" for i, s in enumerate(sents))
        f = parse_json(judge(GROUND_SYSTEM, f"TRANSCRIPT:\n{rec['dialogue']}\n\nNOTE SENTENCES:\n{listing}", 800)) or {}
        return sum(1 for i in range(len(sents)) if str(f.get(str(i + 1))).lower() == "true")
    g = cached(h("ground", jtag, rec["id"], note), _ground)

    lines = [l for l in dialogue.splitlines() if l.strip()]
    cited = [(i, s, sorted({int(x) for m in CITE.findall(s) for x in m.replace(" ", "").split(",") if x})) for i, s in enumerate(sents)]
    cited = [(i, s, c) for i, s, c in cited if c]

    def _link():
        if not cited:
            return 0
        blocks = []
        for i, s, c in cited:
            ev = "\n".join(f"{k}: {lines[k - 1]}" for cc in c for k in range(max(1, cc - 1), min(len(lines), cc + 1) + 1) if k <= len(lines))
            blocks.append(f"{i + 1}. {strip_cites(s).strip()}\nCITED LINES:\n{ev or '(cited line out of range)'}")
        f = parse_json(judge(LINK_SYSTEM, "\n\n".join(blocks), 800)) or {}
        return sum(1 for i, _, _ in cited if str(f.get(str(i + 1))).lower() == "true")
    lk = cached(h("linked", jtag, rec["id"], note), _link)
    n = len(sents)
    return {"sents": n, "grounded": 100 * g / n, "linked": 100 * lk / n, "cited_frac": 100 * len(cited) / n}


def run(name, cfg, ids, recs, note_llm, judge):
    nm = L.note_llm_for(cfg, note_llm)

    def one(cid):
        rec = recs[cid]
        d = L.transcript_for(cfg, rec, note_llm)
        n = L.note_for(cfg, rec, d, nm)
        n = L.gate(cfg, rec, d, n, judge)
        m = L.score_note(rec, n, d, judge)
        m.update(verifiability(rec, n, d, judge))
        return m
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, ids))
    agg = {k: sum(r[k] for r in rows) / len(rows) for k in ("rougeL", "term_recall", "term_precision", "misattrib", "grounded", "linked", "cited_frac")}
    agg["plan_items"] = sum(r["plan_items"] for r in rows)
    agg["plan_found"] = sum(r["plan_found"] for r in rows)
    agg["plan_recall"] = 100 * agg["plan_found"] / max(agg["plan_items"], 1)
    agg["score"] = L.composite(agg)
    agg["score_v"] = agg["score"] + 0.2 * agg["grounded"]
    agg["n"] = len(rows)
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="all")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()
    recs, dev, test = L.load_recs()
    splits = {"dev": dev, "test": test} if args.ids == "all" else {args.ids: dev if args.ids == "dev" else test}
    note_llm = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=8)
    judge = note_llm
    out = {}
    for name in args.configs.split(","):
        for split, ids in splits.items():
            m = run(name, CONFIGS[name], ids, recs, note_llm, judge)
            out[f"{name}/{split}"] = m
            print(name, split, json.dumps(m), flush=True)
    (ROOT / "runs/verif_eval.json").write_text(json.dumps(out, indent=1))
    cols = ["score", "score_v", "grounded", "linked", "cited_frac", "term_recall", "term_precision", "rougeL", "plan_recall", "misattrib", "n"]
    md = ["| config | split | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k, m in out.items():
        md.append(f"| {k.split('/')[0]} | {k.split('/')[1]} | " + " | ".join(str(m[c]) for c in cols) + " |")
    (ROOT / "runs/verif_eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
