"""Score generated notes against reference notes.

  python -m scribe_bench.note_score aci <aci_root> <run_dir> --split test1
  python -m scribe_bench.note_score primock <export_dir> <run_dir>

Metrics: ROUGE-1/2/L F (rouge-score, stemmed), BERTScore F1 (microsoft/deberta-xlarge-mnli,
the ACI-Bench setting), and citation stats when notes carry [[n]] citations.
Prints one row per transcript variant."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rouge_score import rouge_scorer

CITE = re.compile(r"\[\[([\d,\s]+)\]\]")


def strip_cites(t: str) -> str:
    return re.sub(r"\s*\[\[[\d,\s]+\]\]", "", t)


def cite_stats(notes: list[str]) -> dict:
    sents = cited = 0
    for n in notes:
        for s in re.split(r"(?<=[.!?])\s+|\n", n):
            s = s.strip()
            if len(s.split()) < 3 or s.isupper():
                continue
            sents += 1
            cited += bool(CITE.search(s))
    return {"sentences": sents, "cited_frac": round(cited / max(sents, 1), 3)}


def score(pairs: list[tuple[str, str]], bert: bool) -> dict:
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, hyp in pairs:
        s = sc.score(ref, hyp)
        for k in agg:
            agg[k] += s[k].fmeasure
    out = {k: round(100 * v / len(pairs), 2) for k, v in agg.items()}
    if bert:
        from bert_score import score as bscore
        _, _, f = bscore([h for _, h in pairs], [r for r, _ in pairs], model_type="microsoft/deberta-xlarge-mnli",
                         lang="en", verbose=False, batch_size=8)
        out["bertscore_f1"] = round(100 * float(f.mean()), 2)
    out["n"] = len(pairs)
    out["hyp_words"] = round(sum(len(h.split()) for _, h in pairs) / len(pairs))
    out["ref_words"] = round(sum(len(r.split()) for r, _ in pairs) / len(pairs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["aci", "primock"])
    ap.add_argument("root")
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="test1")
    ap.add_argument("--no_bert", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.corpus == "aci":
        from scribe_bench.acibench import load_split
        refs = {k: v["note"] for k, v in load_split(Path(args.root), args.split).items()}
        run = Path(args.run_dir) / args.split
    else:
        refs = {p.stem: json.loads(p.read_text())["note"]["note"] for p in Path(args.root).glob("*.json")}
        run = Path(args.run_dir) / "primock"

    results = {}
    for vdir in sorted(d for d in run.iterdir() if d.is_dir()):
        pairs, raw = [], []
        for p in vdir.glob("*.json"):
            if p.stem in refs:
                note = json.loads(p.read_text())["note"] or ""
                raw.append(note)
                pairs.append((refs[p.stem], strip_cites(note)))
        if not pairs:
            continue
        r = score(pairs, not args.no_bert)
        if any(CITE.search(n) for n in raw):
            r.update(cite_stats(raw))
        results[vdir.name] = r
        print(f"{vdir.name:20s} " + "  ".join(f"{a}={b}" for a, b in r.items()), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
