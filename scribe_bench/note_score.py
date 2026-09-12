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


def load_lexicon(path: str | None) -> set[str]:
    if not path:
        return set()
    return {w.strip().lower() for w in Path(path).read_text().split() if w.strip()}


def med_terms(text: str, lex: set[str]) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", text.lower()) if w in lex}


def score(pairs: list[tuple[str, str]], bert: bool, lex: set[str] = frozenset()) -> dict:
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, hyp in pairs:
        s = sc.score(ref, hyp)
        for k in agg:
            agg[k] += s[k].fmeasure
    out = {k: round(100 * v / len(pairs), 2) for k, v in agg.items()}
    if bert:
        import bert_score.utils as bsu
        from bert_score import score as bscore
        _orig_encode = bsu.sent_encode

        def _capped_encode(tokenizer, sent):  # transformers 5 tokenizers report a huge model_max_length
            try:
                tokenizer.model_max_length = min(int(tokenizer.model_max_length), 512)
            except Exception:  # noqa: BLE001
                tokenizer.model_max_length = 512
            return _orig_encode(tokenizer, sent)
        bsu.sent_encode = _capped_encode
        _, _, f = bscore([h for _, h in pairs], [r for r, _ in pairs], model_type="microsoft/deberta-xlarge-mnli",
                         lang="en", verbose=False, batch_size=8)
        out["bertscore_f1"] = round(100 * float(f.mean()), 2)
    if lex:
        tp = fp = fn = 0
        for ref, hyp in pairs:
            r, h = med_terms(ref, lex), med_terms(hyp, lex)
            tp += len(r & h); fp += len(h - r); fn += len(r - h)
        out["term_recall"] = round(100 * tp / max(tp + fn, 1), 2)
        out["term_precision"] = round(100 * tp / max(tp + fp, 1), 2)
    # effort proxy: character edits needed to turn the draft into the reference, per 100 reference words
    from rapidfuzz.distance import Levenshtein
    ed = sum(Levenshtein.distance(h, r) for r, h in pairs)
    rw = sum(len(r.split()) for r, _ in pairs)
    out["edits_per_100w"] = round(100 * ed / max(rw, 1) / 1.0, 1)
    out["len_ratio"] = round(sum(len(h.split()) for _, h in pairs) / max(rw, 1), 2)
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
    ap.add_argument("--intersect", action="store_true", help="score only encounters present in every variant")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lexicon", default="data/lexicon.txt")
    ap.add_argument("--by", default=None, help="aci only: stratify by metadata field, e.g. patient_gender or age_band")
    args = ap.parse_args()
    lex = load_lexicon(args.lexicon)

    groups = None
    if args.corpus == "aci":
        from scribe_bench.acibench import load_split, load_metadata
        refs = {k: v["note"] for k, v in load_split(Path(args.root), args.split).items()}
        run = Path(args.run_dir) / args.split
        if args.by:
            meta = load_metadata(Path(args.root), args.split)
            groups = {}
            for eid, m in meta.items():
                if args.by == "age_band":
                    try:
                        a = float(m.get("patient_age") or "nan")
                        g = "under 40" if a < 40 else "40 to 64" if a < 65 else "65 and over"
                    except ValueError:
                        g = "unknown"
                else:
                    g = (m.get(args.by) or "unknown").strip().lower() or "unknown"
                groups.setdefault(g, set()).add(eid)
    else:
        refs = {p.stem: json.loads(p.read_text())["note"]["note"] for p in Path(args.root).glob("*.json")}
        run = Path(args.run_dir) / "primock"

    results = {}
    vdirs = sorted(d for d in run.iterdir() if d.is_dir())
    keep = set(refs)
    if args.intersect:
        for d in vdirs:
            keep &= {p.stem for p in d.glob("*.json")}
    for vdir in vdirs:
        pairs, raw = [], []
        for p in vdir.glob("*.json"):
            if p.stem in keep:
                note = json.loads(p.read_text())["note"] or ""
                raw.append(note)
                pairs.append((refs[p.stem], strip_cites(note)))
        if not pairs:
            continue
        r = score(pairs, not args.no_bert, lex)
        if any(CITE.search(n) for n in raw):
            r.update(cite_stats(raw))
        results[vdir.name] = r
        print(f"{vdir.name:20s} " + "  ".join(f"{a}={b}" for a, b in r.items()), flush=True)
        if groups:
            ids = [p.stem for p in vdir.glob("*.json") if p.stem in keep]
            by_id = dict(zip(ids, [pr for pr in pairs]))
            for g, members in sorted(groups.items()):
                sub = [by_id[i] for i in ids if i in members]
                if len(sub) >= 3:
                    rg = score(sub, not args.no_bert, lex)
                    results[f"{vdir.name}/{args.by}={g}"] = rg
                    print(f"  {args.by}={g:12s} " + "  ".join(f"{a}={b}" for a, b in rg.items()), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
