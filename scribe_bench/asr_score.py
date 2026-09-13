"""Score ASR runs against PriMock57 references.

  python -m scribe_bench.asr_score <export_dir> --lexicon data/lexicon.txt \
      --moss runs/moss_plain --moss runs/moss_hot --nemo_manifest runs/nemotron/offline.json ...

Reports per system: WER, medical-term error rate (fraction of reference lexicon
tokens not recovered), and for speaker-attributed systems DER via pyannote.metrics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jiwer

from scribe_bench.textnorm import normalize


def load_refs(export_dir: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(p.read_text()) for p in sorted(export_dir.glob("*.json")))}


def med_term_error(ref: str, hyp: str, lex: set[str]) -> tuple[int, int]:
    """Counts reference lexicon tokens and how many are missing from the hypothesis
    (bag-of-words; a term said twice and recognized once counts one miss)."""
    rc, hc = Counter(ref.split()), Counter(hyp.split())
    total = missed = 0
    for w, n in rc.items():
        if w in lex:
            total += n
            missed += max(0, n - hc.get(w, 0))
    return total, missed


def score_pairs(pairs: list[tuple[str, str]], lex: set[str]) -> dict:
    refs = [normalize(r) for r, _ in pairs]
    hyps = [normalize(h) for _, h in pairs]
    m = jiwer.process_words(refs, hyps)
    tot = mis = 0
    for r, h in zip(refs, hyps):
        t, x = med_term_error(r, h, lex)
        tot += t
        mis += x
    return {"n": len(pairs), "wer": round(100 * m.wer, 2), "sub": m.substitutions, "del": m.deletions,
            "ins": m.insertions, "ref_words": sum(len(r.split()) for r in refs),
            "med_terms": tot, "med_missed": mis, "med_term_err": round(100 * mis / max(tot, 1), 2)}


def attribution_error(export_dir: Path, hyp_dir: Path, ids: list[str]) -> dict:
    """Word-level speaker misattribution, the note-level 'attribution' dimension applied at the transcript level.
    Each hypothesis segment is placed on the reference timeline; the reference speaker at its
    midpoint is the truth. Hypothesis speaker labels are mapped to Doctor/Patient by majority
    overlap per file. Reports the share of hypothesis words carrying the wrong role, split by
    direction (clinician words given to the patient, patient words given to the clinician)."""
    from collections import Counter
    tot = wrong = doc2pat = pat2doc = 0
    for cid in ids:
        ref = json.loads((export_dir / f"{cid}.json").read_text())["utterances"]
        segs = json.loads((hyp_dir / f"{cid}.json").read_text())["segments"]

        def ref_spk(t):
            for u in ref:
                if u["start"] <= t <= u["end"]:
                    return u["speaker"]
            best = min(ref, key=lambda u: min(abs(t - u["start"]), abs(t - u["end"])), default=None)
            return best["speaker"] if best else None

        votes = {}
        placed = []
        for s in segs:
            r = ref_spk((s["start"] + s["end"]) / 2)
            n = len(s["text"].split())
            if r is None or n == 0:
                continue
            votes.setdefault(s["speaker"], Counter())[r] += n
            placed.append((s["speaker"], r, n))
        mapping = {h: c.most_common(1)[0][0] for h, c in votes.items()}
        for h, r, n in placed:
            tot += n
            if mapping[h] != r:
                wrong += n
                if r == "Doctor":
                    doc2pat += n
                else:
                    pat2doc += n
    return {"attr_err": round(100 * wrong / max(tot, 1), 2), "clin_to_pat": round(100 * doc2pat / max(tot, 1), 2),
            "pat_to_clin": round(100 * pat2doc / max(tot, 1), 2)}


def der_for(export_dir: Path, hyp_dir: Path, ids: list[str]) -> dict:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    for cid in ids:
        ref = Annotation(uri=cid)
        for line in (export_dir / "rttm" / f"{cid}.rttm").read_text().splitlines():
            f = line.split()
            ref[Segment(float(f[3]), float(f[3]) + float(f[4]))] = f[7]
        hyp = Annotation(uri=cid)
        for s in json.loads((hyp_dir / f"{cid}.json").read_text())["segments"]:
            if s["end"] > s["start"]:
                hyp[Segment(s["start"], s["end"])] = s["speaker"]
        metric(ref, hyp)
    return {"der": round(100 * abs(metric), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--moss", action="append", default=[], help="dir of MOSS json outputs")
    ap.add_argument("--nemo_manifest", action="append", default=[], help="NeMo pred manifest (jsonl)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    refs = load_refs(export_dir)
    lex = {normalize(t) for t in Path(args.lexicon).read_text().split() if t.strip()}
    results = {}

    for d in args.moss:
        d = Path(d)
        pairs, ids, wall = [], [], 0.0
        for p in sorted(d.glob("*.json")):
            h = json.loads(p.read_text())
            if h["id"] not in refs:
                continue
            pairs.append((refs[h["id"]]["reference_text"], " ".join(s["text"] for s in h["segments"])))
            ids.append(h["id"])
            wall += h.get("wall_s", 0)
        r = score_pairs(pairs, lex)
        r.update(der_for(export_dir, d, ids))
        r.update(attribution_error(export_dir, d, ids))
        r["wall_s"] = round(wall, 1)
        results[d.name] = r

    for mf in args.nemo_manifest:
        mf = Path(mf)
        rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
        by_text = {r["reference_text"]: cid for cid, r in refs.items()}
        pairs = []
        for row in rows:
            cid = row.get("id") or (Path(row["audio_filepath"]).stem if "audio_filepath" in row else by_text.get(row.get("text")))
            if cid in refs:
                pairs.append((refs[cid]["reference_text"], row["pred_text"]))
        results[f"{mf.parent.name}/{mf.stem}"] = score_pairs(pairs, lex)

    for k, v in results.items():
        print(f"{k:32s} " + "  ".join(f"{a}={b}" for a, b in v.items()))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
