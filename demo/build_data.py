"""Assemble cached run outputs into demo/data/<id>.json + demo/data/index.json.

Per consultation: reference (human) transcript + clinician note; "theirs" = pass-1 streaming
transcript -> plain note; "ours" = pass-2 diarized transcript -> cited note -> verifier flags;
per-file metrics for both; judge rows where available. Missing pieces are left null so the UI
degrades gracefully while prep is still running.

  uv run python demo/build_data.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jiwer

from scribe_bench.asr_score import attribution_error, der_for, med_term_error
from scribe_bench.note_score import load_lexicon, med_terms, strip_cites
from scribe_bench.textnorm import normalize

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
EXPORT = ROOT / "data/primock_export"
OUT = ROOT / "demo/data"
LEX = load_lexicon(str(ROOT / "data/lexicon.txt"))
LEX_NORM = {normalize(t) for t in LEX}


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def asr_metrics(ref_text: str, hyp_text: str) -> dict:
    r, h = normalize(ref_text), normalize(hyp_text)
    m = jiwer.process_words([r], [h])
    tot, mis = med_term_error(r, h, LEX_NORM)
    rc, hc = Counter(r.split()), Counter(h.split())
    missed = sorted({w for w in rc if w in LEX_NORM and hc.get(w, 0) < rc[w]}, key=lambda w: -rc[w])
    return {"wer": round(100 * m.wer, 1), "med_terms": tot, "med_missed": mis,
            "med_term_err": round(100 * mis / max(tot, 1), 1), "missed_terms": missed[:25],
            "ref_words": len(r.split()), "hyp_words": len(h.split())}


def note_metrics(ref_note: str, note: str) -> dict:
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    hyp = strip_cites(note or "")
    rl = sc.score(ref_note, hyp)["rougeL"].fmeasure
    r, h = med_terms(ref_note, LEX), med_terms(hyp, LEX)
    return {"rougeL": round(100 * rl, 1), "term_recall": round(100 * len(r & h) / max(len(r), 1), 1),
            "term_precision": round(100 * len(r & h) / max(len(h), 1), 1), "terms_missing": sorted(r - h)[:20],
            "terms_extra": sorted(h - r)[:20], "words": len(hyp.split())}


def judge_rows(path: Path) -> dict:
    d = jload(path)
    return {r["encounter_id"]: r for r in d["rows"]} if d else {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    judged = {}
    for l in (RUNS / "verifier/judged_primock_moss.jsonl").read_text().splitlines() if (RUNS / "verifier/judged_primock_moss.jsonl").exists() else []:
        r = json.loads(l)
        judged.setdefault(r["encounter_id"], []).append(r)
    jt = judge_rows(RUNS / "verifier/judge_primock_nemotron_offline_dir.json")
    jo = judge_rows(RUNS / "verifier/judge_primock_cite_moss.json") or judge_rows(RUNS / "verifier/judge_primock_moss_plain.json")
    index = []
    for p in sorted(EXPORT.glob("*.json")):
        rec = json.loads(p.read_text())
        cid = rec["id"]
        theirs_t = jload(RUNS / f"nemotron_offline_dir/{cid}.json")
        ours_t = jload(RUNS / f"moss_plain/{cid}.json")
        theirs_n = jload(RUNS / f"notes_qwen38/primock/nemotron_offline_dir/{cid}.json")
        ours_n = jload(RUNS / f"notes_qwen38_cite/primock/moss_plain/{cid}.json") or jload(RUNS / f"notes_qwen38/primock/moss_plain/{cid}.json")
        ours_text = " ".join(s["text"] for s in ours_t["segments"]) if ours_t else ""
        d = {
            "id": cid, "audio": f"/api/audio/{cid}",
            "duration_s": round(rec["utterances"][-1]["end"], 1) if rec["utterances"] else None,
            "presenting_complaint": rec["note"].get("presenting_complaint"),
            "reference": {"utterances": rec["utterances"], "dialogue": rec["dialogue"], "note": rec["note"]["note"]},
            "theirs": {
                "label": "Streaming ASR (Nemotron 3.5 0.6B) → note, no verifier",
                "transcript_text": theirs_t["pred_text"] if theirs_t else None,
                "segments": None,
                "asr": asr_metrics(rec["reference_text"], theirs_t["pred_text"]) if theirs_t else None,
                "note": theirs_n["note"] if theirs_n else None,
                "note_metrics": note_metrics(rec["note"]["note"], theirs_n["note"]) if theirs_n else None,
                "judge": jt.get(cid),
            },
            "ours": {
                "label": "Pass-2 diarized ASR (MOSS-TD 0.9B) → cited note → verifier",
                "transcript_text": ours_text or None,
                "segments": ours_t["segments"] if ours_t else None,
                "asr": asr_metrics(rec["reference_text"], ours_text) if ours_t else None,
                "note": ours_n["note"] if ours_n else None,
                "note_metrics": note_metrics(rec["note"]["note"], ours_n["note"]) if ours_n else None,
                "judge": jo.get(cid),
                "claims": judged.get(cid),
            },
        }
        if ours_t:
            d["ours"]["asr"].update(der_for(EXPORT, RUNS / "moss_plain", [cid]))
            d["ours"]["asr"].update(attribution_error(EXPORT, RUNS / "moss_plain", [cid]))
        (OUT / f"{cid}.json").write_text(json.dumps(d))
        index.append({"id": cid, "complaint": d["presenting_complaint"], "duration_s": d["duration_s"],
                      "has_ours_note": bool(ours_n), "has_claims": cid in judged})
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(len(index), "consultations ->", OUT)


if __name__ == "__main__":
    main()
