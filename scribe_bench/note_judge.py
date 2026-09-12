"""LLM judges for two of Abridge's clinician-defined note dimensions (from "The Science of AI
Evaluation for Enterprise Healthcare"): attribution and completeness of follow-ups/referrals.

  python -m scribe_bench.note_judge aci <aci_root> <run_dir> --split test1 --variant humantrans \
      --base_url http://localhost:8500/v1 --model gemma4-vision [--out report.json]

attribution : for each generated note, count statements that attribute to the patient something
              the clinician said (or vice versa), or that record a family member's/other person's
              report as the patient's own. The judge reads the note and the speaker-tagged transcript.
completeness: extract the follow-ups, referrals, orders and return-visit instructions from the
              reference note's plan; check each against the generated note. Recall of those items."""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ATTR_SYSTEM = """You are auditing a draft clinical note against the speaker-tagged transcript of the visit.
Find every sentence in the note that misattributes who said something. Count these kinds:
A. patient_to_clinician: the note presents the patient's statement as the clinician's finding or assessment
B. clinician_to_patient: the note says the patient reported/denied something that only the clinician said
C. other_person_to_patient: the note presents a family member's, caregiver's or other person's report or history as the patient's own
Ignore normal clinical paraphrase. Only count clear misattributions supported by the transcript.
Answer with exactly one JSON object and nothing else: {"A": <int>, "B": <int>, "C": <int>, "examples": ["<short quote from the note>", ...]}"""

COMPLETE_EXTRACT = """From the REFERENCE clinical note below, list every follow-up, referral, order, test, prescription or return-visit instruction in the plan. One item per line, short, no numbering. If there are none, answer NONE.

REFERENCE NOTE:
"""

COMPLETE_CHECK = """You are checking a draft clinical note for completeness. For each ITEM from the reference plan, say whether the DRAFT note contains it (the same follow-up, referral, order, test, prescription or instruction, even if worded differently).
Answer with exactly one JSON object mapping each item's number to true or false, e.g. {"1": true, "2": false}."""


def chat(client, model, system, user, max_tokens=400):
    r = client.chat.completions.create(model=model, temperature=0.0, max_tokens=max_tokens,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content or ""


def parse_json(t: str):
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["aci", "primock"])
    ap.add_argument("root")
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="test1")
    ap.add_argument("--variant", default="humantrans")
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from openai import OpenAI
    from scribe_bench.acibench import load_split
    client = OpenAI(base_url=args.base_url, api_key="x")
    if args.corpus == "aci":
        enc = load_split(Path(args.root), args.split)
        run = Path(args.run_dir) / args.split / args.variant
    else:  # primock: judge against the human speaker-tagged dialogue and the clinician's note
        enc = {}
        for p in Path(args.root).glob("*.json"):
            r = json.loads(p.read_text())
            enc[r["id"]] = {"variants": {"humantrans": r["dialogue"]}, "note": r["note"]["note"]}
        run = Path(args.run_dir) / "primock" / args.variant
    jobs = []
    for p in sorted(run.glob("*.json"))[: args.limit or None]:
        if p.stem in enc:
            note = re.sub(r"\s*\[\[[\d,\s]+\]\]", "", json.loads(p.read_text())["note"] or "")
            jobs.append((p.stem, note, enc[p.stem]))

    def one(j):
        eid, note, e = j
        transcript = e["variants"].get("humantrans") or next(iter(e["variants"].values()))
        a = parse_json(chat(client, args.model, ATTR_SYSTEM, f"TRANSCRIPT:\n{transcript}\n\nDRAFT NOTE:\n{note}", 500)) or {}
        items_txt = chat(client, args.model, "You extract plan items from clinical notes.", COMPLETE_EXTRACT + e["note"], 400)
        items = [l.strip("-• ").strip() for l in items_txt.splitlines() if l.strip() and l.strip().upper() != "NONE"]
        found = {}
        if items:
            listing = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items))
            found = parse_json(chat(client, args.model, COMPLETE_CHECK, f"ITEMS:\n{listing}\n\nDRAFT NOTE:\n{note}", 300)) or {}
        hit = sum(1 for i in range(len(items)) if str(found.get(str(i + 1))).lower() == "true")
        return {"encounter_id": eid, "attribution": {k: a.get(k, 0) for k in "ABC"}, "attr_examples": a.get("examples", [])[:3],
                "plan_items": len(items), "plan_found": hit}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    n = len(rows)
    attr = {k: sum(int(r["attribution"].get(k) or 0) for r in rows) for k in "ABC"}
    notes_with_attr = sum(1 for r in rows if any(int(r["attribution"].get(k) or 0) for k in "ABC"))
    items = sum(r["plan_items"] for r in rows)
    found = sum(r["plan_found"] for r in rows)
    rep = {"n": n, "variant": args.variant, "misattributions_per_note": round(sum(attr.values()) / max(n, 1), 3),
           "notes_with_any_misattribution": round(notes_with_attr / max(n, 1), 3),
           "patient_to_clinician": attr["A"], "clinician_to_patient": attr["B"], "other_person_to_patient": attr["C"],
           "plan_items": items, "plan_item_recall": round(found / max(items, 1), 3)}
    print(json.dumps(rep, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"report": rep, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
