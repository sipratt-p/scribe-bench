"""Generate clinical notes from transcripts with an OpenAI-compatible server.

  python -m scribe_bench.notegen aci  <aci_root> <out_dir> --base_url http://beast:8004/v1 --model qwen3.8-27b \
        --split test1 --variants humantrans,asr,asrcorr [--cite]
  python -m scribe_bench.notegen primock <export_dir> <out_dir> ... [--transcript_dir runs/moss_plain]

Output: <out_dir>/<split>/<variant>/<encounter_id>.json  {note, prompt_tokens, completion_tokens, seconds}
With --cite, every sentence in the note must end with [[n]] citing the transcript line number(s);
the transcript is sent with numbered lines. Reasoning is disabled (Qwen/Nemotron style chat_template_kwargs)."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ACI_SYSTEM = """You are a clinical documentation assistant. Given a doctor-patient conversation transcript, write the visit note.
Use exactly these section headers, in this order, in uppercase on their own line:
CHIEF COMPLAINT
HISTORY OF PRESENT ILLNESS
PHYSICAL EXAM
RESULTS
ASSESSMENT AND PLAN
Write in the third person, past tense, concise clinical prose. Include only information stated in the transcript.
If a section has no content in the transcript, write "None reported." under it. Do not invent vitals, exam findings, or results.
Do not include any information about people other than the patient unless it is clinically relevant family or social history."""

PRIMOCK_SYSTEM = """You are a clinical documentation assistant for a UK primary-care (GP) practice. Given a consultation transcript, write the consultation note in the style a GP writes in the record: terse, abbreviations allowed (hx, PMH, DH, SH, Imp, Plan), organised as presenting complaint and history, relevant negatives, PMH, DH, SH, impression, and plan.
Include only information stated in the transcript. Do not invent examination findings.
Do not include any information about people other than the patient unless it is clinically relevant family or social history."""

CITE_SUFFIX = """
The transcript lines are numbered. Every sentence you write must end with a citation of the line number(s) that support it, formatted like [[12]] or [[12,15]]. A sentence with no supporting line must not be written."""


def number_lines(dialogue: str) -> str:
    return "\n".join(f"{i + 1}: {l}" for i, l in enumerate(dialogue.splitlines()) if l.strip())


def gen_one(client: OpenAI, model: str, system: str, transcript: str, cite: bool, max_tokens: int) -> dict:
    t0 = time.time()
    body = number_lines(transcript) if cite else transcript
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system + (CITE_SUFFIX if cite else "")},
                  {"role": "user", "content": "TRANSCRIPT:\n" + body + "\n\nWrite the note now."}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return {"note": r.choices[0].message.content, "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens, "seconds": round(time.time() - t0, 1)}


def run_jobs(jobs: list[tuple[Path, str, str]], client, model, cite, workers, max_tokens):
    def work(j):
        dst, system, transcript = j
        if dst.exists():
            return dst.name, "cached"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            res = gen_one(client, model, system, transcript, cite, max_tokens)
        except Exception as e:  # noqa: BLE001
            return dst.name, f"ERROR {e}"
        dst.write_text(json.dumps(res, indent=1))
        return dst.name, f"{res['completion_tokens']} tok {res['seconds']}s"

    with ThreadPoolExecutor(workers) as ex:
        for name, msg in ex.map(work, jobs):
            print(name, msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["aci", "primock"])
    ap.add_argument("root")
    ap.add_argument("out_dir")
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test1")
    ap.add_argument("--variants", default="humantrans,asr,asrcorr")
    ap.add_argument("--transcript_dir", action="append", default=[],
                    help="primock: dir of ASR json outputs to use as transcript variant (name = dir name)")
    ap.add_argument("--cite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="x")
    out = Path(args.out_dir) / (args.split if args.corpus == "aci" else "primock")
    jobs = []
    if args.corpus == "aci":
        from scribe_bench.acibench import load_split
        enc = load_split(Path(args.root), args.split)
        for eid, e in list(enc.items())[: args.limit or None]:
            for v in args.variants.split(","):
                if v in e["variants"]:
                    jobs.append((out / v / f"{eid}.json", ACI_SYSTEM, e["variants"][v]))
    else:
        recs = [json.loads(p.read_text()) for p in sorted(Path(args.root).glob("*.json"))][: args.limit or None]
        for rec in recs:
            jobs.append((out / "reference" / f"{rec['id']}.json", PRIMOCK_SYSTEM, rec["dialogue"]))
            for d in args.transcript_dir:
                d = Path(d)
                p = d / f"{rec['id']}.json"
                if p.exists():
                    h = json.loads(p.read_text())
                    if "segments" in h:
                        txt = "\n".join(f"[{s['speaker']}] {s['text']}" for s in h["segments"])
                    else:
                        txt = h.get("pred_text") or h.get("text")
                    jobs.append((out / d.name / f"{rec['id']}.json", PRIMOCK_SYSTEM, txt))
    print(len(jobs), "jobs")
    run_jobs(jobs, client, args.model, args.cite, args.workers, args.max_tokens)


if __name__ == "__main__":
    main()
