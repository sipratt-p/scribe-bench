"""Claim-level faithfulness verification for generated notes.

Three subcommands:

  extract  : split cited notes ([[n]] citations) into claims with their evidence lines
             python -m scribe_bench.verifier extract <notes_dir> <transcripts.json> <claims.jsonl>
  perturb  : inject synthetic unsupported claims into a claims file (drug swap, negation flip,
             fabricated finding, third-party leak) to get labeled positives for recall measurement
             python -m scribe_bench.verifier perturb <claims.jsonl> <perturbed.jsonl> [--rate 0.25]
  judge    : label each claim SUPPORTED / PARTIAL / UNSUPPORTED with an LLM judge and report
             precision/recall against injected labels when present
             python -m scribe_bench.verifier judge <claims.jsonl> <out.jsonl> --base_url ... --model ...

transcripts.json maps encounter id -> dialogue text (numbered lines are derived here the
same way notegen numbers them)."""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CITE = re.compile(r"\[\[([\d,\s]+)\]\]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])|\n")

JUDGE_SYSTEM = """You are a clinical documentation auditor. You are given one sentence from a draft clinical note and the transcript lines it cites (with a little surrounding context). Decide whether the transcript supports the sentence.

First, in one or two sentences, compare each clinical fact in the sentence (drug, dose, number, side, duration, negation, who it is about) with the evidence.
Then, on the LAST line, write exactly: VERDICT: <SUPPORTED|PARTIAL|UNSUPPORTED> LEAK=<yes|no>
- SUPPORTED: every clinical fact in the sentence is stated or directly implied in the evidence
- PARTIAL: some facts are supported but at least one detail (number, side, duration, drug, negation) is not
- UNSUPPORTED: the central claim is not in the evidence, contradicts it, or describes a person other than the patient as if it were the patient
- LEAK=yes if the sentence records health information about a person other than the patient (spouse, child, friend, relative's own diagnosis or medication) that is not the patient's own family-history risk."""


def parse_verdict(txt: str) -> tuple[str, bool]:
    """Verdict is on the last line; fall back to scanning the whole text."""
    lines = [l for l in txt.strip().splitlines() if l.strip()]
    tail = lines[-1].upper() if lines else ""
    src = tail if "VERDICT" in tail else txt.upper()
    label = "UNSUPPORTED" if "UNSUPPORTED" in src else "PARTIAL" if "PARTIAL" in src else "SUPPORTED" if "SUPPORTED" in src else "ERROR"
    leak = bool(re.search(r"LEAK\s*=\s*YES", src))
    return label, leak

DRUGS = ["paracetamol", "ibuprofen", "amoxicillin", "omeprazole", "metformin", "amlodipine", "atorvastatin",
         "sertraline", "salbutamol", "ramipril", "codeine", "naproxen", "loratadine", "prednisolone"]
FINDINGS = ["Chest examination revealed bilateral crackles.", "Blood pressure was 165/98.",
            "Abdomen was tender in the right iliac fossa with guarding.", "Temperature was 38.9 C.",
            "There was a palpable mass in the left breast.", "Oxygen saturation was 89% on air."]
THIRD_PARTY = ["The patient's wife has a history of breast cancer and is on tamoxifen.",
               "The patient's son was recently diagnosed with type 1 diabetes.",
               "The patient's brother is being treated for depression with sertraline."]
NEG_FLIPS = [("no ", "some "), ("denies ", "reports "), ("denied ", "reported "), ("without ", "with "),
             ("nil ", "some "), ("not ", "")]


def number_lines(dialogue: str) -> list[str]:
    return [l for l in dialogue.splitlines() if l.strip()]


def extract(notes_dir: Path, transcripts: dict[str, str], out: Path) -> int:
    n = 0
    with open(out, "w") as f:
        for p in sorted(notes_dir.glob("*.json")):
            eid = p.stem
            if eid not in transcripts:
                continue
            lines = number_lines(transcripts[eid])
            note = json.loads(p.read_text())["note"] or ""
            for s in SENT_SPLIT.split(note):
                s = s.strip()
                if len(s.split()) < 3 or s.isupper() or s.endswith(":"):
                    continue
                cites = sorted({int(x) for m in CITE.findall(s) for x in m.replace(" ", "").split(",") if x})
                claim = CITE.sub("", s).strip()
                ev = []
                for c in cites:
                    for k in range(max(1, c - 1), min(len(lines), c + 1) + 1):
                        ev.append((k, lines[k - 1]))
                ev = sorted(set(ev))
                f.write(json.dumps({"encounter_id": eid, "claim": claim, "cites": cites,
                                    "evidence": [f"{k}: {t}" for k, t in ev], "injected": None}) + "\n")
                n += 1
    return n


def perturb(rows: list[dict], rate: float, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    out = []
    for r in rows:
        r = dict(r)
        if rnd.random() < rate:
            kind = rnd.choice(["drug", "negation", "finding", "third_party", "number"])
            c = r["claim"]
            if kind == "drug" and any(d in c.lower() for d in DRUGS):
                cur = next(d for d in DRUGS if d in c.lower())
                new = rnd.choice([d for d in DRUGS if d != cur])
                r["claim"] = re.sub(cur, new, c, flags=re.I)
            elif kind == "negation" and any(a in c.lower() for a, _ in NEG_FLIPS):
                a, b = next((a, b) for a, b in NEG_FLIPS if a in c.lower())
                r["claim"] = re.sub(a, b, c, count=1, flags=re.I)
            elif kind == "number" and re.search(r"\b\d+\b", c):
                r["claim"] = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) * 2 + 1), c, count=1)
            elif kind == "finding":
                r["claim"] = rnd.choice(FINDINGS)
            elif kind == "third_party":
                r["claim"] = rnd.choice(THIRD_PARTY)
            else:
                r["claim"] = rnd.choice(FINDINGS)
                kind = "finding"
            r["injected"] = kind
        out.append(r)
    return out


def judge(rows: list[dict], base_url: str, model: str, workers: int) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="x")

    def one(r):
        ev = "\n".join(r["evidence"]) or "(no lines cited)"
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=220,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": f"SENTENCE:\n{r['claim']}\n\nEVIDENCE:\n{ev}"}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            txt = resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001
            txt = f"ERROR {e}"
        label, leak = parse_verdict(txt)
        return {**r, "label": label, "leak": leak, "judge_raw": txt}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def report(rows: list[dict]) -> dict:
    from collections import Counter
    lab = Counter(r["label"] for r in rows)
    inj = [r for r in rows if r["injected"]]
    clean = [r for r in rows if not r["injected"]]
    flagged = lambda r: r["label"] in ("UNSUPPORTED", "PARTIAL") or r.get("leak")  # noqa: E731
    res = {"n": len(rows), "labels": dict(lab)}
    if inj:
        tp = sum(flagged(r) for r in inj)
        fp = sum(flagged(r) for r in clean)
        res["injected"] = len(inj)
        res["recall_on_injected"] = round(tp / len(inj), 3)
        res["flag_rate_on_clean"] = round(fp / max(len(clean), 1), 3)
        by = Counter(r["injected"] for r in inj)
        res["recall_by_kind"] = {k: round(sum(flagged(r) for r in inj if r["injected"] == k) / n, 2) for k, n in by.items()}
        res["leak_flag_on_clean"] = round(sum(bool(r.get("leak")) for r in clean) / max(len(clean), 1), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("notes_dir"); e.add_argument("transcripts"); e.add_argument("out")
    p = sub.add_parser("perturb"); p.add_argument("inp"); p.add_argument("out"); p.add_argument("--rate", type=float, default=0.25)
    j = sub.add_parser("judge"); j.add_argument("inp"); j.add_argument("out"); j.add_argument("--base_url", required=True)
    j.add_argument("--model", required=True); j.add_argument("--workers", type=int, default=8); j.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "extract":
        n = extract(Path(a.notes_dir), json.loads(Path(a.transcripts).read_text()), Path(a.out))
        print(n, "claims ->", a.out)
    elif a.cmd == "perturb":
        rows = [json.loads(l) for l in Path(a.inp).read_text().splitlines()]
        out = perturb(rows, a.rate)
        Path(a.out).write_text("\n".join(json.dumps(r) for r in out) + "\n")
        print(sum(bool(r["injected"]) for r in out), "injected of", len(out))
    else:
        rows = [json.loads(l) for l in Path(a.inp).read_text().splitlines()][: a.limit or None]
        out = judge(rows, a.base_url, a.model, a.workers)
        Path(a.out).write_text("\n".join(json.dumps(r) for r in out) + "\n")
        print(json.dumps(report(out), indent=1))


if __name__ == "__main__":
    main()
