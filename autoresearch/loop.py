"""Overnight autoresearch loop for the ambient-scribe pipeline.

Objective: close the gap between notes written from ASR transcripts and notes written from the
human transcript, on the clinician-experience scorecard, and raise the absolute score.

  python -m autoresearch.loop --deadline "2026-09-13 08:30" [--judge_url ...] [--note_url ...]

Knobs (a config is a dict of these):
  asr_variant   cached pass-2 transcript set under runs/asr_variants/<name>/<id>.json
  asr_correct   0/1  LLM post-edit of the transcript for misheard medical terms (lexicon-guided)
  role_map      none|llm  map anonymous speaker labels to Doctor/Patient before note writing
  prompt        key in PROMPTS
  extra         free-text instruction appended to the prompt (the proposer may write new ones)
  cite          0/1  require [[line]] citations per sentence
  gate          none|drop_flagged  run the claim verifier and delete flagged sentences

Each iteration: evaluate on DEV (20 consultations); if the composite beats the best by > 0.5,
confirm on TEST (37); accept only if TEST also improves. Human-transcript notes with the same
note-side config are scored too, so the gap is always measured like-for-like.
Everything is cached by content hash under runs/autoresearch/cache so reruns are free.
Results go to autoresearch/notebook.md and autoresearch/state.json. Touch autoresearch/STOP to end."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scribe_bench.notegen import PRIMOCK_SYSTEM, CITE_SUFFIX, number_lines  # noqa: E402
from scribe_bench.verifier import JUDGE_SYSTEM, SENT_SPLIT, CITE  # noqa: E402
from scribe_bench.note_judge import ATTR_SYSTEM, COMPLETE_EXTRACT, COMPLETE_CHECK, parse_json  # noqa: E402
from scribe_bench.note_score import load_lexicon, med_terms, strip_cites  # noqa: E402
from scribe_bench.textnorm import normalize  # noqa: E402

EXPORT = ROOT / "data/primock_export"
CACHE = ROOT / "runs/autoresearch/cache"
VARIANTS = ROOT / "runs/asr_variants"
NOTEBOOK = ROOT / "autoresearch/notebook.md"
STATE = ROOT / "autoresearch/state.json"
STOP = ROOT / "autoresearch/STOP"
LEX = load_lexicon(str(ROOT / "data/lexicon.txt"))
LEX_LIST = [t.strip() for t in (ROOT / "data/lexicon.txt").read_text().splitlines() if t.strip()]

PROMPTS = {
    "base": PRIMOCK_SYSTEM,
    "attrib": PRIMOCK_SYSTEM + "\nAttribute every statement to the right person: what the patient said is the patient's report; what the clinician said is the clinician's finding, advice or plan. Never present a family member's or other person's history as the patient's own; record it as family history only when clinically relevant.",
    "strict": PRIMOCK_SYSTEM + "\nWrite only what is explicitly stated in the transcript. If a detail (drug, dose, duration, side, number) is not stated, leave it out rather than guess. Do not add examination findings that were not performed. Keep the plan items exactly as the clinician stated them, including follow-up timing and safety-netting advice.",
    "attrib_strict": PRIMOCK_SYSTEM + "\nAttribute every statement to the right person: patient reports vs clinician findings and plan; never record another person's history as the patient's own. Write only what is explicitly stated; leave unstated details out. Keep every plan item, follow-up timing and safety-netting instruction the clinician gave.",
}

CORRECT_SYSTEM = """You are correcting an automatic speech-recognition transcript of a UK GP consultation. The transcript may contain misheard medical terms (drug names, conditions, anatomy). A list of plausible terms is given.
Rewrite the transcript line by line, changing ONLY words that are clearly misrecognitions of medical terms in the list or of common medical vocabulary. Keep speaker labels, line order, wording, fillers and everything else exactly as they are. Do not add, remove or reorder lines. Output the corrected transcript only."""

SCAFFOLD_SYSTEM = """Extract every clinically relevant fact from this consultation transcript as a numbered list of short, atomic propositions. Each proposition must be attributed (patient reports / clinician finds / clinician plans) and must end with the transcript line number(s) that support it, like [[12]]. Include negatives the patient explicitly denied and every plan, follow-up and safety-netting item. Do not infer or add anything not stated. Output only the list."""

ROLE_SYSTEM = """Below is a transcript of a medical consultation with anonymous speaker labels. Decide which label is the CLINICIAN (asks questions, examines, advises, plans) and which is the PATIENT. Answer with one JSON object only, e.g. {"S01": "Doctor", "S02": "Patient"}. Every label that appears must be mapped to exactly "Doctor" or "Patient"; if there are more than two, map each extra one to "Other"."""

DEV_N = 20


def h(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def cached(key: str, fn):
    p = CACHE / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    v = fn()
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v))
    return v


class LLM:
    def __init__(self, base_url, model, workers=8):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key="x", timeout=900, max_retries=0)
        self.model, self.workers = model, workers

    def __call__(self, system, user, max_tokens=1500, tries=4):
        last = None
        for i in range(tries):
            try:
                r = self.c.chat.completions.create(model=self.model, temperature=0.0, max_tokens=max_tokens,
                                                   messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(3 * (i + 1))
        raise last

    def map(self, jobs):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda j: self(*j), jobs))


def load_recs():
    recs = {p.stem: json.loads(p.read_text()) for p in sorted(EXPORT.glob("*.json"))}
    ids = sorted(recs)
    return recs, ids[:DEV_N], ids[DEV_N:]


def transcript_for(cfg, rec, note_llm) -> str:
    """Return the dialogue text the note model will see for this config."""
    if cfg["asr_variant"] == "human":
        return rec["dialogue"]
    p = VARIANTS / cfg["asr_variant"] / f"{rec['id']}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    segs = json.loads(p.read_text())["segments"]
    dialogue = "\n".join(f"[{s['speaker']}] {s['text']}" for s in segs)
    if cfg.get("role_map") == "llm":
        def _role():
            a = parse_json(note_llm(ROLE_SYSTEM, dialogue[:6000], 60)) or {}
            return {k: v for k, v in a.items() if v in ("Doctor", "Patient", "Other")}
        m = cached(h("role", rec["id"], cfg["asr_variant"]), _role)
        if m:
            dialogue = "\n".join(f"[{m.get(s['speaker'], s['speaker'])}] {s['text']}" for s in segs)
    if cfg.get("asr_correct"):
        def _corr():
            terms = ", ".join(LEX_LIST[:400])
            out = note_llm(CORRECT_SYSTEM, f"TERMS: {terms}\n\nTRANSCRIPT:\n{dialogue}", 4000)
            return out if out.count("\n") >= 0.7 * dialogue.count("\n") else dialogue  # reject if lines were dropped
        dialogue = cached(h("corr", rec["id"], cfg["asr_variant"], cfg.get("role_map")), _corr)
    return dialogue


def note_for(cfg, rec, dialogue, note_llm) -> str:
    system = PROMPTS[cfg["prompt"]] + ("\n" + cfg["extra"] if cfg.get("extra") else "") + (CITE_SUFFIX if cfg.get("cite") else "")
    body = number_lines(dialogue) if cfg.get("cite") else dialogue
    if cfg.get("scaffold"):
        props = cached(h("props", rec["id"], dialogue), lambda: note_llm(SCAFFOLD_SYSTEM, "TRANSCRIPT:\n" + number_lines(dialogue), 2500))
        user = ("TRANSCRIPT:\n" + body + "\n\nEXTRACTED PROPOSITIONS (each with supporting line numbers; use only these facts):\n" + props
                + "\n\nWrite the note now" + (", keeping the [[line]] citations on every sentence." if cfg.get("cite") else "."))
        return cached(h("note", rec["id"], system, user), lambda: note_llm(system, user, 1500))
    return cached(h("note", rec["id"], system, body), lambda: note_llm(system, "TRANSCRIPT:\n" + body + "\n\nWrite the note now.", 1500))


def gate(cfg, rec, dialogue, note, judge_llm) -> str:
    if cfg.get("gate") != "drop_flagged" or not cfg.get("cite"):
        return note
    lines = [l for l in dialogue.splitlines() if l.strip()]

    def _judge():
        kept = []
        for s in SENT_SPLIT.split(note):
            s = s.strip()
            if not s:
                continue
            if len(s.split()) < 3 or s.isupper() or s.endswith(":"):
                kept.append(s)
                continue
            cites = sorted({int(x) for m in CITE.findall(s) for x in m.replace(" ", "").split(",") if x})
            ev = "\n".join(f"{k}: {lines[k - 1]}" for c in cites for k in range(max(1, c - 1), min(len(lines), c + 1) + 1)) or "(no lines cited)"
            txt = judge_llm(JUDGE_SYSTEM, f"SENTENCE:\n{CITE.sub('', s).strip()}\n\nEVIDENCE:\n{ev}", 120)
            first = txt.split("\n", 1)[0].upper()
            bad = "UNSUPPORTED" in first or bool(re.search(r"LEAK\s*=\s*yes", txt, re.I))
            if not bad:
                kept.append(s)
        return "\n".join(kept)
    return cached(h("gate", rec["id"], note), _judge)


def score_note(rec, note, dialogue_for_judge, judge_llm) -> dict:
    from rouge_score import rouge_scorer
    ref = rec["note"]["note"]
    hyp = strip_cites(note)
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = 100 * sc.score(ref, hyp)["rougeL"].fmeasure
    r, hh = med_terms(ref, LEX), med_terms(hyp, LEX)
    tr = 100 * len(r & hh) / max(len(r), 1)
    tp = 100 * len(r & hh) / max(len(hh), 1)

    def _judge():
        a = parse_json(judge_llm(ATTR_SYSTEM, f"TRANSCRIPT:\n{rec['dialogue']}\n\nDRAFT NOTE:\n{hyp}", 500)) or {}
        mis = sum(int(a.get(k) or 0) for k in "ABC")
        items_txt = judge_llm("You extract plan items from clinical notes.", COMPLETE_EXTRACT + ref, 400)
        items = [l.strip("-• ").strip() for l in items_txt.splitlines() if l.strip() and l.strip().upper() != "NONE"]
        found = 0
        if items:
            listing = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items))
            f = parse_json(judge_llm(COMPLETE_CHECK, f"ITEMS:\n{listing}\n\nDRAFT NOTE:\n{hyp}", 300)) or {}
            found = sum(1 for i in range(len(items)) if str(f.get(str(i + 1))).lower() == "true")
        return {"misattrib": mis, "plan_items": len(items), "plan_found": found}
    j = cached(h("score", rec["id"], hyp), _judge)
    return {"rougeL": rl, "term_recall": tr, "term_precision": tp, **j}


def composite(m: dict) -> float:
    plan = 100 * m["plan_found"] / max(m["plan_items"], 1)
    return 0.3 * m["term_recall"] + 0.3 * m["term_precision"] + 0.2 * m["rougeL"] + 0.2 * plan - 40 * m["misattrib"]


def evaluate(cfg, ids, recs, note_llm, judge_llm) -> dict:
    rows = []

    def one(cid):
        rec = recs[cid]
        d = transcript_for(cfg, rec, note_llm)
        n = note_for(cfg, rec, d, note_llm)
        n = gate(cfg, rec, d, n, judge_llm)
        return score_note(rec, n, d, judge_llm)
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, ids))
    agg = {k: sum(r[k] for r in rows) / len(rows) for k in ("rougeL", "term_recall", "term_precision")}
    agg["misattrib"] = sum(r["misattrib"] for r in rows) / len(rows)
    agg["plan_items"] = sum(r["plan_items"] for r in rows)
    agg["plan_found"] = sum(r["plan_found"] for r in rows)
    agg["plan_recall"] = 100 * agg["plan_found"] / max(agg["plan_items"], 1)
    agg["score"] = composite(agg)
    agg["n"] = len(rows)
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in agg.items()}


def note_side(cfg):
    return {k: cfg.get(k) for k in ("prompt", "extra", "cite", "gate", "scaffold")}


def log(md: str):
    with open(NOTEBOOK, "a") as f:
        f.write(md + "\n")


PROPOSER_SYSTEM = """You are running an overnight research loop on an ambient clinical scribe. You propose the next experiment as a JSON config.
Knobs: asr_variant (one of the listed cached variants), asr_correct (0/1), role_map ("none"|"llm"), prompt (one of the listed keys), extra (a short free-text instruction to append to the note prompt, or ""), cite (0/1), gate ("none"|"drop_flagged"; only meaningful with cite=1), scaffold (0/1: first extract cited atomic propositions, then write the note from them).
Ideas from the literature you may draw on: source-grounded proposition scaffolds (CAP, BioNLP 2026); lexicon-guided LLM correction of ASR errors (Google 2024, npj 2026); speaker-role identification before note writing; explicit negatives; plan items in the clinician's own words; deleting unsupported sentences.
Goal: maximise the composite score (0.3*term_recall + 0.3*term_precision + 0.2*ROUGE-L + 0.2*plan_recall - 40*misattributions_per_note) on ASR transcripts, and close the gap to the same config on the human transcript.
Read the notebook of past results. Propose 3 configs that are different from everything tried, each a plausible improvement, favouring changes to the weakest metric. Answer with a JSON list of 3 config objects and nothing else."""


def propose(notebook_tail: str, variants: list[str], tried: set[str], proposer_llm) -> list[dict]:
    txt = proposer_llm(PROPOSER_SYSTEM, f"Cached asr_variants: {variants}\nPrompt keys: {list(PROMPTS)}\n\nNOTEBOOK (most recent last):\n{notebook_tail[-6000:]}", 900)
    m = re.search(r"\[.*\]", txt, re.S)
    out = []
    try:
        for c in (json.loads(m.group(0)) if m else []):
            c = {"asr_variant": c.get("asr_variant", "moss_plain"), "asr_correct": int(bool(c.get("asr_correct", 0))),
                 "role_map": c.get("role_map", "none") if c.get("role_map") in ("none", "llm") else "none",
                 "prompt": c.get("prompt") if c.get("prompt") in PROMPTS else "base", "extra": str(c.get("extra") or "")[:400],
                 "cite": int(bool(c.get("cite", 0))), "gate": c.get("gate") if c.get("gate") in ("none", "drop_flagged") else "none",
                 "scaffold": int(bool(c.get("scaffold", 0)))}
            if c["asr_variant"] in variants and h("cfg", c) not in tried:
                out.append(c)
    except Exception:  # noqa: BLE001
        pass
    return out


def seed_queue(variants):
    q = [{"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "none", "prompt": "base", "extra": "", "cite": 0, "gate": "none"}]
    q += [{"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "llm", "prompt": "base", "extra": "", "cite": 0, "gate": "none"},
          {"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "llm", "prompt": "attrib", "extra": "", "cite": 0, "gate": "none"},
          {"asr_variant": "moss_plain", "asr_correct": 1, "role_map": "llm", "prompt": "attrib", "extra": "", "cite": 0, "gate": "none"},
          {"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "llm", "prompt": "attrib", "extra": "", "cite": 1, "gate": "none"},
          {"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "llm", "prompt": "attrib", "extra": "", "cite": 1, "gate": "drop_flagged"},
          {"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "llm", "prompt": "attrib_strict", "extra": "", "cite": 1, "gate": "drop_flagged"},
          {"asr_variant": "moss_plain", "asr_correct": 1, "role_map": "llm", "prompt": "attrib_strict", "extra": "", "cite": 1, "gate": "drop_flagged"},
          {"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "llm", "prompt": "attrib", "extra": "", "cite": 1, "gate": "none", "scaffold": 1},
          {"asr_variant": "moss_plain", "asr_correct": 0, "role_map": "llm", "prompt": "attrib_strict", "extra": "", "cite": 1, "gate": "drop_flagged", "scaffold": 1}]
    for v in variants:
        if v != "moss_plain":
            q.append({"asr_variant": v, "asr_correct": 0, "role_map": "llm", "prompt": "attrib", "extra": "", "cite": 0, "gate": "none"})
    return q


def mutate(best, variants):
    c = dict(best)
    k = random.choice(["asr_variant", "asr_correct", "role_map", "prompt", "cite", "gate", "extra", "scaffold"])
    if k == "scaffold":
        c[k] = 1 - int(c.get(k, 0)); return c
    if k == "asr_variant":
        c[k] = random.choice(variants)
    elif k in ("asr_correct", "cite"):
        c[k] = 1 - int(c.get(k, 0))
    elif k == "role_map":
        c[k] = "llm" if c.get(k) == "none" else "none"
    elif k == "prompt":
        c[k] = random.choice(list(PROMPTS))
    elif k == "gate":
        c[k] = "drop_flagged" if c.get(k) == "none" else "none"
    else:
        c[k] = random.choice(["", "Use the clinician's own words for the plan.", "Record negatives the patient explicitly denied.",
                              "State who reported each symptom.", "Do not include information about people other than the patient."])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note_url", default="http://100.83.231.108:8004/v1")
    ap.add_argument("--note_model", default="qwen3.8-27b")
    ap.add_argument("--judge_url", default="http://100.83.231.108:8005/v1")
    ap.add_argument("--judge_model", default="gemma-4-26b-a4b-it")
    ap.add_argument("--deadline", default="2026-09-13 08:30")
    ap.add_argument("--max_iters", type=int, default=200)
    args = ap.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    note_llm = LLM(args.note_url, args.note_model, workers=8)
    judge_llm = LLM(args.judge_url, args.judge_model, workers=8)
    recs, dev, test = load_recs()
    variants = sorted(p.name for p in VARIANTS.iterdir() if p.is_dir() and len(list(p.glob("*.json"))) >= len(recs))
    state = json.loads(STATE.read_text()) if STATE.exists() else {"tried": {}, "best": None, "best_dev": None, "best_test": None, "iters": 0}
    if not NOTEBOOK.exists():
        NOTEBOOK.write_text(f"# Autoresearch notebook\n\nStarted {datetime.now():%Y-%m-%d %H:%M}. Dev = {len(dev)} PriMock consultations, test = {len(test)}. "
                            f"Cached ASR variants: {variants}. Composite = 0.3 term_recall + 0.3 term_precision + 0.2 ROUGE-L + 0.2 plan_recall - 40 misattrib/note.\n\n"
                            "| iter | asr | corr | role | prompt | extra | cite | gate | dev score | ΔASR-vs-human | test score | note |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    queue = seed_queue(variants)
    human_cache: dict[str, dict] = {}
    it = state["iters"]
    while it < args.max_iters and datetime.now() < deadline and not STOP.exists():
        if not queue:
            props = propose(NOTEBOOK.read_text(), variants, set(state["tried"]), note_llm)
            queue = props or [mutate(state["best"] or seed_queue(variants)[0], variants)]
        cfg = queue.pop(0)
        key = h("cfg", cfg)
        if key in state["tried"]:
            continue
        it += 1
        t0 = time.time()
        try:
            dev_m = evaluate(cfg, dev, recs, note_llm, judge_llm)
            hk = h("human", note_side(cfg))
            if hk not in human_cache:
                human_cache[hk] = evaluate({**cfg, "asr_variant": "human", "asr_correct": 0, "role_map": "none"}, dev, recs, note_llm, judge_llm)
            hum = human_cache[hk]
            gap = round(dev_m["score"] - hum["score"], 2)
            note = ""
            test_m = None
            improved = state["best_dev"] is None or dev_m["score"] > state["best_dev"]["score"] + 0.5
            if improved:
                test_m = evaluate(cfg, test, recs, note_llm, judge_llm)
                if state["best_test"] is None or test_m["score"] > state["best_test"]["score"]:
                    state.update(best=cfg, best_dev=dev_m, best_test=test_m)
                    note = "ACCEPTED as best"
                else:
                    note = "dev gain did not hold on test"
            state["tried"][key] = {"cfg": cfg, "dev": dev_m, "human": hum, "gap": gap, "test": test_m, "minutes": round((time.time() - t0) / 60, 1)}
            log(f"| {it} | {cfg['asr_variant']} | {cfg['asr_correct']} | {cfg['role_map']} | {cfg['prompt']}{' +scaffold' if cfg.get('scaffold') else ''} | {cfg['extra'][:40]} | {cfg['cite']} | {cfg['gate']} | "
                f"{dev_m['score']} (TR {dev_m['term_recall']}, TP {dev_m['term_precision']}, RL {dev_m['rougeL']}, plan {dev_m['plan_recall']}, mis {dev_m['misattrib']}) | "
                f"{gap:+} (human {hum['score']}) | {test_m['score'] if test_m else ''} | {note} |")
        except Exception as e:  # noqa: BLE001
            log(f"| {it} | {cfg} | | | | | | | ERROR {type(e).__name__}: {str(e)[:120]} | | | |")
            time.sleep(30)
        state["iters"] = it
        STATE.write_text(json.dumps(state, indent=1))
    log(f"\nStopped {datetime.now():%Y-%m-%d %H:%M} after {it} iterations. Best: {json.dumps(state['best'])}\nBest dev: {state['best_dev']}\nBest test: {state['best_test']}\n")


if __name__ == "__main__":
    main()
