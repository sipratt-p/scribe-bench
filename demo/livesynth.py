"""Live decision support during the visit: a separate experiment from the loop/results pipeline.

Replays a PriMock consultation on its audio timeline (streaming-ASR text distributed over the human utterance
timestamps) and every `interval` seconds of visit time asks a local model (DeepSeek V4 Flash on the beast by default)
for an in-visit view meant to change what the clinician does before the patient leaves:
  differential (top 3, each with transcript evidence and what is missing), the one question that best separates the
  top two, act-now red flags, safety-netting not yet given, guideline-based plan suggestions kept apart from what the
  clinician actually stated, plus the scribe view (complaint, history, findings, plan stated, gaps, terms).
Streaming words that fuse into or sit one edit from a medical term are flagged as likely mishearings.
At the end the live timeline is scored against the reference note's diagnosis and plan: time-to-diagnosis, whether the
discriminating question was later asked, red-flag precision, plan-suggestion agreement/contradiction, churn.

Routes: GET /live (page), GET /api/live/{cid}?speed=4&interval=20&model=dsflash (server-sent events).
Sessions log to runs/live_synth/<cid>_<timestamp>.json. Headless scoring over all visits: autoresearch/live_eval.py.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "demo/data"
OUT = ROOT / "runs/live_synth"
BEAST = os.environ.get("SCRIBE_BEAST_IP", "100.83.231.108")
MODELS = {
    "dsflash": (f"http://{BEAST}:8000/v1", "DeepSeek-V4-Flash-Abliterated", "DeepSeek V4 Flash NVFP4 (MoE, DSpark draft), 2 GPUs"),
    "qwen27b": (f"http://{BEAST}:8004/v1", "qwen3.8-27b", "Qwen3.8-27B dense FP8, 1 GPU"),
    "flashnext": (f"http://{BEAST}:8003/v1", "Qwen3.8-Flash-Next-ablit-nvfp4", "Qwen3.8-Flash-Next NVFP4 (MoE), 1 GPU"),
}
THINK_OFF = {"dsflash": {"thinking": False}, "qwen27b": {"enable_thinking": False}, "flashnext": {"enable_thinking": False}}

router = APIRouter()

_LEX: set[str] | None = None
_LEX_LONG: list[str] = []
_ENGLISH: set[str] | None = None
_MED: list[str] = []


def lexicon():
    global _LEX, _LEX_LONG, _ENGLISH
    if _LEX is None:
        terms = [t.strip().lower() for t in (ROOT / "data/lexicon.txt").read_text().splitlines() if t.strip()]
        _LEX = set(terms) | {w for t in terms for w in t.split()}
        _LEX_LONG = sorted({t for t in _LEX if " " not in t and len(t) >= 7})
        try:
            _ENGLISH = {w.strip().lower() for w in open("/usr/share/dict/words")}
        except OSError:
            _ENGLISH = set()
    return _LEX, _LEX_LONG, _ENGLISH


def is_english(w: str) -> bool:
    """The system dictionary lists lemmas, not inflections, so check a few stems too."""
    _, _, english = lexicon()
    cands = {w, w[:-1], w[:-2], w[:-3], w[:-4], w[:-1] + "e", w[:-2] + "e", w[:-3] + "e", w[:-4] + "e"}
    return any(len(c) >= 4 and c in english for c in cands)


def medical_terms():
    """Lexicon single words of 7+ letters that are neither ordinary English words nor one edit from one."""
    global _MED
    if not _MED:
        lex, lex_long, english = lexicon()
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
        eng_long = [w for w in english if len(w) >= 6 and w.isalpha()]
        cand = [t for t in lex_long if t.isalpha() and not is_english(t)]
        _MED = [t for t in cand if not process.extractOne(t, eng_long, scorer=Levenshtein.distance, score_cutoff=1)]
    return _MED


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def misheard(new_words: list[str], recent: list[str], seen: set[str]) -> list[dict]:
    """Flag (a) non-words within 2 edits of a medical term, (b) real words 1 edit from a medical term that are not
    just an inflection of it, (c) runs of 2-3 short words that fuse into a medical term ("met form in" -> metformin)."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    lex, _, _ = lexicon()
    med = medical_terms()
    out = []
    for w in new_words:
        lw = w.lower().strip("'-")
        if len(lw) < 6 or lw in lex or lw in seen or not lw.isalpha():
            continue
        seen.add(lw)
        hit = process.extractOne(lw, med, scorer=Levenshtein.distance, score_cutoff=2)
        if not hit:
            continue
        term, d = hit[0], hit[1]
        if is_english(lw):
            if d == 1 and _stem(lw) != _stem(term) and abs(len(lw) - len(term)) <= 1:
                out.append({"heard": w, "maybe": term, "dist": d})
        elif d <= 2:
            out.append({"heard": w, "maybe": term, "dist": d})
    words = [x.lower() for x in (recent + new_words)]
    for n in (2, 3):
        for i in range(max(0, len(words) - len(new_words) - n + 1), len(words) - n + 1):
            run = words[i: i + n]
            if any(len(x) > 6 for x in run) or not all(x.isalpha() for x in run):
                continue
            fused = "".join(run)
            if len(fused) < 8 or is_english(fused) or fused in seen:
                continue
            if fused in lex:
                seen.add(fused)
                out.append({"heard": " ".join(run), "maybe": fused, "dist": 0})
                continue
            hit = process.extractOne(fused, med, scorer=Levenshtein.distance, score_cutoff=1 if len(fused) < 12 else 2)
            if hit:
                seen.add(fused)
                out.append({"heard": " ".join(run), "maybe": hit[0], "dist": hit[1]})
    return out


SYNTH_SYSTEM = """You are an in-visit clinical decision-support assistant listening to a UK GP consultation as it happens. The transcript comes from a streaming speech recognizer: no speaker labels, sentences may be cut mid-way, medical terms may be misheard. Your job is to help the clinician act before the patient leaves, not to write the note. Maintain a running view and output ONE JSON object only, with these keys:
"complaint": short string, or "" if not yet clear; never guess from small talk.
"history": list of short strings actually stated (symptoms with duration/character, PMH, drugs, allergies, social).
"findings": list of examination findings the clinician has stated.
"differential": list of up to 3 objects, most likely first: {"dx": name, "likelihood": "high"|"medium"|"low", "evidence": [up to 3 short quotes or paraphrases from the transcript], "missing": [up to 2 things that would confirm or exclude it]}. Empty list until there is real evidence.
"next_question": the single question the clinician should ask next to best separate the top two diagnoses (or to rule out the most dangerous one). "" if nothing useful.
"red_flags": list of up to 3 objects {"feature": what was heard, "why": the danger, "action": what to do now}. Only genuine act-now features; an empty list is the normal answer.
"plan_stated": list of actions the clinician has actually said (prescriptions, tests, referrals, follow-up timing).
"safety_netting_stated": list of when-to-return / emergency advice the clinician has actually given.
"safety_netting_suggested": list of up to 3 safety-netting points that fit the working diagnosis and have NOT been given yet.
"plan_suggested": list of up to 4 objects {"item": suggested action, "basis": short guideline or reasoning tag such as "NICE CKS gastroenteritis"} that fit the working diagnosis and have NOT been stated yet. Suggestions only; never present them as the clinician's plan.
"gaps": list of up to 4 history items a GP would normally establish for this complaint that have NOT been covered yet.
"terms": list of up to 12 medical terms heard so far.
Rules: everything under history, findings, plan_stated, safety_netting_stated and evidence must come from the transcript; suggestions live only in the *_suggested keys and next_question. The moment the clinician states something that was in plan_suggested or safety_netting_suggested, move it to plan_stated / safety_netting_stated and remove it from the suggested list. Once a next_question has been answered in the transcript, replace it with a new one or "". Keep earlier items unless contradicted. Prefer updating the previous view to rewriting it. Short phrases."""

REF_SYSTEM = """From this clinician's note extract: "diagnosis": the working diagnosis or impression as a short phrase (or "" if none is stated); "plan": list of plan items (prescriptions, tests, referrals, follow-up, safety-netting) as short phrases. Output one JSON object only."""

TIMELINE_SYSTEM = """You are given a consultation transcript with a time in seconds at the start of each line, a diagnosis, and a list of plan items. Find (a) the earliest time the clinician states or clearly implies that diagnosis to the patient, (b) for each plan item the earliest time it is said. Output one JSON object: {"diagnosis_t": seconds or null, "plan_t": {"1": seconds or null, "2": ...}}."""

SCORE_SYSTEM = """You score an in-visit decision-support timeline against the clinician's own note. You get: the reference diagnosis; the reference plan items; the transcript with times; and a list of snapshots, each with a time, a differential (dx names in order), a next_question, red_flags, and plan_suggested items.
Answer one JSON object:
"dx_first_top3": earliest snapshot index whose differential contains the reference diagnosis (same condition, wording may differ) in any position, or null;
"dx_first_top1": earliest snapshot index where it is first in the differential, or null;
"questions_asked": for each snapshot index that had a non-empty next_question, true if the clinician later asks that question in substance (after that snapshot's time), else false, as an object {index: bool};
"red_flags": for each snapshot that had red flags, an object {index: [true/false per flag]} where true means the feature was genuinely present in the transcript AND is a genuine act-now concern;
"plan_suggested_final": for each plan_suggested item in the LAST snapshot: "agrees" (the clinician's note has the same action), "extra" (reasonable, not in the note), or "contradicts" (the note or transcript goes against it), as a list in order."""

WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _llm(system: str, user: str, max_tokens: int = 900, tries: int = 3, model: str = "dsflash") -> str:
    from openai import OpenAI
    url, name, _ = MODELS.get(model, MODELS["dsflash"])
    c = OpenAI(base_url=url, api_key="x", timeout=180, max_retries=0)
    last = None
    for i in range(tries):
        try:
            r = c.chat.completions.create(model=name, temperature=0.0, max_tokens=max_tokens,
                                          messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                          extra_body={"chat_template_kwargs": THINK_OFF.get(model, {"enable_thinking": False})})
            return r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise last


def parse_json(txt: str):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def timeline(rec: dict) -> list[dict]:
    """Streaming-ASR words distributed over the human utterance timeline, in ~4-word chunks."""
    utts = sorted(rec["reference"]["utterances"], key=lambda u: u["start"])
    words = (rec["theirs"].get("transcript_text") or "").split()
    total = sum(len(u["text"].split()) for u in utts) or 1
    chunks, pos = [], 0
    for i, u in enumerate(utts):
        share = len(u["text"].split()) / total
        n = round(share * len(words)) if i < len(utts) - 1 else len(words) - pos
        seg = words[pos: pos + n]
        pos += n
        if not seg:
            continue
        dur = max(u["end"] - u["start"], 0.5)
        k = max(1, len(seg) // 4)
        step = dur / k
        for j in range(k):
            piece = seg[j * 4: (j + 1) * 4] if j < k - 1 else seg[j * 4:]
            chunks.append({"t": round(u["start"] + j * step, 2), "text": " ".join(piece)})
    return chunks


def timed_transcript(rec: dict) -> str:
    return "\n".join(f"[{int(u['start'])}s] {u['speaker']}: {u['text']}" for u in sorted(rec["reference"]["utterances"], key=lambda u: u["start"]))


def ref_note_text(rec: dict) -> str:
    n = rec["reference"].get("note")
    return n if isinstance(n, str) else (n or {}).get("note", "")


def synthesize_once(text: str, prev: dict | None, upto_t: float, model: str) -> dict:
    user = f"PREVIOUS VIEW:\n{json.dumps(prev) if prev else 'none yet'}\n\nTRANSCRIPT SO FAR ({upto_t:.0f} s into the visit):\n{text}\n\nUpdate the view now."
    return parse_json(_llm(SYNTH_SYSTEM, user, model=model)) or prev or {}


def reference_targets(rec: dict, model: str) -> dict:
    """Reference diagnosis + plan from the clinician's note, and when each was said in the visit."""
    ref = parse_json(_llm(REF_SYSTEM, "NOTE:\n" + ref_note_text(rec), 400, model=model)) or {}
    dx = str(ref.get("diagnosis") or "")
    plan = [str(p) for p in (ref.get("plan") or [])]
    listing = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(plan))
    tl = parse_json(_llm(TIMELINE_SYSTEM, f"TRANSCRIPT:\n{timed_transcript(rec)}\n\nDIAGNOSIS: {dx or '(none)'}\n\nPLAN ITEMS:\n{listing}", 300, model=model)) or {}
    plan_t = {}
    for i, p in enumerate(plan):
        v = (tl.get("plan_t") or {}).get(str(i + 1))
        plan_t[p] = v if isinstance(v, (int, float)) else None
    dxt = tl.get("diagnosis_t")
    return {"diagnosis": dx, "plan": plan, "diagnosis_t": dxt if isinstance(dxt, (int, float)) else None, "plan_t": plan_t}


def score_timeline(rec: dict, snapshots: list[dict], targets: dict, model: str) -> dict:
    """LLM-judged part of the scoring: dx timing, question hit rate, red-flag precision, plan-suggestion agreement."""
    snaps = []
    last_q = None
    for i, s in enumerate(snapshots):
        v = dict(s["synth"])
        if (v.get("next_question") or "") == last_q:
            v["next_question"] = ""  # repeated question: judge it once
        else:
            last_q = v.get("next_question") or ""
        snaps.append(f"[{i}] t={s['t']:.0f}s differential={[d.get('dx') for d in (v.get('differential') or []) if isinstance(d, dict)]} "
                     f"next_question={json.dumps(v.get('next_question') or '')} red_flags={[f.get('feature') for f in (v.get('red_flags') or []) if isinstance(f, dict)]} "
                     f"plan_suggested={[p.get('item') for p in (v.get('plan_suggested') or []) if isinstance(p, dict)]}")
    listing = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(targets["plan"]))
    js = parse_json(_llm(SCORE_SYSTEM, f"REFERENCE DIAGNOSIS: {targets['diagnosis'] or '(none)'}\n\nREFERENCE PLAN:\n{listing}\n\nTRANSCRIPT:\n{timed_transcript(rec)}\n\nSNAPSHOTS:\n" + "\n".join(snaps), 700, model=model)) or {}
    n = len(snapshots)

    def snap_t(i):
        return snapshots[int(i)]["t"] if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n else None
    return {"dx_first_top3_t": snap_t(js.get("dx_first_top3")), "dx_first_top1_t": snap_t(js.get("dx_first_top1")),
            "questions_asked": js.get("questions_asked") or {}, "red_flags": js.get("red_flags") or {},
            "plan_suggested_final": js.get("plan_suggested_final") or []}


def churn(snapshots: list[dict]) -> dict:
    """Items that appeared then vanished, over the keys a clinician would watch; and top-1 diagnosis flips."""
    keys = ("history", "plan_stated", "safety_netting_stated", "red_flags")
    vanished, total = 0, 0
    seen: dict[str, set] = {k: set() for k in keys}
    for s in snapshots:
        v = s["synth"]
        for k in keys:
            items = {json.dumps(x, sort_keys=True) if isinstance(x, dict) else str(x) for x in (v.get(k) or [])}
            vanished += len(seen[k] - items)
            total += len(items)
            seen[k] = items
    dxs = [[d.get("dx") for d in (s["synth"].get("differential") or []) if isinstance(d, dict)] for s in snapshots]
    top1_changes = sum(1 for a, b in zip(dxs, dxs[1:]) if a and b and a[0] != b[0])
    return {"vanished_items": vanished, "items_total": total, "top1_dx_changes": top1_changes}


@router.get("/live", response_class=HTMLResponse)
def live_page():
    return (ROOT / "demo/live.html").read_text()


@router.get("/api/live/{cid}")
def live_stream(cid: str, speed: float = 4.0, interval: float = 20.0, model: str = "dsflash"):
    model = model if model in MODELS else "dsflash"
    rec = json.loads((DATA / f"{cid}.json").read_text())
    chunks = timeline(rec)
    duration = rec.get("duration_s") or (chunks[-1]["t"] + 3 if chunks else 0)
    final_note = (rec.get("ours_v2") or rec.get("ours") or {}).get("note") or ""
    q: "queue.Queue[dict]" = queue.Queue()
    state = {"transcript": [], "synth": None, "snapshots": [], "flags": [], "seen": set(), "pending": False, "synth_chunks": -1, "recent": []}
    lock = threading.Lock()

    def synthesize(upto_t: float):
        with lock:
            text = " ".join(state["transcript"])
            prev = state["synth"]
            state["synth_chunks"] = len(state["transcript"])
        t0 = time.time()
        try:
            js = synthesize_once(text, prev, upto_t, model)
        except Exception as e:  # noqa: BLE001
            js = prev or {"error": str(e)}
        lat = round(time.time() - t0, 1)
        with lock:
            state["synth"] = js
            state["snapshots"].append({"t": upto_t, "synth": js, "latency_s": lat})
            state["pending"] = False
        q.put({"event": "synthesis", "t": upto_t, "latency_s": lat, "snapshot": len(state["snapshots"]) - 1, "synth": js})

    def gen():
        yield f"data: {json.dumps({'event': 'start', 'id': cid, 'duration_s': duration, 'speed': speed, 'interval': interval, 'model': MODELS[model][2], 'chunks': len(chunks)})}\n\n"
        wall0 = time.time()
        next_synth = interval
        i = 0
        while i < len(chunks) or not q.empty() or state["pending"]:
            now_t = (time.time() - wall0) * speed
            while i < len(chunks) and chunks[i]["t"] <= now_t:
                c = chunks[i]
                with lock:
                    state["transcript"].append(c["text"])
                ws = WORD.findall(c["text"])
                flags = misheard(ws, state["recent"], state["seen"])
                state["recent"] = (state["recent"] + ws)[-6:]
                if flags:
                    state["flags"].extend({**f, "t": c["t"]} for f in flags)
                yield f"data: {json.dumps({'event': 'transcript', 't': c['t'], 'text': c['text'], 'flags': flags})}\n\n"
                i += 1
            if now_t >= next_synth and not state["pending"] and state["transcript"]:
                state["pending"] = True
                threading.Thread(target=synthesize, args=(min(now_t, duration),), daemon=True).start()
                next_synth += interval
            try:
                ev = q.get(timeout=0.05)
                yield f"data: {json.dumps(ev)}\n\n"
            except queue.Empty:
                pass
            if i >= len(chunks) and not state["pending"] and q.empty():
                break
            time.sleep(0.02)
        if state["synth_chunks"] < len(chunks):
            state["pending"] = True
            synthesize(duration)
        while not q.empty():
            yield f"data: {json.dumps(q.get())}\n\n"
        yield f"data: {json.dumps({'event': 'scoring'})}\n\n"
        try:
            targets = reference_targets(rec, model)
            sc = score_timeline(rec, state["snapshots"], targets, model)
            sc["churn"] = churn(state["snapshots"])
            sc["targets"] = targets
        except Exception as e:  # noqa: BLE001
            sc = {"error": str(e)}
        final = {"event": "final", "note": final_note, "ref_note": ref_note_text(rec), "score": sc, "flags": state["flags"],
                 "latency": [s["latency_s"] for s in state["snapshots"]], "wall_s": round(time.time() - wall0, 1)}
        yield f"data: {json.dumps(final)}\n\n"
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{cid}_{int(wall0)}.json").write_text(json.dumps({"id": cid, "speed": speed, "interval": interval, "model": model,
                                                                  "snapshots": state["snapshots"], "flags": state["flags"], "score": sc}, indent=1))

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
