"""Live synthesis experiment: real-time analysis of a streaming transcript while the visit is happening.

Separate from the loop/results pipeline. Replays a PriMock consultation on its audio timeline (the streaming-ASR
text distributed over the human utterance timestamps), and every `interval` seconds of visit time asks a local
model (Qwen3.8-27B on the beast vLLM) to update a running synthesis: complaint, history, findings, plan,
safety-netting, red flags, gaps still to ask, terms heard. Also flags streaming words that sit one edit away from
a lexicon term ("possible misheard term": what pass 2 will fix). At the end, compares the last synthesis against
the best pass-2 note and records when each final plan item first appeared live.

Routes: GET /live (page), GET /api/live/{cid}?speed=4&interval=20 (server-sent events).
Sessions are logged to runs/live_synth/<cid>_<timestamp>.json.
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
# per-server way to switch reasoning off
THINK_OFF = {"dsflash": {"thinking": False}, "qwen27b": {"enable_thinking": False}, "flashnext": {"enable_thinking": False}}

router = APIRouter()

_LEX: set[str] | None = None
_LEX_LONG: list[str] = []
_ENGLISH: set[str] | None = None


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


SYNTH_SYSTEM = """You are a live clinical scribe assistant listening to a UK GP consultation as it happens. The transcript comes from a streaming speech recognizer: there are no speaker labels, sentences may be cut mid-way, and medical terms may be misheard. Maintain a running synthesis of the visit so far. Output one JSON object only, with these keys:
"complaint": one short string (the presenting complaint), or "" if not yet clear;
"history": list of short strings (symptoms with duration/character, relevant past history, medications, allergies, social context) stated so far;
"findings": list of examination or observation findings the clinician has stated;
"plan": list of actions the clinician has stated so far (prescriptions, tests, referrals, follow-up timing);
"safety_netting": list of when-to-return / emergency advice given so far;
"red_flags": list of clinically concerning features mentioned (max 4);
"gaps": list of up to 5 things a GP would normally establish for this complaint that have NOT been covered yet;
"terms": list of up to 15 medical terms heard so far (drugs, conditions, anatomy, tests).
Rules: include only what was actually said; never invent. Keep earlier items unless the transcript contradicts them. Prefer updating the previous synthesis to rewriting it. Short phrases, no sentences longer than 15 words."""

COMPARE_SYSTEM = """You compare a final clinical note's plan items against a sequence of live synthesis snapshots taken during the visit. For each final plan item, find the EARLIEST snapshot index whose "plan" or "safety_netting" list already contains that item in substance (same action, even if worded differently). Output one JSON object mapping each item number to the earliest snapshot index, or null if no snapshot had it."""

WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _llm(system: str, user: str, max_tokens: int = 700, tries: int = 3, model: str = "qwen27b") -> str:
    from openai import OpenAI
    url, name, _ = MODELS.get(model, MODELS["qwen27b"])
    c = OpenAI(base_url=url, api_key="x", timeout=120, max_retries=0)
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


_MED: list[str] = []


def is_english(w: str) -> bool:
    """The system dictionary lists lemmas, not inflections, so check a few stems too."""
    _, _, english = lexicon()
    cands = {w, w[:-1], w[:-2], w[:-3], w[:-1] + "e", w[:-2] + "e", w[:-3] + "e"}
    return any(len(c) >= 3 and c in english for c in cands)


def medical_terms():
    """Lexicon single words of 7+ letters that are not ordinary English words: drug names, conditions, anatomy."""
    global _MED
    if not _MED:
        lex, lex_long, english = lexicon()
        _MED = [t for t in lex_long if t.isalpha() and not is_english(t)]
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
    lex, _, english = lexicon()
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
            if fused in lex:  # the words fuse into an exact lexicon term: strongest signal
                seen.add(fused)
                out.append({"heard": " ".join(run), "maybe": fused, "dist": 0})
                continue
            hit = process.extractOne(fused, med, scorer=Levenshtein.distance, score_cutoff=1 if len(fused) < 12 else 2)
            if hit:
                seen.add(fused)
                out.append({"heard": " ".join(run), "maybe": hit[0], "dist": hit[1]})
    return out


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
        user = (f"PREVIOUS SYNTHESIS:\n{json.dumps(prev) if prev else 'none yet'}\n\nTRANSCRIPT SO FAR ({upto_t:.0f} s into the visit):\n{text}\n\nUpdate the synthesis now.")
        try:
            js = parse_json(_llm(SYNTH_SYSTEM, user, model=model)) or prev or {}
        except Exception as e:  # noqa: BLE001
            js = prev or {"error": str(e)}
        with lock:
            state["synth"] = js
            state["snapshots"].append({"t": upto_t, "synth": js, "latency_s": round(time.time() - t0, 1)})
            state["pending"] = False
        q.put({"event": "synthesis", "t": upto_t, "latency_s": round(time.time() - t0, 1), "snapshot": len(state["snapshots"]) - 1, "synth": js})

    def compare():
        from scribe_bench.note_judge import COMPLETE_EXTRACT
        items_txt = _llm("You extract plan items from clinical notes.", COMPLETE_EXTRACT + re.sub(r"\[\[[\d,\s]+\]\]", "", final_note), 400, model=model)
        items = [l.strip("-• ").strip() for l in items_txt.splitlines() if l.strip() and l.strip().upper() != "NONE"]
        if not items or not state["snapshots"]:
            return {"items": items, "first_seen": {}, "snapshots": [s["t"] for s in state["snapshots"]]}
        snaps = "\n".join(f"[{i}] t={s['t']:.0f}s plan={json.dumps(s['synth'].get('plan', []))} safety_netting={json.dumps(s['synth'].get('safety_netting', []))}"
                          for i, s in enumerate(state["snapshots"]))
        listing = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items))
        js = parse_json(_llm(COMPARE_SYSTEM, f"FINAL PLAN ITEMS:\n{listing}\n\nSNAPSHOTS:\n{snaps}", 300, model=model)) or {}
        first = {}
        for i, it in enumerate(items):
            v = js.get(str(i + 1))
            first[it] = state["snapshots"][int(v)]["t"] if isinstance(v, int) and 0 <= v < len(state["snapshots"]) else None
        return {"items": items, "first_seen": first, "snapshots": [s["t"] for s in state["snapshots"]]}

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
        # final synthesis on the complete transcript (unless the last one already saw it), then compare against the pass-2 note
        if state["synth_chunks"] < len(chunks):
            state["pending"] = True
            synthesize(duration)
        while not q.empty():
            yield f"data: {json.dumps(q.get())}\n\n"
        try:
            cmp = compare()
        except Exception as e:  # noqa: BLE001
            cmp = {"error": str(e)}
        final = {"event": "final", "note": final_note, "note_metrics": (rec.get("ours_v2") or rec.get("ours") or {}).get("metrics") or (rec.get("ours_v2") or rec.get("ours") or {}).get("note_metrics"),
                 "compare": cmp, "flags": state["flags"], "wall_s": round(time.time() - wall0, 1)}
        yield f"data: {json.dumps(final)}\n\n"
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{cid}_{int(wall0)}.json").write_text(json.dumps({"id": cid, "speed": speed, "interval": interval, "model": model,
                                                                  "snapshots": state["snapshots"], "flags": state["flags"], "compare": cmp}, indent=1))

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
