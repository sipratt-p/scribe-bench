"""Live decision support on real microphone audio.

Browser (getUserMedia → AudioWorklet → 16 kHz PCM16) ──WebSocket /ws/mic──▶ this server ──WebSocket──▶ streaming ASR (vLLM
realtime endpoint, Voxtral Mini 4B Realtime) ──transcription.delta──▶ timestamped chunks ──▶ the same tick loop, hold rules,
lookups and activity log as the replay view (demo/livesynth.py) ──JSON events──▶ browser.

Routes:
  GET  /mic            the page
  WS   /ws/mic?model=&interval=&pack=   one session: binary frames = PCM16 mono 16 kHz audio; text frames = JSON control
                       ({"type":"stop"}). Server sends the same event shapes as /api/live/{cid}: start, transcript, synthesis,
                       activity, plus asr_status and final (no scoring: there is no reference note for a live conversation).
Env:
  SCRIBE_ASR_WS   ws URL of the realtime ASR server   (default ws://{BEAST}:8090/v1/realtime)
  SCRIBE_ASR_MODEL served model name                   (default voxtral-realtime)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from demo.livesynth import BEAST, MODELS, THINK, _slug, lookup, misheard, new_dx_to_lookup
from demo.packs import CANDIDATE_HELP_SYS, PACKS, pack_public

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/live_mic"
ASR_WS = os.environ.get("SCRIBE_ASR_WS", f"ws://{BEAST}:8090/v1/realtime")
ASR_MODEL = os.environ.get("SCRIBE_ASR_MODEL", "voxtral-realtime")
router = APIRouter()

_WORD = re.compile(r"\S+")
_SENT_END = re.compile(r"[.?!]\s")
_QWORD = re.compile(r"^(what|why|how|when|where|which|who|tell me|talk me|walk me|can you|could you|would you|do you|did you|have you|are you|is there|was there|describe|explain|give me|any )", re.I)
FAST_SYS = """You classify one sentence from a live interview transcript (no speaker labels). Reply with one JSON object only:
{"is_question": true|false, "asked_by": "interviewer"|"candidate"|"unclear", "paraphrase": "<the question in at most 12 words, or empty>", "competency": "<one of: technical depth, problem solving, ownership and delivery, communication, collaboration, leadership or influence, learning and adaptability, other, or empty>", "bias_issue": "<empty, or a few words naming the protected characteristic if the sentence is an interviewer question about age, family, marital status, pregnancy, health, disability, religion, nationality, origin, or anything not job-related>"}
Interviewer questions ask the candidate about their experience, skills, decisions or motivation; candidate questions ask about the role, team, process or logistics."""
TURN_SYS = """You are a live assistant for the INTERVIEWER. You get the last interviewer question and the candidate's answer to it (from a speech recogniser, no speaker labels, may be cut off). Reply with one JSON object only:
{"specificity": "specific"|"vague"|"unverified", "evidence": "<at most 15 words quoting the most concrete thing the candidate said, or empty>", "competency": "<one of: technical depth, problem solving, ownership and delivery, communication, collaboration, leadership or influence, learning and adaptability, other>", "next_probe": "<the single best follow-up to ask now, phrased to say aloud: turn a vague answer into specifics (what exactly did you do, what was the result, how did you measure it), or empty if the answer was specific and complete>", "concern": "<at most 12 words if the answer revealed a gap or contradiction, else empty>"}
Never suggest questions about protected characteristics. Short, calm wording."""


def _reachable(base_url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3).read(200)
        return True
    except Exception:  # noqa: BLE001
        return False


@router.get("/mic", response_class=HTMLResponse)
def mic_page():
    return (ROOT / "demo/mic.html").read_text()


@router.get("/api/packs")
def packs():
    return {k: pack_public(v) for k, v in PACKS.items()}


class Session:
    """One live conversation: transcript chunks with wall-clock times, the running view, lookups, activity."""

    def __init__(self, model: str, interval: float, pack: str):
        self.model, self.interval, self.pack = model, interval, pack
        self.P = PACKS[pack]
        self.context: dict = {}
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.transcript: list[dict] = []        # {"t": s, "text": str}
        self.synth: dict | None = None
        self.snapshots: list[dict] = []
        self.refs: dict = {}
        self.inflight: set = set()
        self.activity: list[dict] = []
        self.flags: list[dict] = []
        self.recent: list[str] = []
        self.seen: set[str] = set()
        self.buf = ""
        self.sent_buf = ""            # text waiting for a sentence boundary (fast lane)
        self.last_delta = time.time()
        self.last_full = 0.0
        self.words_at_full = 0
        self.questions: list[dict] = []
        self.turns: list[dict] = []
        self.last_turn_check = 0.0
        self.pending = False
        self.synth_words = 0
        self.pool = ThreadPoolExecutor(3)
        self.events: asyncio.Queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        self.stopped = False

    def now(self) -> float:
        return round(time.time() - self.t0, 1)

    def emit(self, ev: dict):
        self.loop.call_soon_threadsafe(self.events.put_nowait, ev)

    def act(self, task: str, status: str, detail: str, ms: int | None = None, **extra):
        ev = {"event": "activity", "t": self.now(), "task": task, "status": status, "detail": detail, "ms": ms, **extra}
        self.activity.append(ev)
        self.emit(ev)

    # ---- transcript -------------------------------------------------------------------------------------------------
    def add_text(self, delta: str, flush: bool = False):
        """A transcription delta from the recogniser (sub-word pieces, leading spaces mark word starts). Buffer until a
        word boundary so the transcript never contains split words; timestamp on arrival."""
        self.buf += delta
        cut = len(self.buf) if flush else max(self.buf.rfind(" "), self.buf.rfind("\n"))
        if cut <= 0 or not self.buf[:cut].strip():
            return
        piece, self.buf = self.buf[:cut], self.buf[cut:]
        t = self.now()
        words = _WORD.findall(piece)
        flags = misheard(words, self.recent, self.seen) if self.P.mishearings else []
        self.recent = (self.recent + words)[-40:]
        with self.lock:
            self.transcript.append({"t": t, "text": piece})
            self.flags.extend({**f, "t": t} for f in flags)
            self.last_delta = time.time()
        self.emit({"event": "transcript", "t": t, "text": piece.strip(), "flags": flags})
        if self.P.name in ("interview", "candidate"):
            self.sent_buf += piece
            while True:
                m = _SENT_END.search(self.sent_buf)
                if not m:
                    break
                sent, self.sent_buf = self.sent_buf[: m.end()].strip(), self.sent_buf[m.end():]
                if sent.endswith("?") or _QWORD.match(sent):
                    self.pool.submit(self.fast_question, sent, t)

    def fast_question(self, sent: str, t: float):
        """Fast lane: one short model call per candidate sentence, ~1 s, so questions and bias issues show within seconds."""
        t0 = time.time()
        try:
            from demo.livesynth import _llm, parse_json
            js = parse_json(_llm(FAST_SYS, f"SENTENCE: {sent}", max_tokens=120, model=self.model)) or {}
        except Exception as e:  # noqa: BLE001
            js = {"error": str(e)[:80]}
        if not js.get("is_question"):
            return
        who = js.get("asked_by") or "unclear"
        if who == "unclear":
            low = " " + sent.lower() + " "
            who = "interviewer" if (" you " in low or " your " in low or low.lstrip().startswith((" tell", " describe", " walk", " talk"))) else who
        q = {"t": t, "asked_by": who, "paraphrase": js.get("paraphrase") or sent[:120], "competency": js.get("competency") or "", "bias_issue": js.get("bias_issue") or "", "sentence": sent, "ms": int(1000 * (time.time() - t0))}
        with self.lock:
            self.questions.append(q)
        self.emit({"event": "question", **q})
        if q["asked_by"] != "candidate":
            if self.P.name == "candidate":
                self.pool.submit(self.help_answer, q, t)   # help lane: answer support for the question just asked
            else:
                self.pool.submit(self.turn_update, t)   # a new question means the previous answer is complete

    def help_answer(self, q: dict, t: float):
        """Candidate pack help lane: one call per interviewer question, ~3-5 s, with the CV/notes and recent transcript."""
        t0 = time.time()
        with self.lock:
            recent = "".join(c["text"] for c in self.transcript)[-3000:]
        ctx = "".join(f"\n\n{k.upper()}:\n{v.strip()[:3500]}" for k, v in self.context.items() if v and v.strip())
        try:
            from demo.livesynth import _llm, parse_json
            js = parse_json(_llm(CANDIDATE_HELP_SYS, f"{ctx.lstrip()}\n\nCONVERSATION SO FAR (recent):\n{recent}\n\nQUESTION JUST ASKED: {q['sentence']}", max_tokens=700, model=self.model)) or {}
        except Exception as e:  # noqa: BLE001
            js = {"error": str(e)[:80]}
        ev = {"event": "help", "t": t, "question": q["paraphrase"], "sentence": q["sentence"], "ms": int(1000 * (time.time() - t0)),
              **{k: js.get(k) or ([] if k != "code" and k != "type" else "") for k in ("type", "clarify", "outline", "architecture", "code", "pitfalls", "from_your_experience", "likely_follow_ups")}}
        with self.lock:
            q["help"] = {k: ev[k] for k in ("type", "outline", "architecture", "code")}
            self.turns.append(ev)
        self.emit(ev)
        self.act("help", "done", f"{ev['type']} question · {len(ev['outline'])} outline points" + (" · code" if ev["code"] else "") + (" · architecture" if ev["architecture"] else ""), ev["ms"])
        if q["bias_issue"]:
            self.act("bias guard", "alert", f"{q['bias_issue']}: \"{sent[:90]}\"", q["ms"])

    # ---- view -------------------------------------------------------------------------------------------------------
    def do_lookup(self, dx: str, t: float):
        self.act("lookup", "started", f"references: {dx}")
        t0 = time.time()
        try:
            r = lookup(dx, self.model)
        except Exception as e:  # noqa: BLE001
            r = {"condition": dx, "empty": True, "error": str(e)}
        with self.lock:
            self.refs[_slug(dx)] = r
        if r.get("empty"):
            self.act("lookup", "empty", f"{dx}: nothing usable", int(1000 * (time.time() - t0)))
        else:
            self.act("lookup", "done", f"{dx} → {r.get('nhs_entry') or 'Wikipedia (low trust)'}: {len(r.get('key_questions') or [])} questions, "
                     f"{len(r.get('red_flags') or [])} red flags", int(1000 * (time.time() - t0)), sources=r.get("sources"), dx=dx, tier=r.get("tier", "nhs"))

    def synthesize(self):
        with self.lock:
            text = "".join(c["text"] for c in self.transcript).strip()
            prev = self.synth
            refs = dict(self.refs)
            upto_t = self.now()
            self.synth_words = len(text.split())
        self.act("synthesis", "started", f"{self.synth_words} words of transcript" + (f" · {len(self.questions)} questions from the fast lane" if self.questions else ""))
        t0 = time.time()
        try:
            ctx = dict(self.context)
            if self.questions:
                ctx["questions_live"] = "\n".join(f"[{q['t']:.0f}s] {q['asked_by']}: {q['paraphrase']}" + (f"  (bias: {q['bias_issue']})" if q['bias_issue'] else "")
                                                 + (f"  → answer {q['specificity']}" + (f", evidence: {q['evidence']}" if q.get('evidence') else "") if q.get('specificity') else "") for q in self.questions[-40:])
            js = self.P.synthesize(text, prev, upto_t, self.model, refs, ctx, think=THINK)
        except Exception as e:  # noqa: BLE001
            js = prev or {"error": str(e)}
        lat = round(time.time() - t0, 1)
        with self.lock:
            self.synth = js
            self.snapshots.append({"t": upto_t, "synth": js, "latency_s": lat})
            self.pending = False
            todo = new_dx_to_lookup(js, self.refs, self.inflight) if self.P.lookups else []
        dxs = [d.get("dx") for d in (js.get("differential") or []) if isinstance(d, dict)] if self.P.lookups else []
        for r in (js.get("revisions") or []):
            if isinstance(r, dict) and (r.get("was") or r.get("now")):
                self.act("revised", "done", f"{r.get('was')} → {r.get('now')} · because: {r.get('because')}")
        if self.P.name == "clinical":
            done = f"differential {dxs}" + (" · next question set" if js.get("next_question") else "") + (f" · {len(js.get('red_flags') or [])} red flag(s)" if js.get("red_flags") else "")
        else:
            cov = [c.get("name") for c in (js.get("competencies") or []) if isinstance(c, dict) and c.get("status") == "covered"]
            done = f"stage {js.get('stage')} · {len(js.get('claims') or [])} claims · covered {cov}" + (" · probe set" if js.get("next_probe") else "") + (f" · {len(js.get('bias_guard') or [])} bias-guard item(s)" if js.get("bias_guard") else "")
        if js.get("_stale"):
            done = "STALE: " + js["_stale"]
        self.act("synthesis", "done", done, int(1000 * lat))
        self.emit({"event": "synthesis", "t": upto_t, "latency_s": lat, "snapshot": len(self.snapshots) - 1, "synth": js})
        for dx in todo:
            self.pool.submit(self.do_lookup, dx, upto_t)

    MIN_GAP_S = 8.0   # never rebuild the full view more often than this

    def turn_update(self, t: float):
        """Turn lane: after a candidate answer ends, one short call scores that answer and refreshes the next probe (~2 s)."""
        with self.lock:
            qs = [q for q in self.questions if q["asked_by"] != "candidate" and not q.get("bias_issue")]
            if not qs:
                return
            # score the latest unscored interviewer question whose answer is long enough; the answer ends at the next question
            q = None
            for cand in reversed(qs):
                if cand.get("scored"):
                    break
                q = cand
            if q is None:
                return
            later = [x["t"] for x in self.questions if x["t"] > q["t"]]
            t_end = min(later) if later else 1e9
            answer = "".join(c["text"] for c in self.transcript if q["t"] < c["t"] <= t_end).strip()
            if len(answer.split()) < 12:
                return
            q["scored"] = True
        t0 = time.time()
        try:
            from demo.livesynth import _llm, parse_json
            js = parse_json(_llm(TURN_SYS, f"QUESTION: {q['sentence']}\n\nANSWER SO FAR:\n{answer[-2500:]}", max_tokens=180, model=self.model)) or {}
        except Exception as e:  # noqa: BLE001
            js = {"error": str(e)[:80]}
        ev = {"event": "probe", "t": t, "question": q["paraphrase"], "specificity": js.get("specificity") or "unverified", "evidence": js.get("evidence") or "",
              "competency": js.get("competency") or q.get("competency") or "", "next_probe": js.get("next_probe") or "", "concern": js.get("concern") or "", "ms": int(1000 * (time.time() - t0))}
        with self.lock:
            q.update({k: ev[k] for k in ("specificity", "evidence", "next_probe", "concern")})
            self.turns.append(ev)
        self.emit(ev)
        self.act("turn", "done", f"{ev['specificity']} answer to \"{q['paraphrase'][:60]}\"" + (f" · probe: {ev['next_probe'][:70]}" if ev["next_probe"] else " · no probe needed"), ev["ms"])

    def maybe_tick(self):
        """Called from the event loop every second. A full view is rebuilt when EITHER the interval has passed, OR a turn
        seems to have ended (>= 1.5 s with no new words after >= 25 new words), with at least MIN_GAP_S between rebuilds."""
        now = time.time()
        with self.lock:
            quiet = (now - self.last_delta) >= 1.5
            words = sum(len(c["text"].split()) for c in self.transcript)
        if quiet and self.P.name == "interview" and now - self.last_turn_check >= 1.0:
            self.last_turn_check = now
            self.pool.submit(self.turn_update, self.now())
        with self.lock:
            if self.pending:
                return
            new_words = words - self.synth_words
            since_full = now - self.last_full
            interval_due = since_full >= self.interval
            turn_done = new_words >= 25 and quiet and since_full >= self.MIN_GAP_S
            if new_words <= 0 or words < 8 or not (interval_due or turn_done):
                return
            self.pending = True
            self.last_full = now
        self.pool.submit(self.synthesize)

    def dump(self) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{int(self.t0)}.json"
        p.write_text(json.dumps({"t0": self.t0, "model": self.model, "interval": self.interval, "pack": self.pack, "context": self.context, "transcript": self.transcript,
                                 "snapshots": self.snapshots, "flags": self.flags, "activity": self.activity, "lookups": self.refs, "questions": self.questions, "turns": self.turns}, indent=1))
        return p


async def asr_reader(asr, sess: Session):
    async for raw in asr:
        try:
            ev = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        typ = ev.get("type")
        if typ == "transcription.delta":
            sess.add_text(ev.get("delta") or "")
        elif typ == "transcription.done":
            sess.add_text("", flush=True)
            sess.emit({"event": "asr_status", "status": "done", "detail": f"{len((ev.get('text') or '').split())} words final"})
        elif typ == "error":
            sess.emit({"event": "asr_status", "status": "error", "detail": ev.get("error")})
        elif typ == "session.created":
            sess.emit({"event": "asr_status", "status": "connected", "detail": ASR_MODEL})


@router.websocket("/ws/mic")
async def ws_mic(ws: WebSocket, model: str = "gemma_beast", interval: float = 20.0, pack: str = "interview"):
    await ws.accept()
    model = model if model in MODELS else "gemma_beast"
    pack = pack if pack in PACKS else "interview"
    fallback = None
    if not _reachable(MODELS[model][0]):
        for alt in ("gemma_beast", "gemma", "qwen27b"):
            if alt != model and _reachable(MODELS[alt][0]):
                fallback, model = f"{MODELS[model][2]} unreachable, using {MODELS[alt][2]}", alt
                break
    sess = Session(model, interval, pack)
    if fallback:
        sess.act("writer", "fallback", fallback)
    await ws.send_json({"event": "start", "id": "mic", "duration_s": 0, "speed": 1, "interval": interval, "model": MODELS[model][2], "asr": ASR_MODEL, "pack": pack_public(PACKS[pack])})
    try:
        asr = await websockets.connect(ASR_WS, max_size=None, ping_interval=20)
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"event": "asr_status", "status": "error", "detail": f"cannot reach ASR at {ASR_WS}: {e}"})
        await ws.close()
        return
    await asr.send(json.dumps({"type": "session.update", "model": ASR_MODEL}))
    await asr.send(json.dumps({"type": "input_audio_buffer.commit"}))
    reader = asyncio.create_task(asr_reader(asr, sess))

    async def pump_events():
        while True:
            ev = await sess.events.get()
            await ws.send_json(ev)

    async def ticker():
        while True:
            await asyncio.sleep(1.0)
            sess.maybe_tick()

    pump = asyncio.create_task(pump_events())
    tick = asyncio.create_task(ticker())
    audio_bytes = 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                audio_bytes += len(msg["bytes"])
                await asr.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(msg["bytes"]).decode()}))
            elif msg.get("text"):
                try:
                    ctl = json.loads(msg["text"])
                except Exception:  # noqa: BLE001
                    ctl = {}
                if ctl.get("type") == "stop":
                    break
                if ctl.get("type") == "context":
                    sess.context = {f["key"]: str(ctl.get(f["key"]) or "")[:6000] for f in PACKS[pack].context_fields}
                    sess.t0 = time.time()  # the clock starts when the context is in and audio is about to flow
                    sess.act("context", "done", ", ".join(f"{k}: {len(v)} chars" for k, v in sess.context.items() if v) or "no documents supplied")
    except WebSocketDisconnect:
        pass
    finally:
        sess.stopped = True
        tick.cancel()
        try:
            await asr.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            await asyncio.wait_for(reader, timeout=8)
        except Exception:  # noqa: BLE001
            reader.cancel()
        try:
            await asr.close()
        except Exception:  # noqa: BLE001
            pass
        # one last view over everything heard, then flush events
        if sum(len(c["text"].split()) for c in sess.transcript) > sess.synth_words:
            await asyncio.get_running_loop().run_in_executor(None, sess.synthesize)
        await asyncio.sleep(0.2)
        pump.cancel()
        while not sess.events.empty():
            try:
                await ws.send_json(sess.events.get_nowait())
            except Exception:  # noqa: BLE001
                break
        p = sess.dump()
        try:
            await ws.send_json({"event": "final", "note": "", "score": None, "flags": sess.flags, "lookups": sess.refs, "latency": [s["latency_s"] for s in sess.snapshots],
                                "wall_s": round(time.time() - sess.t0, 1), "audio_s": round(audio_bytes / 32000, 1), "saved": str(p.relative_to(ROOT))})
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
