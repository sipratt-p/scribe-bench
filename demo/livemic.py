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

from demo.livesynth import BEAST, MODELS, THINK, _slug, lookup, misheard, new_dx_to_lookup, synthesize_once

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/live_mic"
ASR_WS = os.environ.get("SCRIBE_ASR_WS", f"ws://{BEAST}:8090/v1/realtime")
ASR_MODEL = os.environ.get("SCRIBE_ASR_MODEL", "voxtral-realtime")
router = APIRouter()

_WORD = re.compile(r"\S+")


@router.get("/mic", response_class=HTMLResponse)
def mic_page():
    return (ROOT / "demo/mic.html").read_text()


class Session:
    """One live conversation: transcript chunks with wall-clock times, the running view, lookups, activity."""

    def __init__(self, model: str, interval: float, pack: str):
        self.model, self.interval, self.pack = model, interval, pack
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
    def add_text(self, delta: str):
        """A transcription delta from the recogniser; timestamped on arrival."""
        if not delta.strip():
            return
        t = self.now()
        words = _WORD.findall(delta)
        flags = misheard(words, self.recent, self.seen)
        self.recent = (self.recent + words)[-40:]
        with self.lock:
            self.transcript.append({"t": t, "text": delta})
            self.flags.extend({**f, "t": t} for f in flags)
        self.emit({"event": "transcript", "t": t, "text": delta, "flags": flags})

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
            text = " ".join(c["text"] for c in self.transcript)
            prev = self.synth
            refs = dict(self.refs)
            upto_t = self.now()
            self.synth_words = len(text.split())
        self.act("synthesis", "started", f"{self.synth_words} words of transcript")
        t0 = time.time()
        try:
            js = synthesize_once(text, prev, upto_t, self.model, refs, think=THINK)
        except Exception as e:  # noqa: BLE001
            js = prev or {"error": str(e)}
        lat = round(time.time() - t0, 1)
        with self.lock:
            self.synth = js
            self.snapshots.append({"t": upto_t, "synth": js, "latency_s": lat})
            self.pending = False
            todo = new_dx_to_lookup(js, self.refs, self.inflight)
        dxs = [d.get("dx") for d in (js.get("differential") or []) if isinstance(d, dict)]
        for r in (js.get("revisions") or []):
            if isinstance(r, dict) and (r.get("was") or r.get("now")):
                self.act("revised", "done", f"{r.get('was')} → {r.get('now')} · because: {r.get('because')}")
        self.act("synthesis", "done", f"differential {dxs}" + (" · next question set" if js.get("next_question") else "")
                 + (f" · {len(js.get('red_flags') or [])} red flag(s)" if js.get("red_flags") else ""), int(1000 * lat))
        self.emit({"event": "synthesis", "t": upto_t, "latency_s": lat, "snapshot": len(self.snapshots) - 1, "synth": js})
        for dx in todo:
            self.pool.submit(self.do_lookup, dx, upto_t)

    def maybe_tick(self):
        """Called from the event loop every second: start a synthesis when the interval has passed and new words exist."""
        with self.lock:
            due = (not self.pending) and (time.time() - self.t0 >= self.interval * (len(self.snapshots) + 1) - 0.01 or
                                          (self.snapshots and time.time() - self.t0 - self.snapshots[-1]["t"] >= self.interval))
            words = sum(len(c["text"].split()) for c in self.transcript)
            if due and words > self.synth_words and words >= 8:
                self.pending = True
            else:
                due = False
        if due:
            self.pool.submit(self.synthesize)

    def dump(self) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{int(self.t0)}.json"
        p.write_text(json.dumps({"t0": self.t0, "model": self.model, "interval": self.interval, "pack": self.pack, "transcript": self.transcript,
                                 "snapshots": self.snapshots, "flags": self.flags, "activity": self.activity, "lookups": self.refs}, indent=1))
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
            sess.emit({"event": "asr_status", "status": "done", "detail": f"{len((ev.get('text') or '').split())} words final"})
        elif typ == "error":
            sess.emit({"event": "asr_status", "status": "error", "detail": ev.get("error")})
        elif typ == "session.created":
            sess.emit({"event": "asr_status", "status": "connected", "detail": ASR_MODEL})


@router.websocket("/ws/mic")
async def ws_mic(ws: WebSocket, model: str = "qwen27b", interval: float = 20.0, pack: str = "clinical"):
    await ws.accept()
    model = model if model in MODELS else "qwen27b"
    sess = Session(model, interval, pack)
    await ws.send_json({"event": "start", "id": "mic", "duration_s": 0, "speed": 1, "interval": interval, "model": MODELS[model][2], "asr": ASR_MODEL})
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
