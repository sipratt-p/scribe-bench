"""Demo server: two ambient-scribe pipelines side by side on one recording.

  uv run uvicorn demo.server:app --host 127.0.0.1 --port 8700

Cached mode replays real outputs from demo/data (built by demo/build_data.py).
Live mode (POST /api/run) runs both pipelines on any wav: ASR over ssh on the GPU box,
notes on the local Qwen server, verifier + attribution judge on the local Gemma server.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "demo/data"
UPLOADS = ROOT / "demo/uploads"
AUDIO = ROOT / "data/primock57/mixed"
GPU_HOST = os.environ.get("SCRIBE_GPU_HOST", "beast")
GPU_PROJECT = os.environ.get("SCRIBE_GPU_PROJECT", "~/projects/scribe-bench")
NOTE_URL = os.environ.get("SCRIBE_NOTE_URL", "http://localhost:8600/v1")
NOTE_MODEL = os.environ.get("SCRIBE_NOTE_MODEL", "EigenLabs--Qwen3.8-27B-4bit")
JUDGE_URL = os.environ.get("SCRIBE_JUDGE_URL", "http://localhost:8500/v1")
JUDGE_MODEL = os.environ.get("SCRIBE_JUDGE_MODEL", "gemma4-vision")

app = FastAPI(title="scribe-bench demo")
JOBS: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "demo/index.html").read_text()


@app.get("/api/consultations")
def consultations():
    return json.loads((DATA / "index.json").read_text())


@app.get("/api/consultation/{cid}")
def consultation(cid: str):
    p = DATA / f"{cid}.json"
    if not p.exists():
        raise HTTPException(404)
    return json.loads(p.read_text())


@app.get("/api/audio/{cid}")
def audio(cid: str):
    for p in (AUDIO / f"{cid}.wav", UPLOADS / f"{cid}.wav"):
        if p.exists():
            return FileResponse(p, media_type="audio/wav")
    raise HTTPException(404)


@app.get("/api/summary")
def summary():
    p = ROOT / "demo/summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.post("/api/upload")
async def upload(file: UploadFile):
    UPLOADS.mkdir(exist_ok=True)
    cid = "upload_" + re.sub(r"[^A-Za-z0-9]+", "_", Path(file.filename or "audio").stem)[:40] + "_" + uuid.uuid4().hex[:6]
    raw = UPLOADS / f"{cid}.orig"
    raw.write_bytes(await file.read())
    wav = UPLOADS / f"{cid}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return {"id": cid, "audio": f"/api/audio/{cid}"}


@app.post("/api/run")
def run(body: dict):
    cid = body["id"]
    wav = AUDIO / f"{cid}.wav" if (AUDIO / f"{cid}.wav").exists() else UPLOADS / f"{cid}.wav"
    if not wav.exists():
        raise HTTPException(404, "no audio")
    job = {"id": uuid.uuid4().hex[:8], "cid": cid, "status": "queued", "steps": [], "result": None, "error": None}
    JOBS[job["id"]] = job
    threading.Thread(target=_run_job, args=(job, wav), daemon=True).start()
    return {"job": job["id"]}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    return j


def _step(job, text):
    job["steps"].append({"t": round(time.time(), 1), "text": text})


def _llm(base_url, model, system, user, max_tokens=1500):
    from openai import OpenAI
    c = OpenAI(base_url=base_url, api_key="x", timeout=900)
    r = c.chat.completions.create(model=model, temperature=0.0, max_tokens=max_tokens,
                                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content or ""


def _run_job(job, wav: Path):
    from scribe_bench.notegen import PRIMOCK_SYSTEM, CITE_SUFFIX, number_lines
    from scribe_bench.verifier import JUDGE_SYSTEM, SENT_SPLIT, CITE
    from scribe_bench.note_judge import ATTR_SYSTEM, parse_json
    from scribe_bench.textnorm import normalize
    try:
        job["status"] = "running"
        # 1. ASR on the GPU box (both passes in one process)
        _step(job, "Uploading audio to the GPU box and running pass 1 (streaming ASR) and pass 2 (diarized ASR)")
        remote = f"/tmp/scribe_live_{job['id']}.wav"
        subprocess.run(["scp", "-q", str(wav), f"{GPU_HOST}:{remote}"], check=True)
        gpu = os.environ.get("SCRIBE_GPU", "0")
        cmd = (f"cd {GPU_PROJECT} && source .venv/bin/activate && CUDA_VISIBLE_DEVICES={gpu} "
               f"python -m scribe_bench.live_asr {remote} --out {remote}.json > {remote}.log 2>&1; cat {remote}.json")
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", GPU_HOST, cmd], capture_output=True, text=True, timeout=1800)
        asr = json.loads(r.stdout)
        theirs_text = asr["pass1"]["text"]
        ours_segs = asr["pass2"]["segments"]
        ours_dialogue = "\n".join(f"[{s['speaker']}] {s['text']}" for s in ours_segs)
        _step(job, f"Pass 1 done in {asr['pass1']['wall_s']} s, pass 2 done in {asr['pass2']['wall_s']} s ({len(ours_segs)} speaker turns)")

        # 2. notes
        _step(job, "Writing the note from the streaming transcript (their pipeline)")
        theirs_note = _llm(NOTE_URL, NOTE_MODEL, PRIMOCK_SYSTEM, "TRANSCRIPT:\n" + theirs_text + "\n\nWrite the note now.")
        _step(job, "Writing the cited note from the diarized transcript (proposed pipeline)")
        ours_note = _llm(NOTE_URL, NOTE_MODEL, PRIMOCK_SYSTEM + CITE_SUFFIX,
                         "TRANSCRIPT:\n" + number_lines(ours_dialogue) + "\n\nWrite the note now.")

        # 3. verifier on the cited note
        _step(job, "Verifying every cited claim against its transcript lines")
        lines = [l for l in ours_dialogue.splitlines() if l.strip()]
        claims = []
        for s in SENT_SPLIT.split(ours_note):
            s = s.strip()
            if len(s.split()) < 3 or s.isupper() or s.endswith(":"):
                continue
            cites = sorted({int(x) for m in CITE.findall(s) for x in m.replace(" ", "").split(",") if x})
            ev = sorted({(k, lines[k - 1]) for c in cites for k in range(max(1, c - 1), min(len(lines), c + 1) + 1)})
            claims.append({"claim": CITE.sub("", s).strip(), "cites": cites, "evidence": [f"{k}: {t}" for k, t in ev]})
        for cl in claims:
            txt = _llm(JUDGE_URL, JUDGE_MODEL, JUDGE_SYSTEM, f"SENTENCE:\n{cl['claim']}\n\nEVIDENCE:\n" + ("\n".join(cl["evidence"]) or "(no lines cited)"), 120)
            first = txt.split("\n", 1)[0].upper()
            cl["label"] = "UNSUPPORTED" if "UNSUPPORTED" in first else "PARTIAL" if "PARTIAL" in first else "SUPPORTED" if "SUPPORTED" in first else "ERROR"
            cl["leak"] = bool(re.search(r"LEAK\s*=\s*yes", txt, re.I))
            cl["judge_raw"] = txt

        # 4. attribution judge on both notes, against the diarized transcript
        _step(job, "Judging attribution on both notes")
        judges = {}
        for k, note in (("theirs", theirs_note), ("ours", CITE.sub("", ours_note))):
            a = parse_json(_llm(JUDGE_URL, JUDGE_MODEL, ATTR_SYSTEM, f"TRANSCRIPT:\n{ours_dialogue}\n\nDRAFT NOTE:\n{note}", 500)) or {}
            judges[k] = {"attribution": {kk: a.get(kk, 0) for kk in "ABC"}, "attr_examples": a.get("examples", [])[:3]}

        # 5. medical-term counts (no reference for uploads; term overlap between the two transcripts)
        from scribe_bench.note_score import load_lexicon
        lex = {normalize(t) for t in load_lexicon(str(ROOT / "data/lexicon.txt"))}
        t1 = {w for w in normalize(theirs_text).split() if w in lex}
        t2 = {w for w in normalize(" ".join(s["text"] for s in ours_segs)).split() if w in lex}
        job["result"] = {
            "theirs": {"transcript_text": theirs_text, "note": theirs_note, "judge": judges["theirs"],
                       "terms": sorted(t1), "terms_only_here": sorted(t1 - t2), "wall_s": asr["pass1"]["wall_s"]},
            "ours": {"segments": ours_segs, "transcript_text": ours_dialogue, "note": ours_note, "claims": claims,
                     "judge": judges["ours"], "terms": sorted(t2), "terms_only_here": sorted(t2 - t1), "wall_s": asr["pass2"]["wall_s"]},
        }
        job["status"] = "done"
        _step(job, "Done")
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"[:500]
