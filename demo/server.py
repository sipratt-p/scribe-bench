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


NOTEBOOK_REMOTE = os.environ.get("SCRIBE_NOTEBOOK_REMOTE", "beast:~/projects/scribe-bench/autoresearch/notebook.md")
STATE_REMOTE = os.environ.get("SCRIBE_STATE_REMOTE", "beast:~/projects/scribe-bench/autoresearch/state.json")


def _sync_notebook():
    dst = ROOT / "autoresearch/notebook.md"
    try:
        subprocess.run(["scp", "-q", NOTEBOOK_REMOTE, str(dst)], timeout=30, check=False)
        subprocess.run(["scp", "-q", STATE_REMOTE, str(ROOT / "autoresearch/state.json")], timeout=30, check=False)
    except Exception:  # noqa: BLE001
        pass
    return dst.read_text() if dst.exists() else "# no notebook yet\n"


def _md_table_to_html(md: str) -> str:
    out, in_table, header_done = [], False, False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            tag = "th" if not in_table else "td"
            cls = ""
            row_txt = " ".join(cells)
            if "ACCEPTED" in row_txt:
                cls = ' class="accepted"'
            elif "ERROR" in row_txt:
                cls = ' class="err"'
            elif "did not hold" in row_txt or "disagreed" in row_txt:
                cls = ' class="rejected"'
            esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            out.append(("<table><thead>" if not in_table else "") + f"<tr{cls}>" + "".join(f"<{tag}>{esc(c)}</{tag}>" for c in cells) + "</tr>" + ("</thead><tbody>" if not in_table else ""))
            in_table = True
        else:
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            t = line.strip()
            if t.startswith("# "):
                out.append(f"<h1>{t[2:]}</h1>")
            elif t:
                out.append(f"<p>{t.replace('&', '&amp;').replace('<', '&lt;')}</p>")
    if in_table:
        out.append("</tbody></table>")
    return "\n".join(out)


@app.get("/experiments", response_class=HTMLResponse)
def experiments():
    md = _sync_notebook()
    best = ""
    sp = ROOT / "autoresearch/state.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        if st.get("best"):
            best = (f"<div class='best'><b>Accepted best</b> · dev {st['best_dev']['score']} · test {st['best_test']['score']}"
                    f"{' · judge 2 ' + str(st['best_test2']['score']) if st.get('best_test2') else ''}"
                    f"<pre>{json.dumps(st['best'], indent=1)}</pre></div>")
    style = """<style>
    :root{color-scheme:light}
    body{font-family:-apple-system,'Public Sans',Helvetica,Arial,sans-serif;font-size:13.5px;line-height:1.45;margin:0;padding:24px 32px;background:#F4F6F8;color:#172029}
    h1{font-family:Newsreader,Georgia,serif;font-weight:500;font-size:28px;margin:0 0 6px;color:#172029} p{max-width:90ch;color:#5C6B78}
    table{border-collapse:collapse;width:100%;background:#FFFFFF;font-size:12.5px;margin-top:14px;color:#172029} th,td{border-top:1px solid #CAD3DB;padding:6px 8px;text-align:left;vertical-align:top;color:#172029}
    th{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#5C6B78}
    tr.accepted td{background:#DDF1E6;color:#0F3D24;font-weight:600} tr.rejected td{background:#FBEFD2;color:#4A3208} tr.err td{background:#F9E1DF;color:#5A1712}
    .best{background:#DDEEF4;color:#0B3A4A;border-radius:6px;padding:10px 14px;margin:10px 0} pre{font-size:12px;white-space:pre-wrap;margin:6px 0 0;color:#0B3A4A}
    .legend{font-size:12px;color:#5C6B78;margin:8px 0} .legend span{display:inline-block;padding:1px 8px;border-radius:4px;margin-right:8px;color:#172029}
    a{color:#0F6B8A}
    </style>"""
    legend = ("<div class='legend'><span style='background:#DDF1E6'>accepted</span><span style='background:#FBEFD2'>gain on dev, rejected on test or by judge 2</span>"
              "<span style='background:#F9E1DF'>error</span> Columns: dev composite (with term recall TR, term precision TP, ROUGE-L RL, follow-up recall, misattributions per note), "
              "ΔASR-vs-human = same note config on the human transcript, test composite when run. <a href='/'>← back to the demo</a></div>")
    return "<!doctype html><html><head><meta charset='utf-8'><title>Scribe Bench Experiments</title>" + style + "</head><body>" + _md_table_to_html(md).replace("</h1>", "</h1>" + best + legend, 1) + "</body></html>"


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


def _llm(base_url, model, system, user, max_tokens=1500, tries=4):
    from openai import OpenAI
    c = OpenAI(base_url=base_url, api_key="x", timeout=900, max_retries=0)
    last = None
    for i in range(tries):
        try:
            r = c.chat.completions.create(model=model, temperature=0.0, max_tokens=max_tokens,
                                          messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - shared local servers throw transient 500s under load
            last = e
            time.sleep(2 * (i + 1))
    raise last


def _run_job(job, wav: Path):
    from scribe_bench.notegen import PRIMOCK_SYSTEM, CITE_SUFFIX, number_lines
    from scribe_bench.verifier import JUDGE_SYSTEM, SENT_SPLIT, CITE, parse_verdict
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
        from concurrent.futures import ThreadPoolExecutor

        def _judge(cl):
            try:
                txt = _llm(JUDGE_URL, JUDGE_MODEL, JUDGE_SYSTEM, f"SENTENCE:\n{cl['claim']}\n\nEVIDENCE:\n" + ("\n".join(cl["evidence"]) or "(no lines cited)"), 220)
            except Exception as e:  # noqa: BLE001
                txt = f"ERROR {e}"
            cl["label"], cl["leak"] = parse_verdict(txt)
            cl["judge_raw"] = txt
            return cl
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(_judge, claims))

        # 4. attribution judge on both notes, against the diarized transcript
        _step(job, "Judging attribution on both notes")
        judges = {}
        for k, note in (("theirs", theirs_note), ("ours", CITE.sub("", ours_note))):
            try:
                a = parse_json(_llm(JUDGE_URL, JUDGE_MODEL, ATTR_SYSTEM, f"TRANSCRIPT:\n{ours_dialogue}\n\nDRAFT NOTE:\n{note}", 500)) or {}
            except Exception:  # noqa: BLE001
                a = {}
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
