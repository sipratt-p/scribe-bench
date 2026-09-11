"""Pass-2 via Nemotron 3 Nano Omni served by vLLM (audio in, speaker-attributed transcript out).

  vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --port 8010 --trust-remote-code \
       --max-model-len 65536 --limit-mm-per-prompt audio=1
  python -m scribe_bench.asr_omni <export_dir> <out_dir> --base_url http://localhost:8010/v1 \
       --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 [--limit N]

The prompt asks for one line per utterance: "[mm:ss.s-mm:ss.s] [Sxx] text". Output is parsed into the
same segments shape as asr_moss.py so asr_score.py scores WER, term error and DER on it. Audio longer
than --chunk_s seconds is split into chunks with a small overlap and offsets are added back."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import soundfile as sf
from openai import OpenAI

PROMPT = ("Transcribe this recording of a medical consultation verbatim. Identify the speakers and label them "
          "S01, S02, ... in order of first appearance. Output one line per utterance in exactly this format and nothing else:\n"
          "[start-end] [Sxx] words\n"
          "where start and end are seconds with one decimal, e.g. [12.4-15.9] [S01] Good morning, how can I help?")
LINE = re.compile(r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*\[(S\d+)\]\s*(.+)")


def chunks(path: str, chunk_s: float, overlap_s: float):
    data, sr = sf.read(path, dtype="int16")
    n = len(data)
    step = int((chunk_s - overlap_s) * sr)
    for start in range(0, n, step):
        seg = data[start: start + int(chunk_s * sr)]
        buf = io.BytesIO()
        sf.write(buf, seg, sr, format="WAV")
        yield start / sr, base64.b64encode(buf.getvalue()).decode()
        if start + int(chunk_s * sr) >= n:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunk_s", type=float, default=600.0)
    ap.add_argument("--overlap_s", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="x")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted(Path(args.export_dir).glob("*.json"))][: args.limit or None]
    for rec in recs:
        dst = out / f"{rec['id']}.json"
        if dst.exists():
            continue
        t0 = time.time()
        segs, raw_all = [], []
        for offset, b64 in chunks(str(Path(rec["audio"]).resolve()), args.chunk_s, args.overlap_s):
            r = client.chat.completions.create(
                model=args.model, temperature=0.0, max_tokens=8192,
                messages=[{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": PROMPT}]}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            raw = r.choices[0].message.content or ""
            raw_all.append(raw)
            for line in raw.splitlines():
                m = LINE.match(line.strip())
                if m:
                    s, e, spk, txt = m.groups()
                    segs.append({"start": float(s) + offset, "end": float(e) + offset, "speaker": spk, "text": txt.strip()})
        dt = time.time() - t0
        dst.write_text(json.dumps({"id": rec["id"], "raw": "\n".join(raw_all), "segments": segs, "wall_s": dt}, indent=1))
        print(f"{rec['id']}: {len(segs)} segs, {len({s['speaker'] for s in segs})} spk, {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
