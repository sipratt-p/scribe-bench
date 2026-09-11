"""Pass-2 ASR: MOSS-Transcribe-Diarize over full PriMock57 recordings.

Usage (beast):
  python -m scribe_bench.asr_moss <primock_export_dir> <out_dir> [--hotwords lexicon.txt] [--limit N]

Writes <out_dir>/<cid>.json with raw text, parsed segments (start, end, speaker, text)
and wall-clock seconds."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, build_transcription_messages, generate_transcription

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--hotwords", default=None, help="text file, one term per line; top --n_hot used")
    ap.add_argument("--n_hot", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    prompt = DEFAULT_PROMPT
    if args.hotwords:
        terms = [t.strip() for t in Path(args.hotwords).read_text().splitlines() if t.strip()][: args.n_hot]
        prompt = DEFAULT_PROMPT.strip() + "\n热词提示：" + ", ".join(terms)
        print("hotword prompt with", len(terms), "terms")

    recs = sorted(Path(args.export_dir).glob("*.json"))
    if args.limit:
        recs = recs[: args.limit]
    for p in recs:
        rec = json.loads(p.read_text())
        dst = out / f"{rec['id']}.json"
        if dst.exists():
            continue
        t0 = time.time()
        messages = build_transcription_messages(rec["audio"], prompt=prompt)
        res = generate_transcription(model, processor, messages, max_new_tokens=8192, do_sample=False,
                                     device=device, dtype=torch.bfloat16)
        dt = time.time() - t0
        segs = [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text}
                for s in parse_transcript(res["text"])]
        dst.write_text(json.dumps({"id": rec["id"], "raw": res["text"], "segments": segs,
                                   "wall_s": dt, "generated_tokens": res["generated_tokens"]}, indent=1))
        print(f"{rec['id']}: {len(segs)} segs, {res['generated_tokens']} tok, {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
