"""Canary-Qwen 2.5B (NeMo SALM speech-LLM) over PriMock57, chunked at --chunk_s seconds.

  python -m scribe_bench.asr_canary_qwen <export_dir> <out_dir> [--chunk_s 30] [--limit N]

Writes <out_dir>/pred.json (jsonl: id, audio_filepath, text, pred_text) like asr_nemo.py."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="nvidia/canary-qwen-2.5b")
    ap.add_argument("--chunk_s", type=float, default=30.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    from nemo.collections.speechlm2.models import SALM

    model = SALM.from_pretrained(args.model).bfloat16().eval().cuda()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted(Path(args.export_dir).glob("*.json"))][: args.limit or None]
    tmp = Path(tempfile.mkdtemp())
    t0 = time.time()
    with open(out / "pred.json", "w") as f:
        for rec in recs:
            audio = str(Path(rec["audio"]).resolve())
            data, sr = sf.read(audio, dtype="float32")
            n = int(args.chunk_s * sr)
            paths = []
            for i, start in enumerate(range(0, len(data), n)):
                p = tmp / f"{rec['id']}_{i:03d}.wav"
                sf.write(p, data[start: start + n], sr)
                paths.append(str(p))
            texts = []
            for b in range(0, len(paths), args.batch):
                batch = paths[b: b + args.batch]
                prompts = [[{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}",
                             "audio": [p]}] for p in batch]
                with torch.inference_mode():
                    ids = model.generate(prompts=prompts, max_new_tokens=256)
                texts.extend(model.tokenizer.ids_to_text(x.cpu()) for x in ids)
            pred = " ".join(t.strip() for t in texts)
            f.write(json.dumps({"id": rec["id"], "audio_filepath": audio, "text": rec["reference_text"],
                                "pred_text": pred}) + "\n")
            f.flush()
            print(rec["id"], len(paths), "chunks", flush=True)
    (out / "wall_s.txt").write_text(f"{time.time() - t0:.1f}\n")
    print(args.model, len(recs), "files", f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
