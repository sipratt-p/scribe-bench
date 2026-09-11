"""Generic NeMo offline ASR runner (Parakeet-TDT, Canary-Qwen, Canary) over PriMock57.

  python -m scribe_bench.asr_nemo <export_dir> <out_dir> --model nvidia/parakeet-tdt-0.6b-v3 [--batch 4] [--limit N]

Writes <out_dir>/pred.json (jsonl: id, audio_filepath, text, pred_text) and wall_s.txt.
Long recordings (5-10 min) are transcribed whole; Parakeet handles long-form natively,
Canary-Qwen is chunked by NeMo's transcribe when needed."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted(Path(args.export_dir).glob("*.json"))][: args.limit or None]
    files = [str(Path(r["audio"]).resolve()) for r in recs]
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    t0 = time.time()
    hyps = model.transcribe(files, batch_size=args.batch)
    dt = time.time() - t0
    if isinstance(hyps, tuple):  # some models return (best, all)
        hyps = hyps[0]
    with open(out / "pred.json", "w") as f:
        for r, h in zip(recs, hyps):
            text = h.text if hasattr(h, "text") else h
            f.write(json.dumps({"id": r["id"], "audio_filepath": str(Path(r["audio"]).resolve()),
                                "text": r["reference_text"], "pred_text": text}) + "\n")
    (out / "wall_s.txt").write_text(f"{dt:.1f}\n")
    print(args.model, len(files), "files", f"{dt:.1f}s")


if __name__ == "__main__":
    main()
