"""Pass-1 ASR: Nemotron-3.5 streaming (cache-aware) over PriMock57 via NeMo's
simulation script. Also runs the same model offline (full context) for reference.

Usage (beast):
  python -m scribe_bench.asr_nemotron <primock_export_dir> <out_dir> --nemo_src <NeMo repo> [--ctx 70,13] [--limit N]

Writes <out_dir>/manifest.json (input), <out_dir>/streaming_<ctx>.json (pred manifest) and
<out_dir>/offline.json."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"


def write_manifest(export_dir: Path, dst: Path, limit: int) -> list[dict]:
    rows = []
    for p in sorted(export_dir.glob("*.json"))[: limit or None]:
        rec = json.loads(p.read_text())
        audio = str(Path(rec["audio"]).resolve())
        rows.append({"audio_filepath": audio, "duration": sf.info(audio).duration,
                     "text": rec["reference_text"], "id": rec["id"]})
    dst.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def run_streaming(nemo_src: Path, manifest: Path, out_dir: Path, ctx: str, batch: int) -> Path:
    script = nemo_src / "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
    out_path = out_dir / f"streaming_{ctx.replace(',', '_')}"
    out_path.mkdir(exist_ok=True)
    cmd = [sys.executable, str(script), f"pretrained_name={MODEL}", f"dataset_manifest={manifest}",
           f"batch_size={batch}", f"att_context_size=[{ctx}]", f"output_path={out_path}", "amp=true"]
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    (out_path / "wall_s.txt").write_text(f"{time.time() - t0:.1f}\n")
    return out_path


def run_offline(manifest: Path, out_dir: Path, batch: int) -> Path:
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    t0 = time.time()
    hyps = model.transcribe([r["audio_filepath"] for r in rows], batch_size=batch)
    dt = time.time() - t0
    out = out_dir / "offline.json"
    with open(out, "w") as f:
        for r, h in zip(rows, hyps):
            text = h.text if hasattr(h, "text") else h
            f.write(json.dumps({"id": r["id"], "audio_filepath": r["audio_filepath"], "text": r["text"],
                                "pred_text": text}) + "\n")
    (out_dir / "offline_wall_s.txt").write_text(f"{dt:.1f}\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--nemo_src", required=True)
    ap.add_argument("--ctx", default="70,13")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_offline", action="store_true")
    ap.add_argument("--skip_streaming", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    write_manifest(Path(args.export_dir), manifest, args.limit)
    if not args.skip_streaming:
        run_streaming(Path(args.nemo_src), manifest, out, args.ctx, args.batch)
    if not args.skip_offline:
        run_offline(manifest, out, args.batch)


if __name__ == "__main__":
    main()
