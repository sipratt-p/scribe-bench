"""NVIDIA pass-2 equivalent: Sortformer diarization + a NeMo ASR transcript with word timestamps,
combined into speaker-attributed segments (the two-model pipeline MOSS-TD replaces with one pass).

Runs as three separate processes because loading Sortformer and Parakeet in one process
faults with cudaErrorIllegalAddress on the third file (NeMo 3.0):

  python -m scribe_bench.diar_sortformer <export_dir> <out_dir> --phase asr   --asr nvidia/parakeet-tdt-0.6b-v3
  python -m scribe_bench.diar_sortformer <export_dir> <out_dir> --phase diar  --diar nvidia/diar_sortformer_4spk-v1
  python -m scribe_bench.diar_sortformer <export_dir> <out_dir> --phase combine

Final output: <out_dir>/<cid>.json in the same shape as asr_moss.py (segments with
start/end/speaker/text), so asr_score.py computes WER, term error and DER on it directly."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def assign_speakers(words: list[dict], turns: list[tuple[float, float, str]]) -> list[dict]:
    """Give each word the speaker whose turn overlaps its midpoint (nearest turn if none)."""
    segs: list[dict] = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        best, best_d = None, 1e9
        for s, e, spk in turns:
            if s <= mid <= e:
                best, best_d = spk, 0
                break
            d = min(abs(mid - s), abs(mid - e))
            if d < best_d:
                best, best_d = spk, d
        spk = best or "S00"
        if segs and segs[-1]["speaker"] == spk and w["start"] - segs[-1]["end"] < 1.0:
            segs[-1]["end"] = w["end"]
            segs[-1]["text"] += " " + w["word"]
        else:
            segs.append({"start": w["start"], "end": w["end"], "speaker": spk, "text": w["word"]})
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--phase", choices=["asr", "diar", "combine"], required=True)
    ap.add_argument("--asr", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--diar", default="nvidia/diar_sortformer_4spk-v1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted(Path(args.export_dir).glob("*.json"))][: args.limit or None]

    if args.phase == "asr":
        import nemo.collections.asr as nemo_asr
        asr = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr)
        t0 = time.time()
        for rec in recs:
            dst = work / f"{rec['id']}.asr.json"
            if dst.exists():
                continue
            hyp = asr.transcribe([str(Path(rec["audio"]).resolve())], timestamps=True, batch_size=1)
            h = hyp[0] if not isinstance(hyp, tuple) else hyp[0][0]
            words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in h.timestamp["word"]]
            dst.write_text(json.dumps({"text": h.text, "words": words}))
            print(rec["id"], len(words), "words", flush=True)
        (work / "asr_wall_s.txt").write_text(f"{time.time() - t0:.1f}\n")

    elif args.phase == "diar":
        from nemo.collections.asr.models import SortformerEncLabelModel
        diar = SortformerEncLabelModel.from_pretrained(args.diar)
        diar.eval()
        t0 = time.time()
        for rec in recs:
            dst = work / f"{rec['id']}.diar.json"
            if dst.exists():
                continue
            pred = diar.diarize(audio=[str(Path(rec["audio"]).resolve())], batch_size=1)
            turns = []
            for line in pred[0]:
                s, e, spk = line.split()
                turns.append((float(s), float(e), spk))
            dst.write_text(json.dumps({"turns": turns}))
            print(rec["id"], len(turns), "turns", len({t[2] for t in turns}), "spk", flush=True)
        (work / "diar_wall_s.txt").write_text(f"{time.time() - t0:.1f}\n")

    else:
        wall = sum(float((work / f).read_text()) for f in ("asr_wall_s.txt", "diar_wall_s.txt") if (work / f).exists())
        n = 0
        for rec in recs:
            a, d = work / f"{rec['id']}.asr.json", work / f"{rec['id']}.diar.json"
            if not (a.exists() and d.exists()):
                continue
            words = json.loads(a.read_text())["words"]
            turns = [tuple(t) for t in json.loads(d.read_text())["turns"]]
            segs = assign_speakers(words, turns)
            (out / f"{rec['id']}.json").write_text(json.dumps({"id": rec["id"], "raw": json.loads(a.read_text())["text"],
                                                                 "segments": segs, "turns": turns,
                                                                 "wall_s": wall / max(len(recs), 1)}, indent=1))
            n += 1
        print("combined", n, "files")


if __name__ == "__main__":
    main()
