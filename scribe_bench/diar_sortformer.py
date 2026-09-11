"""NVIDIA pass-2 equivalent: Sortformer diarization + a NeMo ASR transcript with word timestamps,
combined into speaker-attributed segments (the two-model pipeline MOSS-TD replaces with one pass).

  python -m scribe_bench.diar_sortformer <export_dir> <out_dir> --asr nvidia/parakeet-tdt-0.6b-v3 \
      --diar nvidia/diar_sortformer_4spk-v1 [--limit N]

Writes <out_dir>/<cid>.json in the same shape as asr_moss.py (segments with start/end/speaker/text),
so asr_score.py computes WER, term error and DER on it directly."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def assign_speakers(words: list[dict], turns: list[tuple[float, float, str]]) -> list[dict]:
    """Give each word the speaker whose turn overlaps its midpoint most (nearest turn if none)."""
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
    ap.add_argument("--asr", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--diar", default="nvidia/diar_sortformer_4spk-v1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import SortformerEncLabelModel

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted(Path(args.export_dir).glob("*.json"))][: args.limit or None]
    asr = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr)
    diar = SortformerEncLabelModel.from_pretrained(args.diar)
    diar.eval()
    for rec in recs:
        dst = out / f"{rec['id']}.json"
        if dst.exists():
            continue
        audio = str(Path(rec["audio"]).resolve())
        t0 = time.time()
        hyp = asr.transcribe([audio], timestamps=True, batch_size=1)
        h = hyp[0] if not isinstance(hyp, tuple) else hyp[0][0]
        words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
                 for w in h.timestamp["word"]]
        pred = diar.diarize(audio=[audio], batch_size=1)
        turns = []
        for line in pred[0]:  # "start end speaker_k"
            s, e, spk = line.split()
            turns.append((float(s), float(e), spk))
        segs = assign_speakers(words, turns)
        dt = time.time() - t0
        dst.write_text(json.dumps({"id": rec["id"], "raw": h.text, "segments": segs, "turns": turns,
                                   "wall_s": dt}, indent=1))
        print(f"{rec['id']}: {len(words)} words, {len(turns)} turns, {len({t[2] for t in turns})} spk, {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
