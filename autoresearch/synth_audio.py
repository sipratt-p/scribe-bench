"""Synthetic two-voice audio for ACI-Bench encounters (Kokoro TTS), so the ASR pipelines can be
tested on a second, US-English clinical corpus that has reference transcripts AND notes.
Clearly synthetic: clean studio speech, no overlap, no room noise. Use it for term-loss and
attribution trends, not absolute WER.

  python -m autoresearch.synth_audio <aci_root> <out_dir> --split test1 [--limit N] [--device cuda:1]

Writes <out_dir>/<encounter_id>.wav (16 kHz mono) and <out_dir>/<encounter_id>.json with the
utterance list (speaker, start, end, text) in the PriMock export shape, plus reference_text/dialogue/note,
so the same scorers work unchanged."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scribe_bench.acibench import load_split  # noqa: E402

VOICES = {"doctor": "am_michael", "patient": "af_heart"}
SR_TTS, SR_OUT = 24000, 16000


def resample(a: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    n = int(len(a) * sr_out / sr_in)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aci_root")
    ap.add_argument("out_dir")
    ap.add_argument("--split", default="test1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gap_s", type=float, default=0.6)
    args = ap.parse_args()
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", device=args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    enc = load_split(Path(args.aci_root), args.split)
    for eid, e in list(enc.items())[: args.limit or None]:
        if (out / f"{eid}.wav").exists():
            continue
        dialogue = e["variants"]["humantrans"]
        pieces, utts, t = [], [], 0.0
        for line in dialogue.splitlines():
            m = re.match(r"\[(doctor|patient)\]\s*(.+)", line.strip())
            if not m:
                continue
            spk, text = m.group(1), m.group(2).strip()
            if not text:
                continue
            audio = np.concatenate([a for _, _, a in pipe(text, voice=VOICES[spk])]) if text else np.zeros(1)
            a16 = resample(np.asarray(audio, dtype=np.float32), SR_TTS, SR_OUT)
            gap = np.zeros(int(args.gap_s * SR_OUT), dtype=np.float32)
            utts.append({"speaker": spk.capitalize(), "start": round(t, 3), "end": round(t + len(a16) / SR_OUT, 3), "text": text})
            pieces += [a16, gap]
            t += (len(a16) + len(gap)) / SR_OUT
        wav = np.concatenate(pieces) if pieces else np.zeros(SR_OUT, dtype=np.float32)
        sf.write(out / f"{eid}.wav", wav, SR_OUT)
        rec = {"id": eid, "audio": str((out / f"{eid}.wav").resolve()), "utterances": utts,
               "reference_text": " ".join(u["text"] for u in utts), "dialogue": dialogue,
               "note": {"note": e["note"], "presenting_complaint": None}, "synthetic": True}
        (out / f"{eid}.json").write_text(json.dumps(rec))
        (out / "rttm").mkdir(exist_ok=True)
        (out / "rttm" / f"{eid}.rttm").write_text("".join(
            f"SPEAKER {eid} 1 {u['start']:.3f} {u['end'] - u['start']:.3f} <NA> <NA> {u['speaker']} <NA> <NA>\n" for u in utts))
        print(eid, len(utts), "utts", round(t / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
