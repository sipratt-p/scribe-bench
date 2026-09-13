"""Live ASR for the demo: one wav -> both pipelines' transcripts as JSON on stdout.
Runs on the GPU box (models cached there). Called over ssh by the demo server.

  python -m scribe_bench.live_asr <wav> [--device cuda:0]

Output: {"pass1": {"text": ..., "wall_s": ...}, "pass2": {"segments": [...], "raw": ..., "wall_s": ...}}"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_pass1", action="store_true")
    ap.add_argument("--pass1_device", default="auto", help="cuda:N, cpu, or auto (cpu when the GPU has < 14 GB free)")
    ap.add_argument("--out", required=True, help="write the JSON result here (stdout is polluted by NeMo logs)")
    args = ap.parse_args()
    import torch
    out = {}

    if not args.skip_pass1:
        import nemo.collections.asr as nemo_asr
        t0 = time.time()
        p1 = args.pass1_device
        if p1 == "auto":
            free = torch.cuda.mem_get_info(torch.device(args.device))[0] / 2**30 if torch.cuda.is_available() else 0
            p1 = args.device if free >= 14 else "cpu"
        torch.set_num_threads(max(4, (__import__("os").cpu_count() or 8) - 4))
        m = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/nemotron-speech-streaming-en-0.6b", map_location=p1)
        m = m.to(p1).eval()
        hyp = m.transcribe([args.wav], batch_size=1)
        h = hyp[0] if not isinstance(hyp, tuple) else hyp[0][0]
        out["pass1"] = {"text": h.text if hasattr(h, "text") else h, "wall_s": round(time.time() - t0, 1),
                        "model": "nvidia/nemotron-speech-streaming-en-0.6b", "device": str(p1)}
        del m
        torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription
    t0 = time.time()
    M = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
    res = generate_transcription(model, proc, build_transcription_messages(args.wav), max_new_tokens=8192,
                                 do_sample=False, device=device, dtype=torch.bfloat16)
    segs = [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in parse_transcript(res["text"])]
    out["pass2"] = {"segments": segs, "raw": res["text"], "wall_s": round(time.time() - t0, 1), "model": M}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("LIVE_ASR_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
