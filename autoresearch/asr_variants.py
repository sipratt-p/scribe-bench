"""ASR variants for the autoresearch loop (runs on the GPU box, cached per variant).

  python -m autoresearch.asr_variants <export_dir> <out_root> --variant <name> [--ids id1,id2,...] [--device cuda:1]

Variants:
  moss_plain            MOSS-TD, default prompt
  moss_hot50/150/400    MOSS-TD with the top-N lexicon terms as hotwords
  moss_hotcc            MOSS-TD with hotwords drawn from the presenting complaint's lexicon neighbourhood
                        (terms from the lexicon that share a stem with words in the complaint) + top 50
Output: <out_root>/<variant>/<id>.json in the asr_moss.py shape (segments, raw, wall_s)."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def hotwords_for(variant: str, lexicon: list[str], complaint: str | None) -> list[str]:
    if variant == "moss_plain":
        return []
    if variant.startswith("moss_hot") and variant[8:].isdigit():
        return lexicon[: int(variant[8:])]
    if variant == "moss_hotcc":
        stems = {w[:5] for w in re.findall(r"[a-z]{5,}", (complaint or "").lower())}
        related = [t for t in lexicon if t[:5] in stems][:100]
        return list(dict.fromkeys(related + lexicon[:50]))
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_root")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--lexicon", default="data/lexicon.txt")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, build_transcription_messages, generate_transcription

    lexicon = [t.strip() for t in Path(args.lexicon).read_text().splitlines() if t.strip()]
    out = Path(args.out_root) / args.variant
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted(Path(args.export_dir).glob("*.json"))]
    if args.ids:
        keep = set(args.ids.split(","))
        recs = [r for r in recs if r["id"] in keep]
    todo = [r for r in recs if not (out / f"{r['id']}.json").exists()]
    if not todo:
        print("nothing to do")
        return
    device = torch.device(args.device)
    M = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    model = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
    for rec in todo:
        hw = hotwords_for(args.variant, lexicon, rec.get("note", {}).get("presenting_complaint"))
        prompt = DEFAULT_PROMPT.strip() + ("\n热词提示：" + ", ".join(hw) if hw else "")
        t0 = time.time()
        res = generate_transcription(model, proc, build_transcription_messages(rec["audio"], prompt=prompt),
                                     max_new_tokens=8192, do_sample=False, device=device, dtype=torch.bfloat16)
        segs = [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in parse_transcript(res["text"])]
        (out / f"{rec['id']}.json").write_text(json.dumps({"id": rec["id"], "raw": res["text"], "segments": segs,
                                                            "wall_s": round(time.time() - t0, 1), "hotwords": len(hw)}))
        print(rec["id"], len(segs), "segs", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
