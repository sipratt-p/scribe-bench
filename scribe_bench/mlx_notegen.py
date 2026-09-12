"""Generate ACI-Bench notes with an MLX model in one process (no server), same prompts and
output layout as notegen.py so note_score.py / note_judge.py work unchanged.

  python -m scribe_bench.mlx_notegen <mlx_model_dir> <aci_root> <out_dir> --splits test1,test2,test3 \
      --variants humantrans [--cite] [--max_tokens 1200] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from scribe_bench.acibench import load_split
from scribe_bench.notegen import ACI_SYSTEM, CITE_SUFFIX, number_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("aci_root")
    ap.add_argument("out_dir")
    ap.add_argument("--splits", default="test1,test2,test3")
    ap.add_argument("--variants", default="humantrans")
    ap.add_argument("--cite", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, tok = load(args.model)
    system = ACI_SYSTEM + (CITE_SUFFIX if args.cite else "")
    jobs = []
    for split in args.splits.split(","):
        enc = load_split(Path(args.aci_root), split)
        for eid, e in enc.items():
            for v in args.variants.split(","):
                if v in e["variants"]:
                    dst = Path(args.out_dir) / split / v / f"{eid}.json"
                    if not dst.exists():
                        jobs.append((dst, e["variants"][v]))
    jobs = jobs[: args.limit or None]
    print(len(jobs), "jobs", flush=True)
    sampler = make_sampler(temp=0.0)
    for i, (dst, transcript) in enumerate(jobs, 1):
        body = number_lines(transcript) if args.cite else transcript
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": "TRANSCRIPT:\n" + body + "\n\nWrite the note now."}]
        try:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        t0 = time.time()
        text = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        dt = time.time() - t0
        n_out = len(tok.encode(text))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"note": text.strip(), "prompt_tokens": len(prompt) if isinstance(prompt, list) else None,
                                   "completion_tokens": n_out, "seconds": round(dt, 1)}, indent=1))
        print(f"{i}/{len(jobs)} {dst.name} {n_out} tok {dt:.0f}s ({n_out / max(dt, 1e-6):.1f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
