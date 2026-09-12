"""Generate ACI-Bench notes with a local HF model via transformers.generate (no vLLM needed).
Same prompts, output layout and JSON shape as notegen.py, so note_score.py / note_judge.py work unchanged.

  python -m scribe_bench.hf_notegen <model_dir> <aci_root> <out_dir> --splits test1,test2,test3 \
      --variants humantrans [--cite] [--batch 8] [--trust_remote_code]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=1200)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir to apply on top of --model (merged in memory)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 trust_remote_code=args.trust_remote_code).eval()
    if args.adapter:
        from peft import PeftConfig, get_peft_model
        from safetensors.torch import load_file
        pm = get_peft_model(model, PeftConfig.from_pretrained(args.adapter))
        sd = load_file(f"{args.adapter}/adapter_model.safetensors")
        sd = {k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): v
              for k, v in sd.items()}
        missing, unexpected = pm.load_state_dict(sd, strict=False)
        lora_missing = [k for k in missing if "lora_" in k]
        assert not lora_missing and not unexpected, f"adapter load mismatch: missing {lora_missing[:3]} unexpected {unexpected[:3]}"
        model = pm.merge_and_unload().eval()
        print(f"adapter merged in memory: {args.adapter} ({len(sd)} tensors)", flush=True)
    system = ACI_SYSTEM + (CITE_SUFFIX if args.cite else "")

    jobs = []
    for split in args.splits.split(","):
        enc = load_split(Path(args.aci_root), split)
        for eid, e in enc.items():
            for v in args.variants.split(","):
                if v in e["variants"]:
                    dst = Path(args.out_dir) / split / v / f"{eid}.json"
                    if dst.exists():
                        continue
                    body = number_lines(e["variants"][v]) if args.cite else e["variants"][v]
                    msgs = [{"role": "system", "content": system},
                            {"role": "user", "content": "TRANSCRIPT:\n" + body + "\n\nWrite the note now."}]
                    try:
                        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    except TypeError:
                        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    jobs.append((dst, prompt))
    jobs.sort(key=lambda j: str(j[0]))
    if args.limit:
        jobs = jobs[: max(0, args.limit - len([1 for split in args.splits.split(",") for p in (Path(args.out_dir) / split).glob("*/*.json")]))]
    print(len(jobs), "jobs", flush=True)
    jobs.sort(key=lambda j: len(j[1]))
    for b in range(0, len(jobs), args.batch):
        chunk = jobs[b: b + args.batch]
        t0 = time.time()
        inputs = tok([p for _, p in chunk], return_tensors="pt", padding=True).to(model.device)
        gen_kwargs = {}
        try:  # hybrid Mamba models need their own cache object or generation runs without a cache
            import sys as _sys
            mod = _sys.modules[model.__class__.__module__]
            Cache = getattr(mod, "HybridMambaAttentionDynamicCache", None) or getattr(mod, "NemotronHHybridDynamicCache", None)
            if Cache is not None:
                gen_kwargs["past_key_values"] = Cache(model.config, len(chunk), dtype=torch.bfloat16, device=model.device)
                gen_kwargs["use_cache"] = True
        except Exception as e:  # noqa: BLE001
            print("no hybrid cache:", e, flush=True)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id, **gen_kwargs)
        dt = time.time() - t0
        for (dst, _), seq in zip(chunk, out):
            gen = seq[inputs["input_ids"].shape[1]:]
            text = tok.decode(gen, skip_special_tokens=True).strip()
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps({"note": text, "prompt_tokens": int(inputs["input_ids"].shape[1]),
                                       "completion_tokens": int((gen != tok.pad_token_id).sum()),
                                       "seconds": round(dt / len(chunk), 1)}, indent=1))
        print(f"batch {b // args.batch + 1}: {len(chunk)} notes in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
