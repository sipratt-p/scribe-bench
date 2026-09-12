"""Minimal LoRA SFT loop (PEFT + torch, no TRL) for models where TRL's loss patches misbehave
(hybrid Mamba/MoE under a sharded device map).

  python -m scribe_bench.sft_train_min --model <dir> --data data/sft --out runs/lora_x \
      --targets q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,in_proj --device_map auto [--merge]

Prompt tokens are masked with -100; the model's own forward computes the loss."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def encode(tok, messages, max_len):
    prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    prompt_ids = list(prompt_ids["input_ids"]) if hasattr(prompt_ids, "keys") else list(prompt_ids)
    full_ids = list(full_ids["input_ids"]) if hasattr(full_ids, "keys") else list(full_ids)
    if len(full_ids) > max_len or len(prompt_ids) >= len(full_ids):
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return torch.tensor(full_ids), torch.tensor(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--grad_acc", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    rows = [json.loads(l)["messages"] for l in open(f"{args.data}/train.jsonl")]
    vrows = [json.loads(l)["messages"] for l in open(f"{args.data}/valid.jsonl")]
    train = [e for e in (encode(tok, m, args.max_len) for m in rows) if e is not None]
    valid = [e for e in (encode(tok, m, args.max_len) for m in vrows) if e is not None][:32]
    if args.limit:
        train = train[: args.limit]
    vocab = len(tok)
    assert all(int(x.max()) < vocab and int(x.min()) >= 0 for x, _ in train), "token id outside vocab"
    print(f"train {len(train)} valid {len(valid)} (dropped {len(rows) - len(train)} over max_len)", flush=True)

    dm = "auto" if args.device_map == "auto" else {"": int(args.device_map)}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dm,
                                                 trust_remote_code=args.trust_remote_code)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=args.targets[3:] if args.targets.startswith("re:") else args.targets.split(","))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    first_device = next(model.parameters()).device

    steps_total = math.ceil(len(train) * args.epochs / args.grad_acc)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * steps_total)), steps_total)

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for ids, lab in valid:
                out = model(input_ids=ids[None].to(first_device), labels=lab[None].to(first_device))
                tot += float(out.loss); n += 1
        model.train()
        return tot / max(n, 1)

    log = []
    model.train()
    t0 = time.time()
    step = 0
    micro = 0
    run_loss = 0.0
    order = list(range(len(train)))
    n_micro_total = int(len(train) * args.epochs)
    while micro < n_micro_total:
        random.shuffle(order)
        for i in order:
            if micro >= n_micro_total:
                break
            ids, lab = train[i]
            out = model(input_ids=ids[None].to(first_device), labels=lab[None].to(first_device))
            (out.loss / args.grad_acc).backward()
            run_loss += float(out.loss)
            micro += 1
            if micro % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1:
                    n_steps = 1 if step == 1 else (9 if step == 10 else 10)
                    rec = {"step": step, "of": steps_total, "loss": round(run_loss / args.grad_acc / n_steps, 4),
                           "lr": sched.get_last_lr()[0], "elapsed_min": round((time.time() - t0) / 60, 1)}
                    if step % 50 == 0:
                        rec["eval_loss"] = round(evaluate(), 4)
                    log.append(rec); print(json.dumps(rec), flush=True)
                    run_loss = 0.0
    rec = {"step": step, "eval_loss": round(evaluate(), 4), "train_runtime_min": round((time.time() - t0) / 60, 1)}
    log.append(rec); print(json.dumps(rec), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "train_log.json").write_text(json.dumps(log, indent=1))
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{args.out}/merged", safe_serialization=True)
        tok.save_pretrained(f"{args.out}/merged")
        print("merged ->", f"{args.out}/merged", flush=True)


if __name__ == "__main__":
    main()
