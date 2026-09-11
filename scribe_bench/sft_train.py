"""LoRA SFT of the note model with TRL (GPU box).

  python -m scribe_bench.sft_train --model /mnt/models/Qwen3.8-27B --data data/sft --out runs/lora_qwen38 \
      [--epochs 1] [--max_len 4096] [--lr 1e-4] [--r 32]

Assistant-only loss (completion_only_loss), bf16, gradient checkpointing. Saves the adapter to --out
and a merged bf16 model to --out/merged for vLLM serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("json", data_files={"train": f"{args.data}/train.jsonl", "valid": f"{args.data}/valid.jsonl"})

    # drop rows whose full chat exceeds max_len so no assistant target is truncated away
    def fits(ex):
        ids = tok.apply_chat_template(ex["messages"], tokenize=True, add_generation_prompt=False)
        return len(ids) <= args.max_len
    ds = ds.filter(fits, num_proc=8)
    # prompt/completion form so TRL masks the prompt (Qwen's template has no {% generation %} markers)
    ds = ds.map(lambda ex: {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]}, remove_columns=["messages"])
    print({k: len(v) for k, v in ds.items()}, flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    lora = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.bs,
                    gradient_accumulation_steps=args.grad_acc, learning_rate=args.lr, lr_scheduler_type="cosine",
                    warmup_steps=10, logging_steps=10, save_strategy="epoch", eval_strategy="steps", eval_steps=100,
                    bf16=True, gradient_checkpointing=True, max_length=args.max_len, packing=False,
                    completion_only_loss=True, report_to=[], dataloader_num_workers=2)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["valid"],
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    Path(args.out, "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(f"{args.out}/merged", safe_serialization=True)
        tok.save_pretrained(f"{args.out}/merged")
        print("merged ->", f"{args.out}/merged")


if __name__ == "__main__":
    main()
