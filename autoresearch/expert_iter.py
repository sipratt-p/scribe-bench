"""Expert iteration (rejection-sampling fine-tuning) on the note writer, as an RL exercise.

Policy   : Qwen3.8-27B (bf16, GPU 1) + LoRA, prompted exactly like the loop's best cited config
           (MOSS-TD hotcc transcript, LLM role map, precision-first extra, citations required).
Reward   : loop composite (0.3 TR + 0.3 TP + 0.2 RL + 0.2 plan - 40 misattrib)
           + 0.2 * grounded + 0.1 * linked - length penalty (20 per unit outside round-0 mean +/- 0.6).
Loop     : sample n notes per dev transcript -> reward -> keep top k -> LoRA on the union of all kept
           samples so far -> greedy eval on dev + test with judge 1 (Qwen 27B) and judge 2 (DeepSeek V4 Flash)
           -> repeat. Weights are updated only from selected samples; nothing else in the project does that.
Readouts : composite, grounded, linked, misattrib, plan recall, edits/100w, len ratio, passive-voice per 100w,
           agentless sentence share, judge-1 vs judge-2 gap on test.

  python -m autoresearch.expert_iter run --rounds 4 --n 8 --k 2
  python -m autoresearch.expert_iter gen --adapter <dir|none> --ids dev --n 8 --temp 0.8 --out <json>
  python -m autoresearch.expert_iter train --data <jsonl...> --out <adapter dir>
"""
from __future__ import annotations
import os

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoresearch import loop as L  # noqa: E402
from autoresearch.loop import CITE_SUFFIX, PROMPTS, SENT_SPLIT, cached, h, number_lines, strip_cites  # noqa: E402
from autoresearch.verif_eval import BEST_EXTRA, CONFIGS, verifiability  # noqa: E402

MODEL = str(Path.home() / "models/Qwen3.8-27B")
OUT = ROOT / "runs/expert_iter"
CFG = CONFIGS["best_cite"]
SYSTEM = PROMPTS["base"] + "\n" + BEST_EXTRA + CITE_SUFFIX
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PASSIVE = re.compile(r"\b(was|were|is|are|been)\s+(reported|noted|stated|mentioned|described|advised|discussed|documented)\b|\bit was\b", re.I)
AGENT = re.compile(r"\b(patient|she|he|they|doctor|clinician|gp|dr|mum|mother|father|son|daughter)\b", re.I)

PREDICTIONS = """## Predictions, written before the first sample (12 Sep 2026)
1. Round 1 gains are real: best-of-8 on the composite picks up 2-3 dev points; about half survives on test.
2. By round 3 the reward is gamed: term precision -> shorter notes; grounded -> transcript copying; plan recall -> plan section balloons;
   misattribution penalty -> attribution disappears (passive voice, agentless sentences), and the judge stops flagging.
3. Verifier recall on injected errors does not move (it is not trained); what changes is whether the writer emits catchable claims.
4. Judge 2 (DeepSeek) scores tuned notes lower than judge 1 (Qwen) more each round; the judge gap is the reward-hacking meter.
Guards: length band round-0 mean +/- 0.6; held-out test; judge 2; clinician readouts (misattrib, plan recall, edits/100w) per round.
"""


def user_for(dialogue: str) -> str:
    return "TRANSCRIPT:\n" + number_lines(dialogue) + "\n\nWrite the note now."


def messages(dialogue: str, note: str | None = None):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_for(dialogue)}]
    if note is not None:
        m.append({"role": "assistant", "content": note})
    return m


def transcripts(ids, recs):
    llm = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=8)
    return {cid: L.transcript_for(CFG, recs[cid], llm) for cid in ids}


def load_policy(adapter: str | None, device: int = 1):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": device}).eval()
    if adapter and adapter != "none":
        from peft import PeftConfig, get_peft_model
        from safetensors.torch import load_file
        pm = get_peft_model(model, PeftConfig.from_pretrained(adapter))
        sd = load_file(f"{adapter}/adapter_model.safetensors")
        sd = {k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): v for k, v in sd.items()}
        missing, unexpected = pm.load_state_dict(sd, strict=False)
        lora_missing = [k for k in missing if "lora_" in k]
        assert not lora_missing and not unexpected, f"adapter mismatch: {lora_missing[:3]} {unexpected[:3]}"
        model = pm.merge_and_unload().eval()
        print("adapter merged:", adapter, flush=True)
    return tok, model


def chat_prompt(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def cmd_gen(args):
    import torch
    recs, dev, test = L.load_recs()
    ids = {"dev": dev, "test": test, "all": dev + test}[args.ids]
    out = Path(args.out)
    done = json.loads(out.read_text()) if out.exists() else {}
    todo = [c for c in ids if c not in done]
    if not todo:
        print("gen: nothing to do", flush=True)
        return
    tr = transcripts(todo, recs)
    tok, model = load_policy(args.adapter, args.device)
    jobs = sorted(todo, key=lambda c: len(tr[c]))
    t0 = time.time()
    sample = args.temp > 0
    if sample:  # one transcript at a time, n return sequences
        for cid in jobs:
            enc = tok([chat_prompt(tok, messages(tr[cid]))], return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=args.temp, top_p=0.95, max_new_tokens=args.max_new,
                                   num_return_sequences=args.n, pad_token_id=tok.pad_token_id)
            done[cid] = [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
            out.write_text(json.dumps(done))
            print(cid, len(done[cid]), "samples", round(time.time() - t0), "s", flush=True)
    else:  # greedy, batched
        for b in range(0, len(jobs), args.batch):
            chunk = jobs[b: b + args.batch]
            enc = tok([chat_prompt(tok, messages(tr[c])) for c in chunk], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id)
            for c, x in zip(chunk, g):
                done[c] = [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()]
            out.write_text(json.dumps(done))
            print(b + len(chunk), "/", len(jobs), round(time.time() - t0), "s", flush=True)


def reward_parts(rec, note, dialogue, judge, r0):
    m = L.score_note(rec, note, dialogue, judge)
    v = verifiability(rec, note, dialogue, judge)
    ref, hyp = rec["note"]["note"], strip_cites(note)
    hw, rw = len(hyp.split()), max(len(ref.split()), 1)
    lr = hw / rw
    from rapidfuzz.distance import Levenshtein
    sents = [s for s in SENT_SPLIT.split(hyp) if len(s.split()) >= 3]
    parts = {**m, **v, "len_ratio": lr, "edits_per_100w": 100 * Levenshtein.distance(hyp, ref) / rw,
             "passive_per_100w": 100 * len(PASSIVE.findall(hyp)) / max(hw, 1),
             "agentless_frac": 100 * sum(1 for s in sents if not AGENT.search(s)) / max(len(sents), 1),
             "comp": L.composite(m)}
    pen = 20 * max(0.0, abs(lr - r0) - 0.6) if r0 is not None else 0.0
    parts["reward"] = parts["comp"] + 0.2 * v["grounded"] + 0.1 * v["linked"] - pen
    return parts


def score_samples(samples: dict, recs, tr, judge, r0):
    jobs = [(cid, i, n) for cid, ns in samples.items() for i, n in enumerate(ns)]

    def one(j):
        cid, i, n = j
        return cid, i, reward_parts(recs[cid], n, tr[cid], judge, r0)
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs))
    out = {}
    for cid, i, p in rows:
        out.setdefault(cid, {})[i] = p
    return out


def aggregate(scored: dict, judge2_rows: dict | None = None) -> dict:
    rows = [p[0] for p in scored.values()]
    keys = ["rougeL", "term_recall", "term_precision", "misattrib", "grounded", "linked", "cited_frac", "len_ratio",
            "edits_per_100w", "passive_per_100w", "agentless_frac", "reward"]
    agg = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    agg["plan_items"] = sum(r["plan_items"] for r in rows)
    agg["plan_found"] = sum(r["plan_found"] for r in rows)
    agg["plan_recall"] = 100 * agg["plan_found"] / max(agg["plan_items"], 1)
    agg["score"] = L.composite(agg)
    agg["score_v"] = agg["score"] + 0.2 * agg["grounded"]
    if judge2_rows:
        r2 = list(judge2_rows.values())
        a2 = {k: agg[k] for k in ("rougeL", "term_recall", "term_precision")}
        a2["misattrib"] = sum(r["misattrib"] for r in r2) / len(r2)
        a2["plan_items"] = sum(r["plan_items"] for r in r2)
        a2["plan_found"] = sum(r["plan_found"] for r in r2)
        agg["score_j2"] = L.composite(a2)
        agg["misattrib_j2"] = a2["misattrib"]
        agg["plan_recall_j2"] = 100 * a2["plan_found"] / max(a2["plan_items"], 1)
        agg["judge_gap"] = agg["score"] - agg["score_j2"]
    agg["n"] = len(rows)
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in agg.items()}


def cmd_train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    random.seed(0)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l)["messages"] for f in args.data for l in open(f)]

    def encode(msgs):
        try:
            p = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            f = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
            f = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        p = list(p["input_ids"]) if hasattr(p, "keys") else list(p)
        f = list(f["input_ids"]) if hasattr(f, "keys") else list(f)
        if len(f) > args.max_len or len(p) >= len(f) or f[: len(p)] != p:
            return None
        return torch.tensor(f), torch.tensor([-100] * len(p) + f[len(p):])
    train = [e for e in (encode(m) for m in rows) if e is not None]
    print(f"train {len(train)} of {len(rows)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": args.device})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=TARGETS))
    model.print_trainable_parameters()
    dev = next(model.parameters()).device
    n_micro = int(len(train) * args.epochs)
    steps = math.ceil(n_micro / args.grad_acc)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    model.train()
    micro, step, run, t0 = 0, 0, 0.0, time.time()
    order = list(range(len(train)))
    while micro < n_micro:
        random.shuffle(order)
        for i in order:
            if micro >= n_micro:
                break
            ids, lab = train[i]
            out = model(input_ids=ids[None].to(dev), labels=lab[None].to(dev))
            (out.loss / args.grad_acc).backward()
            run += float(out.loss)
            micro += 1
            if micro % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                print(json.dumps({"step": step, "of": steps, "loss": round(run / args.grad_acc, 4), "min": round((time.time() - t0) / 60, 1)}), flush=True)
                run = 0.0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print("saved", args.out, flush=True)


def sub(cmd: list[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-m", "autoresearch.expert_iter"] + cmd, check=True, cwd=str(ROOT))


def cmd_run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    nb = OUT / "notebook.md"
    if not nb.exists():
        nb.write_text("# Expert iteration on the note writer\n\n" + PREDICTIONS + "\n| round | split | reward | score | score_v | grounded | linked | TR | TP | RL | plan | mis | len | edits/100w | passive | agentless | score_j2 | mis_j2 | plan_j2 | judge_gap |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    recs, dev, test = L.load_recs()
    tr = transcripts(dev + test, recs)
    judge = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=8)
    judge2 = L.LLM(os.environ.get("SCRIBE_MAC_URL", "http://localhost:8600") + "/v1", "v4-flash", workers=4)
    state_p = OUT / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"r0": None, "rounds": {}}

    def evaluate(rnd: int, adapter: str):
        greedy = OUT / f"greedy_r{rnd}.json"
        sub(["gen", "--adapter", adapter, "--ids", "all", "--n", "1", "--temp", "0", "--out", str(greedy), "--device", str(args.device)])
        notes = json.loads(greedy.read_text())
        res = {}
        for split, ids in (("dev", dev), ("test", test)):
            scored = score_samples({c: notes[c] for c in ids}, recs, tr, judge, state["r0"])
            j2 = None
            if split == "test":
                with ThreadPoolExecutor(4) as ex:
                    j2 = dict(zip(ids, ex.map(lambda c: L.score_note(recs[c], notes[c][0], tr[c], judge2), ids)))
            res[split] = aggregate(scored, j2)
            (OUT / f"scored_r{rnd}_{split}.json").write_text(json.dumps(scored))
            a = res[split]
            row = [rnd, split, a["reward"], a["score"], a["score_v"], a["grounded"], a["linked"], a["term_recall"], a["term_precision"], a["rougeL"],
                   a["plan_recall"], a["misattrib"], a["len_ratio"], a["edits_per_100w"], a["passive_per_100w"], a["agentless_frac"],
                   a.get("score_j2", ""), a.get("misattrib_j2", ""), a.get("plan_recall_j2", ""), a.get("judge_gap", "")]
            with nb.open("a") as f:
                f.write("| " + " | ".join(str(x) for x in row) + " |\n")
            print("EVAL", rnd, split, json.dumps(a), flush=True)
        return res

    if "0" not in state["rounds"]:
        r = evaluate(0, "none")
        state["r0"] = r["dev"]["len_ratio"]
        state["rounds"]["0"] = r
        state_p.write_text(json.dumps(state, indent=1))
    for rnd in range(1, args.rounds + 1):
        if str(rnd) in state["rounds"]:
            continue
        prev = "none" if rnd == 1 else str(OUT / f"adapter_r{rnd - 1}")
        samples_p = OUT / f"samples_r{rnd}.json"
        sub(["gen", "--adapter", prev, "--ids", "dev", "--n", str(args.n), "--temp", str(args.temp), "--out", str(samples_p), "--device", str(args.device)])
        samples = json.loads(samples_p.read_text())
        scored = score_samples(samples, recs, tr, judge, state["r0"])
        (OUT / f"scored_samples_r{rnd}.json").write_text(json.dumps(scored))
        data_p = OUT / f"data_r{rnd}.jsonl"
        with data_p.open("w") as f:
            for cid, ss in scored.items():
                ranked = sorted(ss.items(), key=lambda kv: -kv[1]["reward"])
                for i, p in ranked[: args.k]:
                    f.write(json.dumps({"messages": messages(tr[cid], samples[cid][int(i)]), "reward": p["reward"]}) + "\n")
        rewards = [p["reward"] for ss in scored.values() for p in ss.values()]
        best = [max(p["reward"] for p in ss.values()) for ss in scored.values()]
        with nb.open("a") as f:
            f.write(f"\nround {rnd} samples: mean reward {sum(rewards) / len(rewards):.2f}, mean best-of-{args.n} {sum(best) / len(best):.2f}, "
                    f"kept top {args.k} per transcript ({sum(1 for _ in open(data_p))} examples; cumulative {sum(1 for r_ in range(1, rnd + 1) for _ in open(OUT / f'data_r{r_}.jsonl'))})\n\n")
        adapter = OUT / f"adapter_r{rnd}"
        sub(["train", "--out", str(adapter), "--device", str(args.device), "--epochs", str(args.epochs), "--lr", str(args.lr), "--r", str(args.lora_r),
             "--data"] + [str(OUT / f"data_r{r_}.jsonl") for r_ in range(1, rnd + 1)])
        state["rounds"][str(rnd)] = evaluate(rnd, str(adapter))
        state_p.write_text(json.dumps(state, indent=1))
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--adapter", default="none"); g.add_argument("--ids", default="dev"); g.add_argument("--n", type=int, default=8)
    g.add_argument("--temp", type=float, default=0.8); g.add_argument("--out", required=True); g.add_argument("--device", type=int, default=1)
    g.add_argument("--max_new", type=int, default=900); g.add_argument("--batch", type=int, default=4)
    t = sp.add_parser("train")
    t.add_argument("--data", nargs="+", required=True); t.add_argument("--out", required=True); t.add_argument("--device", type=int, default=1)
    t.add_argument("--max_len", type=int, default=4096); t.add_argument("--grad_acc", type=int, default=8); t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--r", type=int, default=16); t.add_argument("--epochs", type=float, default=2.0)
    r = sp.add_parser("run")
    r.add_argument("--rounds", type=int, default=4); r.add_argument("--n", type=int, default=8); r.add_argument("--k", type=int, default=2)
    r.add_argument("--temp", type=float, default=0.8); r.add_argument("--device", type=int, default=1)
    r.add_argument("--epochs", type=float, default=2.0); r.add_argument("--lr", type=float, default=1e-4); r.add_argument("--lora_r", type=int, default=16)
    args = ap.parse_args()
    {"gen": cmd_gen, "train": cmd_train, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
