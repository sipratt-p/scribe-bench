"""Recover plan-item recall under required citations.

  python -m autoresearch.plan_recall [--test_top 1]

Evaluates cited variants of the loop's best config on dev with the verifiability scorer, ranks by
plan recall subject to score_v >= the cited baseline - 0.5, then runs the top variant(s) on test.
Writes runs/plan_recall.json / .md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoresearch import loop as L  # noqa: E402
from autoresearch.verif_eval import BEST_EXTRA, CONFIGS, run  # noqa: E402

PLAN_A = ("Prioritize high-precision terminology. Do not hallucinate medications or instructions. "
          "Capture every plan item the clinician actually stated: prescriptions, investigations, referrals, "
          "follow-up timing, and safety-netting advice. Write each as its own sentence with its citation. "
          "Explicitly state 'no plan' only if none were mentioned.")
PLAN_B = (BEST_EXTRA + " Before finishing, re-read the last third of the transcript, where the plan is usually agreed, "
          "and make sure every action, follow-up and safety-netting instruction is in the note with its citation.")
PLAN_C = ("Prioritize high-precision terminology. Do not hallucinate medications or instructions. "
          "End the note with a 'Plan' section: one bullet per action the clinician stated (treatment, investigation, "
          "referral, follow-up timing, safety-netting), each bullet cited. If no plan was discussed, write 'Plan: none stated'.")
PLAN_D = (BEST_EXTRA + " A plan item that you can cite must always be included; omit only what you cannot cite.")

VARIANTS = {"cite_A_enumerate": PLAN_A, "cite_B_reread": PLAN_B, "cite_C_plan_section": PLAN_C, "cite_D_must_include": PLAN_D}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_top", type=int, default=1)
    args = ap.parse_args()
    recs, dev, test = L.load_recs()
    note_llm = L.LLM("http://localhost:8004/v1", "qwen3.8-27b", workers=8)
    judge = note_llm
    prior = json.loads((ROOT / "runs/verif_eval.json").read_text())
    out = {"best_cite/dev": prior["best_cite/dev"], "best_cite/test": prior["best_cite/test"], "best/test": prior["best/test"]}
    cfgs = {}
    for name, extra in VARIANTS.items():
        cfgs[name] = {**CONFIGS["best_cite"], "extra": extra}
        m = run(name, cfgs[name], dev, recs, note_llm, judge)
        out[f"{name}/dev"] = m
        print(name, "dev", json.dumps(m), flush=True)
    floor = prior["best_cite/dev"]["score_v"] - 0.5
    ranked = sorted(VARIANTS, key=lambda n: -out[f"{n}/dev"]["plan_recall"])
    ok = [n for n in ranked if out[f"{n}/dev"]["score_v"] >= floor] or ranked[:1]
    for name in ok[: args.test_top]:
        m = run(name, cfgs[name], test, recs, note_llm, judge)
        out[f"{name}/test"] = m
        print(name, "test", json.dumps(m), flush=True)
    (ROOT / "runs/plan_recall.json").write_text(json.dumps({"results": out, "variants": VARIANTS}, indent=1))
    cols = ["score", "score_v", "plan_recall", "grounded", "linked", "term_recall", "term_precision", "rougeL", "misattrib", "n"]
    md = ["| config | split | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k, m in out.items():
        md.append(f"| {k.split('/')[0]} | {k.split('/')[1]} | " + " | ".join(str(m[c]) for c in cols) + " |")
    (ROOT / "runs/plan_recall.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
