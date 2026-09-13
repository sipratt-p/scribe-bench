"""Item-level re-judging of a live-view eval directory with any judge model, one verdict per item, short prompts.

  python -m autoresearch.live_itemjudge --src runs/live_synth/eval_v5 --judge gemma --out runs/live_synth/itemjudge_v5_gemma.json

Three item types, each judged with the minimum context so that schema drift and context limits cannot bias the result:
  flag   : each unique red flag (feature, tier, time) -> was the feature actually stated in the transcript up to that time,
           and is the tier appropriate?  yes/no
  plan   : each plan suggestion in the LAST snapshot -> agrees / extra / contradicts, against the stored reference plan
  dx     : for each snapshot's differential list -> does it contain the stored reference diagnosis? yes/no
           (earliest yes gives dx_first_top3 for this judge)
The reference diagnosis and plan are taken from the eval file (extracted once), so judges are compared on the verdict step only.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from demo.livesynth import DATA, _llm  # noqa: E402

FLAG_SYS = """You check one red flag raised by an in-visit assistant during a GP consultation. You get the transcript up to the moment the flag was raised, the flagged feature, and its tier ("act now" = ambulance / A&E / same-hour assessment; "check today" = the clinician should ask or examine for it before the visit ends). Answer with one word: YES if the feature was actually stated in the transcript by that time AND the tier is appropriate for it; NO otherwise (a merely possible or unclarified feature is NO)."""
PLAN_SYS = """You compare one plan suggestion made by an in-visit assistant with the clinician's actual plan for the same visit. Answer with one word: AGREES if the clinician's plan contains the same action; EXTRA if the suggestion is reasonable but not in the plan; CONTRADICTS if the plan or the transcript goes against it."""
DX_SYS = """You are given a reference diagnosis from the clinician's note and a list of candidate diagnoses from an assistant. Answer with one word: YES if the list contains the same condition as the reference (wording may differ; a broader or narrower label for the same condition counts), NO otherwise."""


def word(txt: str, options: tuple[str, ...]) -> str | None:
    t = (txt or "").strip().upper()
    for o in options:
        if re.match(r"^\W*" + o, t):
            return o
    for o in options:
        if o in t:
            return o
    return None


def transcript_upto(rec: dict, t: float) -> str:
    return "\n".join(f"{u['speaker']}: {u['text']}" for u in sorted(rec["reference"]["utterances"], key=lambda u: u["start"]) if u["start"] <= t)


def judge_visit(path: str, judge: str) -> dict:
    r = json.loads(Path(path).read_text())
    rec = json.loads((DATA / f"{r['id']}.json").read_text())
    tg = r.get("targets") or (r.get("score") or {}).get("targets") or {}
    ref_dx, ref_plan = tg.get("diagnosis") or "", tg.get("plan") or []
    out = {"id": r["id"], "flags": [], "plan": [], "dx_first_top3_t": None, "dx_yes_snapshots": 0}
    seen = set()
    for s in r["snapshots"]:
        v = s["synth"]
        for f in v.get("red_flags") or []:
            if not isinstance(f, dict):
                continue
            key = (f.get("feature") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ans = _llm(FLAG_SYS, f"TRANSCRIPT UP TO {s['t']:.0f} s:\n{transcript_upto(rec, s['t'])}\n\nFLAG: {f.get('feature')}\nTIER: {f.get('tier') or 'check today'}\nWHY: {f.get('why') or ''}", 8, model=judge)
            out["flags"].append({"t": s["t"], "feature": f.get("feature"), "tier": f.get("tier"), "genuine": word(ans, ("YES", "NO")) == "YES", "raw": (ans or "")[:40]})
    if ref_dx:
        for i, s in enumerate(r["snapshots"]):
            dxs = [d.get("dx") for d in (s["synth"].get("differential") or []) if isinstance(d, dict) and d.get("dx")]
            if not dxs:
                continue
            ans = _llm(DX_SYS, f"REFERENCE DIAGNOSIS: {ref_dx}\n\nCANDIDATES: {json.dumps(dxs)}", 8, model=judge)
            if word(ans, ("YES", "NO")) == "YES":
                out["dx_yes_snapshots"] += 1
                if out["dx_first_top3_t"] is None:
                    out["dx_first_top3_t"] = s["t"]
    last = r["snapshots"][-1]["synth"] if r["snapshots"] else {}
    listing = "\n".join(f"- {p}" for p in ref_plan) or "(no plan items)"
    for p in last.get("plan_suggested") or []:
        item = p.get("item") if isinstance(p, dict) else str(p)
        ans = _llm(PLAN_SYS, f"CLINICIAN'S PLAN:\n{listing}\n\nTRANSCRIPT (end of visit):\n{transcript_upto(rec, 1e9)[-6000:]}\n\nSUGGESTION: {item}", 8, model=judge)
        out["plan"].append({"item": item, "verdict": (word(ans, ("AGREES", "EXTRA", "CONTRADICTS")) or "unparsed").lower(), "raw": (ans or "")[:40]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.src}/*.json"))
    res = []
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda f: judge_visit(f, a.judge), files):
            res.append(r)
            print(r["id"], "flags", [(f["tier"], f["genuine"]) for f in r["flags"]], "dx@", r["dx_first_top3_t"], "plan", [p["verdict"][:5] for p in r["plan"]], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    nf = sum(len(r["flags"]) for r in res)
    print(f"SUMMARY judge={a.judge} visits={len(res)} flags={nf} genuine={sum(f['genuine'] for r in res for f in r['flags'])} "
          f"dx_top3={sum(1 for r in res if r['dx_first_top3_t'] is not None)} plan={dict((k, sum(1 for r in res for p in r['plan'] if p['verdict'] == k)) for k in ('agrees', 'extra', 'contradicts', 'unparsed'))}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
