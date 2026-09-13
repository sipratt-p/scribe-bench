"""Headless evaluation of the in-visit decision-support view over PriMock consultations (separate experiment).

  python -m autoresearch.live_eval [--ids all|dev|test] [--interval 20] [--model dsflash] [--workers 4]

For each consultation: replay the streaming transcript on the utterance timeline without waiting, take a synthesis
snapshot every `interval` seconds of visit time (same prompt and model as the /live page), then score the timeline
against the clinician's note. Writes runs/live_synth/eval/<cid>.json and runs/live_synth/eval_summary.{json,md}.

Metrics (per visit, then aggregated):
  dx_top3_t / dx_top1_t   visit time when the reference diagnosis first appeared in the live top-3 / top-1 (null = never)
  dx_lead_s               diagnosis_t (when the GP said it) - dx_top3_t; positive = the view had it before the GP said it
  q_hit_rate              share of suggested next-questions the GP later asked
  rf_raised / rf_genuine  red flags raised and how many the judge accepts as genuinely present and act-now
  plan_sug agree/extra/contradict   plan suggestions at the end vs the clinician's plan
  premature_complaint     complaint stated at snapshot 1 that differs from the final complaint
  top1_flips / vanished   stability
  flags_precision         mishearing flags whose 'heard' text is absent from the human transcript and whose suggested
                          term is present in it (ground truth, no LLM)
  latency mean / p95
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from demo.livesynth import DATA, WORD, _slug, churn, lookup, misheard, new_dx_to_lookup, reference_targets, score_timeline, synthesize_once, timeline  # noqa: E402

OUT = ROOT / "runs/live_synth/eval"
TAG = ""
LOOKUPS_ON = True


def run_visit(cid: str, interval: float, model: str) -> dict:
    out_dir = ROOT / f"runs/live_synth/eval{('_' + TAG) if TAG else ''}"
    p = out_dir / f"{cid}.json"
    if p.exists():
        return json.loads(p.read_text())
    rec = json.loads((DATA / f"{cid}.json").read_text())
    chunks = timeline(rec)
    duration = rec.get("duration_s") or (chunks[-1]["t"] + 3 if chunks else 0)
    transcript, snapshots, flags, seen, recent = [], [], [], set(), []
    prev = None
    refs, inflight, n_lookups = {}, set(), 0

    def synth(text, prev, t):
        nonlocal n_lookups
        js = synthesize_once(text, prev, t, model, refs if LOOKUPS_ON else None)
        if LOOKUPS_ON:
            for dx in new_dx_to_lookup(js, refs, inflight):
                try:
                    refs[_slug(dx)] = lookup(dx, model)
                    n_lookups += 1
                except Exception as ex:  # noqa: BLE001
                    refs[_slug(dx)] = {"condition": dx, "empty": True, "error": str(ex)}
        return js
    next_t = interval
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if c["t"] > next_t:
            t0 = time.time()
            prev = synth(" ".join(transcript), prev, next_t)
            snapshots.append({"t": next_t, "synth": prev, "latency_s": round(time.time() - t0, 1)})
            next_t += interval
            continue
        transcript.append(c["text"])
        ws = WORD.findall(c["text"])
        for f in misheard(ws, recent, seen):
            flags.append({**f, "t": c["t"]})
        recent = (recent + ws)[-6:]
        i += 1
    t0 = time.time()
    prev = synth(" ".join(transcript), prev, duration)
    snapshots.append({"t": duration, "synth": prev, "latency_s": round(time.time() - t0, 1)})
    targets = reference_targets(rec, model)
    sc = score_timeline(rec, snapshots, targets, model)
    sc["churn"] = churn(snapshots)
    # ground-truth mishearing precision: heard text absent from the human transcript, suggested term present
    human = " ".join(u["text"] for u in rec["reference"]["utterances"]).lower()
    fl_ok = sum(1 for f in flags if f["heard"].lower() not in human and f["maybe"].lower() in human)
    first_c = (snapshots[0]["synth"].get("complaint") or "").strip().lower() if snapshots else ""
    last_c = (snapshots[-1]["synth"].get("complaint") or "").strip().lower() if snapshots else ""
    res = {"id": cid, "duration_s": duration, "interval": interval, "model": model, "targets": targets, "score": sc,
           "lookups": n_lookups, "lookups_used": sum(1 for r in refs.values() if not r.get("empty")),
           "flags": flags, "flags_ok": fl_ok, "premature_complaint": bool(first_c and last_c and first_c != last_c),
           "first_complaint": first_c, "final_complaint": last_c,
           "latency": [s["latency_s"] for s in snapshots], "snapshots": snapshots}
    out_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return res


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    dx_present = [r for r in rows if r["targets"].get("diagnosis")]
    top3 = [r for r in dx_present if r["score"].get("dx_first_top3_t") is not None]
    top1 = [r for r in dx_present if r["score"].get("dx_first_top1_t") is not None]
    leads = [r["targets"]["diagnosis_t"] - r["score"]["dx_first_top3_t"] for r in top3 if r["targets"].get("diagnosis_t") is not None]
    def uniq_q(r):
        seen, out = set(), []
        for idx, v in (r["score"].get("questions_asked") or {}).items():
            try:
                qtxt = (r["snapshots"][int(idx)]["synth"].get("next_question") or "").strip().lower()
            except (ValueError, IndexError):
                qtxt = str(idx)
            if qtxt and qtxt not in seen:
                seen.add(qtxt)
                out.append(bool(v))
        return out

    def uniq_rf(r):
        seen, out = set(), []
        for idx, vs in (r["score"].get("red_flags") or {}).items():
            try:
                feats = [f.get("feature") if isinstance(f, dict) else str(f) for f in (r["snapshots"][int(idx)]["synth"].get("red_flags") or [])]
            except (ValueError, IndexError):
                feats = []
            vs = vs if isinstance(vs, list) else [vs]
            for f, v in zip(feats, vs):
                key = (f or "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(bool(v))
        return out
    q = [v for r in rows for v in uniq_q(r)]
    rf = [v for r in rows for v in uniq_rf(r)]
    ps = [v for r in rows for v in (r["score"].get("plan_suggested_final") or [])]
    rv = [v for r in rows for vs in (r["score"].get("revisions") or {}).values() for v in (vs if isinstance(vs, list) else [vs])]
    lat = [x for r in rows for x in r["latency"]]
    lat_s = sorted(lat)
    flags = sum(len(r["flags"]) for r in rows)
    return {
        "visits": n, "visits_with_reference_dx": len(dx_present),
        "dx_in_top3_ever": f"{len(top3)}/{len(dx_present)}", "dx_top1_ever": f"{len(top1)}/{len(dx_present)}",
        "dx_top3_median_t": sorted(r["score"]["dx_first_top3_t"] for r in top3)[len(top3) // 2] if top3 else None,
        "dx_before_gp_said_it": f"{sum(1 for x in leads if x > 0)}/{len(leads)}",
        "dx_lead_median_s": sorted(leads)[len(leads) // 2] if leads else None,
        "q_suggested_unique": len(q), "q_hit_rate": round(100 * sum(1 for v in q if v) / max(len(q), 1), 1),
        "rf_raised_unique": len(rf), "rf_genuine": sum(1 for v in rf if v), "rf_precision": round(100 * sum(1 for v in rf if v) / max(len(rf), 1), 1),
        "visits_with_red_flags": sum(1 for r in rows if uniq_rf(r)),
        "plan_sug_total": len(ps), "plan_sug_agree": ps.count("agrees"), "plan_sug_extra": ps.count("extra"), "plan_sug_contradict": ps.count("contradicts"),
        "revisions_total": len(rv), "revisions_toward": rv.count("toward"), "revisions_away": rv.count("away"),
        "premature_complaint_rate": round(100 * sum(1 for r in rows if r["premature_complaint"]) / max(n, 1), 1),
        "top1_flips_mean": round(sum(r["score"]["churn"]["top1_dx_changes"] for r in rows) / max(n, 1), 2),
        "vanished_items_mean": round(sum(r["score"]["churn"]["vanished_items"] for r in rows) / max(n, 1), 2),
        "flags_total": flags, "flags_precision": round(100 * sum(r["flags_ok"] for r in rows) / max(flags, 1), 1),
        "latency_mean_s": round(sum(lat) / max(len(lat), 1), 2), "latency_p95_s": lat_s[int(0.95 * (len(lat_s) - 1))] if lat_s else None,
        "snapshots_total": len(lat),
        "lookups_used_total": sum(r.get("lookups_used", 0) for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="all")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--model", default="dsflash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="", help="output under runs/live_synth/eval_<tag>/ and eval_summary_<tag>.md")
    ap.add_argument("--no-lookups", action="store_true")
    args = ap.parse_args()
    global TAG, LOOKUPS_ON
    TAG, LOOKUPS_ON = args.tag, not args.no_lookups
    ids = sorted(p.stem for p in DATA.glob("*.json") if "consultation" in p.stem)
    if args.ids == "dev":
        ids = ids[:20]
    elif args.ids == "test":
        ids = ids[20:]
    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(lambda c: run_visit(c, args.interval, args.model), ids):
            rows.append(r)
            sc = r["score"]
            print(r["id"], "dx:", (r["targets"].get("diagnosis") or "-")[:40], "top3@", sc.get("dx_first_top3_t"), "gp@", r["targets"].get("diagnosis_t"),
                  "q", sum(1 for v in (sc.get("questions_asked") or {}).values() if v), "/", len(sc.get("questions_asked") or {}),
                  "rf", len([v for vs in (sc.get("red_flags") or {}).values() for v in (vs if isinstance(vs, list) else [vs])]),
                  "plan", sc.get("plan_suggested_final"), "lat", round(sum(r["latency"]) / max(len(r["latency"]), 1), 1), flush=True)
    summ = summarize(rows)
    suf = ("_" + TAG) if TAG else ""
    (ROOT / f"runs/live_synth/eval_summary{suf}.json").write_text(json.dumps(summ, indent=1))
    md = "| metric | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in summ.items()) + "\n"
    (ROOT / f"runs/live_synth/eval_summary{suf}.md").write_text(md)
    print(md)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
