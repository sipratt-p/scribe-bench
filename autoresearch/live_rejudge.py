"""Re-score an existing live-view eval directory with a different judge (no re-synthesis).
  python -m autoresearch.live_rejudge --src runs/live_synth/eval_v5 --judge gemma --out runs/live_synth/eval_v5_judge_gemma.json"""
import argparse, json, sys, glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo.livesynth import DATA, reference_targets, score_timeline, churn
ap=argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--judge", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2); a=ap.parse_args()
files=sorted(glob.glob(f"{a.src}/*.json"))
def one(f):
    r=json.load(open(f)); rec=json.loads((DATA/f"{r['id']}.json").read_text())
    try:
        t=reference_targets(rec, a.judge); sc=score_timeline(rec, r["snapshots"], t, a.judge); sc["churn"]=churn(r["snapshots"]); sc["targets"]=t
        return {"id": r["id"], "score": sc, "n_snapshots": len(r["snapshots"])}
    except Exception as e:
        return {"id": r["id"], "error": str(e)[:200]}
out=[]
with ThreadPoolExecutor(a.workers) as ex:
    for res in ex.map(one, files):
        out.append(res); print(res["id"], "err" if "error" in res else f"top3@{res['score'].get('dx_first_top3_t')} rf={sum(len(v) if isinstance(v,list) else 1 for v in (res['score'].get('red_flags') or {}).values())} plan={res['score'].get('plan_suggested_final')}", flush=True)
json.dump(out, open(a.out,"w"), indent=1); print("DONE", flush=True)
