"""Summarise item-level judgings of one live-view run and their pairwise agreement.

  python -m autoresearch.itemjudge_agree runs/live_synth/eval_v5 runs/live_synth/itemjudge_v5_dsflash.json runs/live_synth/itemjudge_v5_gemma.json [...]

For each judging file: visits with the reference diagnosis ever in top-3, median first time, flags genuine / raised,
plan agree / extra / contradict. The eval directory supplies the timeline-prompt verdicts as a first column.
Then Cohen's kappa for top-3 (per visit) and for plan verdicts (per suggestion, three-way and contradict-vs-not) between every pair.
"""
import glob, json, sys
from itertools import combinations
from pathlib import Path


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def timeline(d):
    out = {}
    for f in glob.glob(f"{d}/*.json"):
        r = json.loads(Path(f).read_text()); s = r["score"]
        seen, flags = set(), []
        for idx, vs in (s.get("red_flags") or {}).items():
            feats = [x.get("feature") if isinstance(x, dict) else str(x) for x in (r["snapshots"][int(idx)]["synth"].get("red_flags") or [])]
            for ft, v in zip(feats, vs if isinstance(vs, list) else [vs]):
                k = (ft or "").strip().lower()
                if k and k not in seen:
                    seen.add(k); flags.append(bool(v))
        last = r["snapshots"][-1]["synth"] if r["snapshots"] else {}
        n_last = len(last.get("plan_suggested") or [])  # the timeline judge is asked for one verdict per item of the last snapshot; clip any surplus
        out[r["id"]] = {"top3": s.get("dx_first_top3_t") is not None, "t": s.get("dx_first_top3_t"), "flags": flags,
                        "plan": [{"agrees": "agrees", "extra": "extra", "contradicts": "contradicts"}.get(v, "unparsed") for v in (s.get("plan_suggested_final") or [])][:n_last]}
    return out


def itemfile(p):
    out = {}
    for r in json.loads(Path(p).read_text()):
        out[r["id"]] = {"top3": r["dx_first_top3_t"] is not None, "t": r["dx_first_top3_t"], "flags": [f["genuine"] for f in r["flags"]],
                        "plan": [x["verdict"] for x in r["plan"]]}
    return out


cols = {"timeline": timeline(sys.argv[1])}
for p in sys.argv[2:]:
    cols[Path(p).stem.split("_")[-1]] = itemfile(p)
ids = sorted(set.intersection(*(set(c) for c in cols.values())))
print("| Quantity | " + " | ".join(cols) + " |"); print("|---" * (len(cols) + 1) + "|")
print("| Visits with the reference diagnosis ever in the live top-3 | " + " | ".join(f"{sum(c[i]['top3'] for i in ids)} / {len(ids)}" for c in cols.values()) + " |")
def med(c):
    ts = sorted(c[i]["t"] for i in ids if c[i]["t"] is not None); return f"{ts[len(ts)//2]:.0f} s" if ts else "–"
print("| Median first time in top-3 | " + " | ".join(med(c) for c in cols.values()) + " |")
print("| Flags judged genuine / raised | " + " | ".join(f"{sum(sum(c[i]['flags']) for i in ids)} / {sum(len(c[i]['flags']) for i in ids)}" for c in cols.values()) + " |")
print("| Plan suggestions agree / extra / contradict | " + " | ".join("/".join(str(sum(c[i]["plan"].count(k) for i in ids)) for k in ("agrees", "extra", "contradicts")) for c in cols.values()) + " |")
print()
for a, b in combinations(cols, 2):
    ka = kappa([cols[a][i]["top3"] for i in ids], [cols[b][i]["top3"] for i in ids])
    pa = [v for i in ids for v in cols[a][i]["plan"]]; pb = [v for i in ids for v in cols[b][i]["plan"]]
    if len(pa) != len(pb):
        m = min(len(pa), len(pb)); pa, pb = pa[:m], pb[:m]
    kp = kappa(pa, pb); kc = kappa([v == "contradicts" for v in pa], [v == "contradicts" for v in pb])
    fa = [v for i in ids for v in cols[a][i]["flags"]]; fb = [v for i in ids for v in cols[b][i]["flags"]]
    kf = kappa(fa, fb) if len(fa) == len(fb) and fa else float("nan")
    print(f"κ {a} vs {b}: top-3 {ka:+.2f}; plan three-way {kp:+.2f}; contradict-vs-not {kc:+.2f}; flag genuine {kf:+.2f} (n plan {len(pa)}, flags {len(fa)})")
