"""Paired bootstrap over visits for two live-view eval directories.

  python -m autoresearch.live_ci runs/live_synth/eval_v5off runs/live_synth/eval_v5
"""
import glob, json, random, sys
from pathlib import Path


def load(d):
    out = {}
    for f in glob.glob(f"{d}/*.json"):
        r = json.loads(Path(f).read_text()); s = r.get("score") or {}
        seen, rf = set(), []
        for idx, vs in (s.get("red_flags") or {}).items():
            try:
                feats = [f.get("feature") if isinstance(f, dict) else str(f) for f in (r["snapshots"][int(idx)]["synth"].get("red_flags") or [])]
            except (ValueError, IndexError):
                feats = []
            for f, v in zip(feats, vs if isinstance(vs, list) else [vs]):
                key = (f or "").strip().lower()
                if key and key not in seen:
                    seen.add(key); rf.append(bool(v))
        ps = s.get("plan_suggested_final") or []
        out[r["id"]] = {"dx_top3": 1.0 if s.get("dx_first_top3_t") is not None else 0.0,
                        "contradict": float(ps.count("contradicts")), "agree": float(ps.count("agrees")),
                        "rf_raised": float(len(rf)), "rf_false": float(len(rf) - sum(rf)),
                        "premature": 1.0 if r.get("premature_complaint") else 0.0,
                        "dx_t": s.get("dx_first_top3_t"), "latency": (r.get("latency") or {}).get("mean") if isinstance(r.get("latency"), dict) else None}
    return out


def boot(a, b, k, n=10000, seed=0):
    ids = sorted(set(a) & set(b)); rng = random.Random(seed)
    diffs = [a[i][k] - b[i][k] for i in ids]; m = len(diffs)
    bs = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(diffs) / m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1], m


a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"| {Path(sys.argv[1]).name} − {Path(sys.argv[2]).name} | " + " | ".join(["dx in top-3 (0/1)", "contradicting suggestions", "red flags raised", "false-alarm flags", "agreeing suggestions", "premature complaint"]) + " |")
print("|---" * 7 + "|")
cells = []
for k in ("dx_top3", "contradict", "rf_raised", "rf_false", "agree", "premature"):
    d, lo, hi, m = boot(a, b, k); star = "**" if lo > 0 or hi < 0 else ""
    cells.append(f"{star}{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
print("| paired n = %d | " % m + " | ".join(cells) + " |")
for name, x in ((sys.argv[1], a), (sys.argv[2], b)):
    ts = sorted(v["dx_t"] for v in x.values() if v["dx_t"] is not None)
    print(f"{Path(name).name}: top3 ever {len(ts)}/{len(x)}, median first top-3 t {ts[len(ts)//2] if ts else None} s, flags {sum(v['rf_raised'] for v in x.values()):.0f}, contradict {sum(v['contradict'] for v in x.values()):.0f}, agree {sum(v['agree'] for v in x.values()):.0f}, premature {sum(v['premature'] for v in x.values()):.0f}")
