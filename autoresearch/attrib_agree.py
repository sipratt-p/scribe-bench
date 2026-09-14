"""Attribution agreement on the 74 held-out notes across judges: Qwen (judge 1), DeepSeek (judge 2), Gemma 4, Claude (four labelling agents), and any human label files.

  python -m autoresearch.attrib_agree [human_labels.json ...]

Prints flagged counts per judge and per config, pairwise Cohen's kappa on flagged/not-flagged, and notes flagged by k judges.
Claude labels: runs/paper_labels_claude.json (merged from the agents' label_out_*.json).
"""
import json, sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
notes = json.load(open(ROOT / "runs/paper_notes_74.json")); keys = [(n["cfg"], n["id"]) for n in notes]
t = json.load(open(ROOT / "runs/paper_pervisit_test.json"))
J = {"Qwen": {k: t[k[0]][k[1]]["misattrib"] for k in keys},
     "DeepSeek": {k: t[k[0]][k[1]].get("j2_misattrib", 0) for k in keys},
     "Gemma": {(x["cfg"], x["id"]): x["misattrib"] for x in json.load(open(ROOT / "runs/paper_judge3_gemma.json"))}}
cl = ROOT / "runs/paper_labels_claude.json"
if cl.exists():
    J["Claude"] = {(x["cfg"], x["id"]): x["misattrib"] for x in json.load(open(cl))}
for p in sys.argv[1:]:
    d = json.load(open(p)); J[Path(p).stem] = {(x["cfg"], x["id"]): x.get("count", x.get("misattrib", 0)) for x in (d if isinstance(d, list) else d.values())}


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in (True, False))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


common = [k for k in keys if all(k in j for j in J.values())]
print(f"notes with every judging: {len(common)}")
print("| judge | flagged | flagged vanilla | flagged best | mean / note vanilla | mean / note best |"); print("|---|---|---|---|---|---|")
for name, j in J.items():
    v = [j[k] for k in common if k[0] == "theirs"]; b = [j[k] for k in common if k[0] != "theirs"]
    print(f"| {name} | {sum(1 for k in common if j[k] > 0)} | {sum(1 for x in v if x > 0)} | {sum(1 for x in b if x > 0)} | {sum(v) / max(len(v), 1):.3f} | {sum(b) / max(len(b), 1):.3f} |")
print()
for a, b in combinations(J, 2):
    print(f"κ {a}–{b}: {kappa([J[a][k] > 0 for k in common], [J[b][k] > 0 for k in common]):+.2f}")
cnt = {}
for k in common:
    m = sum(1 for j in J.values() if j[k] > 0)
    cnt[m] = cnt.get(m, 0) + 1
print("notes flagged by k judges:", dict(sorted(cnt.items())))
for k in common:
    who = [n for n, j in J.items() if j[k] > 0]
    if len(who) >= 2:
        print("  ", k, who)
