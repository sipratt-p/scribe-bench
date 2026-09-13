"""Paired-bootstrap tables for a per-visit results file (Phase D reconciliation).

  python -m autoresearch.paper_ci runs/paper_pervisit_official.json --split test --base theirs

Prints mean per config, then for every other config the mean paired difference vs --base with a 95% bootstrap
interval (10,000 resamples over visits, seed 0). Plan recall is pooled over plan items (found/items) and
bootstrapped over visits. A star marks an interval that excludes zero.
"""
from __future__ import annotations

import argparse
import json
import random

METRICS = ["composite", "composite_v", "term_recall", "term_precision", "rougeL", "plan_recall", "misattrib", "grounded", "cited_frac", "linked"]


def cv(m: dict) -> float:
    return m.get("composite", 0) + 0.2 * m.get("grounded", 0)


def val(m: dict, k: str):
    if k == "composite_v":
        return cv(m)
    if k == "plan_recall":
        return 100 * m.get("plan_found", 0) / max(m.get("plan_items", 0), 1) if "plan_items" in m else None
    return m.get(k)


def plan_pool(rows: list[dict]) -> float:
    pi = sum(r.get("plan_items", 0) for r in rows); pf = sum(r.get("plan_found", 0) for r in rows)
    return 100 * pf / max(pi, 1)


def boot(a_rows, b_rows, k, n=10000, seed=0):
    rng = random.Random(seed); m = len(a_rows); out = []
    if k == "plan_recall":
        for _ in range(n):
            idx = [rng.randrange(m) for _ in range(m)]
            out.append(plan_pool([a_rows[i] for i in idx]) - plan_pool([b_rows[i] for i in idx]))
        d = plan_pool(a_rows) - plan_pool(b_rows)
    else:
        diffs = [val(a, k) - val(b, k) for a, b in zip(a_rows, b_rows)]
        for _ in range(n):
            out.append(sum(diffs[rng.randrange(m)] for _ in range(m)) / m)
        d = sum(diffs) / m
    out.sort()
    return d, out[int(0.025 * n)], out[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--split", default="test"); ap.add_argument("--base", default="theirs")
    ap.add_argument("--metrics", default=",".join(METRICS)); ap.add_argument("--md", action="store_true", help="markdown tables")
    a = ap.parse_args()
    d = json.load(open(a.path)); d = d.get(a.split, d)
    mets = [m for m in a.metrics.split(",") if m]
    cfgs = list(d)
    ok = {c: {i: r for i, r in d[c].items() if "err" not in r} for c in cfgs}
    ids = sorted(set.intersection(*(set(ok[c]) for c in cfgs)))
    print(f"split={a.split} configs={cfgs} paired n={len(ids)}", {c: len(ok[c]) for c in cfgs})
    mets = [m for m in mets if all(val(ok[c][ids[0]], m) is not None for c in cfgs)]
    hdr = "| config | " + " | ".join(mets) + " |"
    print(hdr); print("|---" * (len(mets) + 1) + "|")
    for c in cfgs:
        rows = [ok[c][i] for i in ids]
        cells = [f"{plan_pool(rows):.1f}" if m == "plan_recall" else f"{sum(val(r, m) for r in rows) / len(rows):.2f}" for m in mets]
        print(f"| {c} | " + " | ".join(cells) + " |")
    print()
    print(f"| config − {a.base} | " + " | ".join(mets) + " |"); print("|---" * (len(mets) + 1) + "|")
    for c in cfgs:
        if c == a.base:
            continue
        cells = []
        for m in mets:
            dd, lo, hi = boot([ok[c][i] for i in ids], [ok[a.base][i] for i in ids], m)
            star = "*" if lo > 0 or hi < 0 else ""
            cells.append(f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}")
        print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
