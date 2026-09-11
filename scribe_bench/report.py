"""Aggregate run outputs into runs/RESULTS.md.

  python -m scribe_bench.report runs/asr_score.json runs/note_score_test*.json runs/verifier/*_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def table(rows: dict[str, dict], cols: list[str], title: str) -> str:
    out = [f"### {title}", "", "| system | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, v in rows.items():
        out.append(f"| {k} | " + " | ".join(str(v.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    parts = ["# scribe-bench results\n"]
    for f in sys.argv[1:]:
        p = Path(f)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if "asr" in p.name:
            parts.append(table(d, ["n", "wer", "med_term_err", "der", "sub", "del", "ins", "wall_s"], f"ASR — {p.stem}"))
        elif "note" in p.name:
            parts.append(table(d, ["n", "rougeL", "rouge1", "rouge2", "bertscore_f1", "term_recall", "term_precision",
                                   "cited_frac", "hyp_words", "ref_words"], f"Notes — {p.stem}"))
        elif "verifier" in str(p):
            parts.append(f"### Verifier — {p.stem}\n\n```json\n{json.dumps(d, indent=1)}\n```\n")
    Path("runs/RESULTS.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
