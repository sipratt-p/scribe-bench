"""Build a primary-care medical lexicon from open dialogue/note corpora.

A token counts as a medical term if it appears in MedSynth or MTS-Dialog notes
at least MIN_COUNT times and is rare in general English (wordfreq zipf < ZIPF_MAX).
Used for (a) hotword biasing of pass-2 ASR and (b) medical-term error rate.
No PriMock57 text is used, so there is no leakage into the eval."""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

from wordfreq import zipf_frequency

MIN_COUNT = 3
ZIPF_MAX = 3.6
TOK = re.compile(r"[a-z][a-z\-]{3,}")


def iter_note_text(root: Path):
    mts = root / "MTS-Dialog" / "Main-Dataset"
    for f in mts.glob("*.csv"):
        for r in csv.DictReader(open(f)):
            yield r.get("section_text") or ""
            yield r.get("dialogue") or ""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(root / "medsynth"))["train"]
        for r in ds:
            yield r.get(" Note") or ""
            yield r.get("Dialogue") or ""
    except Exception as e:  # noqa: BLE001
        print("medsynth skipped:", e, file=sys.stderr)


def build(root: Path) -> list[str]:
    c = Counter()
    for t in iter_note_text(root):
        c.update(TOK.findall(t.lower()))
    terms = [w for w, n in c.items() if n >= MIN_COUNT and zipf_frequency(w, "en") < ZIPF_MAX]
    return sorted(terms, key=lambda w: -c[w])


if __name__ == "__main__":
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    terms = build(root)
    out.write_text("\n".join(terms) + "\n")
    print(len(terms), "terms ->", out, "e.g.", terms[:25])
