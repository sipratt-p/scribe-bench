"""ACI-Bench loader. Returns encounters keyed by encounter id with every
transcript variant available for that encounter:

  humantrans : challenge_data dialogue (human transcript for aci/virtassist; for
               virtscribe the challenge dialogue is the human transcription too)
  asr        : raw ASR transcript (src_experiment_data *_asr.csv)
  asrcorr    : ASR corrected by a human (aci subset only)

Notes are identical across variants."""
from __future__ import annotations

import csv
from pathlib import Path

SPLITS = {
    "train": ("train.csv", "train"),
    "valid": ("valid.csv", "valid"),
    "test1": ("clinicalnlp_taskB_test1.csv", "test1"),
    "test2": ("clinicalnlp_taskC_test2.csv", "test2"),
    "test3": ("clef_taskC_test3.csv", "test3"),
}


def _rows(f: Path) -> list[dict]:
    return list(csv.DictReader(open(f, encoding="utf-8")))


def load_split(root: Path, split: str) -> dict[str, dict]:
    ch_file, prefix = SPLITS[split]
    data = root / "data"
    meta = {r["id"]: r["encounter_id"] for r in _rows(data / "challenge_data" / ch_file.replace(".csv", "_metadata.csv"))}
    enc: dict[str, dict] = {}
    for r in _rows(data / "challenge_data" / ch_file):
        enc[r["encounter_id"]] = {"encounter_id": r["encounter_id"], "dataset": r["dataset"], "note": r["note"],
                                  "variants": {"humantrans": r["dialogue"]}}
    for f in (data / "src_experiment_data").glob(f"{prefix}_*.csv"):
        if f.name.endswith("_metadata.csv"):
            continue
        variant = f.stem.split("_", 2)[2]  # e.g. aci_asr -> asr ; virtscribe_humantrans -> humantrans
        variant = variant.split("_", 1)[1] if "_" in variant else variant
        for r in _rows(f):
            eid = meta.get(r["id"])
            if eid in enc:
                enc[eid]["variants"][variant] = r["dialogue"]
    return enc


if __name__ == "__main__":
    import sys
    from collections import Counter
    root = Path(sys.argv[1])
    for s in SPLITS:
        e = load_split(root, s)
        c = Counter(v for x in e.values() for v in x["variants"])
        print(s, len(e), dict(c))
