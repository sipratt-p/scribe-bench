"""Build SFT data for the note model: transcript -> note pairs as chat jsonl.

  python -m scribe_bench.sft_prepare <data_root> <out_dir> [--medsynth_n 4000]

Sources (train only, no ACI-Bench test splits, no PriMock57):
  ACI-Bench train+valid humantrans/asrcorr/asr variants (87 encounters, up to ~200 pairs)
  MTS-Dialog training set (section-level snippets -> section text, 1,201)
  MedSynth (synthetic full dialogue -> note, sampled)

Output: train.jsonl / valid.jsonl in {"messages":[system,user,assistant]} form, which
mlx_lm.lora and TRL both accept."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from scribe_bench.acibench import load_split
from scribe_bench.notegen import ACI_SYSTEM

MTS_SYSTEM = ("You are a clinical documentation assistant. Given a short excerpt of a doctor-patient conversation, "
              "write the {section} section of the visit note in concise clinical prose. Include only stated information.")
MEDSYNTH_SYSTEM = ("You are a clinical documentation assistant. Given a doctor-patient conversation transcript, write the visit "
                   "note in SOAP format (Subjective, Objective, Assessment, Plan). Include only information stated in the transcript.")

SECTION_NAMES = {"GENHX": "HISTORY OF PRESENT ILLNESS", "CC": "CHIEF COMPLAINT", "PASTMEDICALHX": "PAST MEDICAL HISTORY",
                 "MEDICATIONS": "MEDICATIONS", "ALLERGY": "ALLERGIES", "FAM/SOCHX": "FAMILY AND SOCIAL HISTORY",
                 "ASSESSMENT": "ASSESSMENT", "PLAN": "PLAN", "EXAM": "PHYSICAL EXAM", "ROS": "REVIEW OF SYSTEMS",
                 "PASTSURGICAL": "PAST SURGICAL HISTORY", "DIAGNOSIS": "DIAGNOSIS", "DISPOSITION": "DISPOSITION",
                 "EDCOURSE": "ED COURSE", "IMMUNIZATIONS": "IMMUNIZATIONS", "LABS": "LABS", "IMAGING": "IMAGING",
                 "PROCEDURES": "PROCEDURES", "OTHER_HISTORY": "OTHER HISTORY", "GYNHX": "GYNECOLOGIC HISTORY"}


def chat(system: str, transcript: str, note: str) -> dict:
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "TRANSCRIPT:\n" + transcript + "\n\nWrite the note now."},
                         {"role": "assistant", "content": note.strip()}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out_dir")
    ap.add_argument("--medsynth_n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    root, out = Path(args.root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, srcs = [], {}
    for split in ("train", "valid"):
        for e in load_split(root / "aci-bench", split).values():
            for v, dlg in e["variants"].items():
                rows.append(("aci", chat(ACI_SYSTEM, dlg, e["note"])))
    for r in csv.DictReader(open(root / "MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv", encoding="utf-8")):
        sec = SECTION_NAMES.get(r["section_header"], r["section_header"])
        rows.append(("mts", chat(MTS_SYSTEM.format(section=sec), r["dialogue"], r["section_text"])))
    from datasets import load_from_disk
    ds = load_from_disk(str(root / "medsynth"))["train"]
    idx = random.sample(range(len(ds)), min(args.medsynth_n, len(ds)))
    for i in idx:
        r = ds[i]
        rows.append(("medsynth", chat(MEDSYNTH_SYSTEM, r["Dialogue"], r[" Note"])))

    random.shuffle(rows)
    n_valid = max(50, len(rows) // 50)
    valid, train = rows[:n_valid], rows[n_valid:]
    for name, rs in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w") as f:
            for src, r in rs:
                srcs[src] = srcs.get(src, 0) + 1
                f.write(json.dumps(r) + "\n")
    print("train", len(train), "valid", len(valid), "sources", srcs)


if __name__ == "__main__":
    main()
