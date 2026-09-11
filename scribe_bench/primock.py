"""PriMock57 loader: per-speaker TextGrids -> time-ordered utterances, reference
transcript, RTTM for diarization scoring, and clinician notes."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import textgrid

TAG_UNIN = re.compile(r"<UNIN/>|<INAUDIBLE_SPEECH/>")
TAG_UNSURE = re.compile(r"</?UNSURE>")


@dataclass
class Utt:
    speaker: str  # "Doctor" | "Patient"
    start: float
    end: float
    text: str


def clean_text(t: str) -> str:
    t = TAG_UNIN.sub(" ", t)
    t = TAG_UNSURE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def consultation_ids(root: Path) -> list[str]:
    return sorted(p.stem.replace("_doctor", "") for p in (root / "transcripts").glob("*_doctor.TextGrid"))


def load_utterances(root: Path, cid: str) -> list[Utt]:
    utts: list[Utt] = []
    for role in ("doctor", "patient"):
        tg = textgrid.TextGrid.fromFile(str(root / "transcripts" / f"{cid}_{role}.TextGrid"))
        for tier in tg.tiers:
            for iv in tier.intervals:
                txt = clean_text(iv.mark or "")
                if not txt:
                    continue
                utts.append(Utt(role.capitalize(), float(iv.minTime), float(iv.maxTime), txt))
    utts.sort(key=lambda u: u.start)
    return utts


def reference_text(utts: list[Utt]) -> str:
    return " ".join(u.text for u in utts)


def reference_dialogue(utts: list[Utt]) -> str:
    """ACI-Bench style speaker-tagged dialogue, one utterance per line."""
    return "\n".join(f"[{u.speaker.lower()}] {u.text}" for u in utts)


def to_rttm(utts: list[Utt], cid: str) -> str:
    lines = []
    for u in utts:
        lines.append(f"SPEAKER {cid} 1 {u.start:.3f} {u.end - u.start:.3f} <NA> <NA> {u.speaker} <NA> <NA>")
    return "\n".join(lines) + "\n"


def load_note(root: Path, cid: str) -> dict:
    return json.loads((root / "notes" / f"{cid}.json").read_text())


def export(root: Path, out: Path) -> None:
    """Write one JSON per consultation with utterances, reference text, dialogue, note; plus RTTMs."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "rttm").mkdir(exist_ok=True)
    for cid in consultation_ids(root):
        utts = load_utterances(root, cid)
        rec = {
            "id": cid,
            "audio": str(root / "mixed" / f"{cid}.wav"),
            "utterances": [asdict(u) for u in utts],
            "reference_text": reference_text(utts),
            "dialogue": reference_dialogue(utts),
            "note": load_note(root, cid),
        }
        (out / f"{cid}.json").write_text(json.dumps(rec, indent=1))
        (out / "rttm" / f"{cid}.rttm").write_text(to_rttm(utts, cid))


if __name__ == "__main__":
    import sys
    export(Path(sys.argv[1]), Path(sys.argv[2]))
    print("exported", len(consultation_ids(Path(sys.argv[1]))), "consultations to", sys.argv[2])
