"""Text normalization for WER. Deliberately simple and symmetric: lowercase,
strip punctuation, collapse fillers and whitespace. Applied identically to
reference and hypothesis."""
import re

FILLERS = {"um", "uh", "mm", "hmm", "mhm", "erm", "er", "ah", "uh-huh", "mm-hmm"}
PUNCT = re.compile(r"[^\w\s']")
# British -> American spellings seen in UK clinical speech; applied to both sides.
UK_US = {
    "diarrhoea": "diarrhea", "anaemia": "anemia", "anaesthetic": "anesthetic", "oedema": "edema",
    "oesophagus": "esophagus", "haemorrhage": "hemorrhage", "haemoglobin": "hemoglobin", "foetus": "fetus",
    "paediatric": "pediatric", "tumour": "tumor", "colour": "color", "fibre": "fiber", "centre": "center",
    "practise": "practice", "organise": "organize", "realise": "realize", "recognise": "recognize",
    "counselling": "counseling", "programme": "program", "mum": "mom", "whilst": "while", "ok": "okay",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
}
NUMWORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize(text: str) -> str:
    t = text.lower().replace("-", " ")
    t = PUNCT.sub(" ", t)
    toks = []
    for w in t.split():
        w = w.strip("'")
        if not w or w in FILLERS:
            continue
        w = UK_US.get(w, w)
        toks.append(NUMWORDS.get(w, w))
    return " ".join(toks)
