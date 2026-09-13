# Naming the clinician fixes attribution; anonymous labels do not

**Tag: transferred** (mechanistic; seen first on ACI/PriMock attribution judging, then as the loop's first accepted change, then in the vanilla-vs-v1 tie).

- Notes written from the diarized MOSS-TD transcript with anonymous S01/S02 labels carried **0.19 misattributions per note**, the same as notes from the undiarized streaming transcript. Notes from the human transcript (Doctor/Patient labelled) carried **0.05**.
- The vanilla pipeline and the first two-pass design (diarized, anonymous labels) **tie** on the loop composite: dev 44.17 vs 44.26, test 40.82 vs 39.62 ([[vanilla-vs-best]]).
- Adding an LLM role map (S01/S02 → Doctor/Patient, one 60-token call per transcript, cached) was the loop's first accepted change: dev 44.26 → 45.74, test 39.62 → 41.79. Ablation on dev: about +0.9.

Caveat (13 Sep): the per-note misattribution counts come from one judge; a three-judge check on the 74 held-out notes found pairwise κ between −0.02 and +0.29 and one judge reversing the direction ([[judges]]). The mechanism claim stands on the dev-set acceptance and the human-transcript comparison; the size of the effect does not.

Implication for the design: diarization is necessary but not sufficient; the note model has to know *which* speaker is the clinician. One line in the pipeline, and only a clinician-defined dimension (attribution) would have caught it.

Sources: `autoresearch/loop.py` (`ROLE_SYSTEM`, `transcript_for`), `autoresearch/notebook.md` iterations 1–4, artifact section 6 callout.
