# Metrics

## Transcript layer (scored once per ASR system, all 57 files)
- **WER**: (substitutions + deletions + insertions) / reference words, after symmetric normalization (fillers, UK/US spelling, digits). Dominated by fillers and contractions; a weak signal for a scribe.
- **Medical-term miss rate**: share of the 1,741 lexicon tokens present in the human transcripts that the hypothesis lost. Lexicon: 7,092 primary-care terms from MTS-Dialog + MedSynth ([[datasets]]). The number the doctor feels.
- **DER**: diarization error rate, pyannote, 250 ms collar, against per-speaker TextGrid timestamps.
- **Word-level speaker misattribution**: share of words assigned to the wrong speaker (clinician→patient and patient→clinician reported separately).

## Note layer
- **ROUGE-L** against the clinician's note (stemmed). PriMock references are terse UK GP notes (~136 words) vs ~250-word model notes, so absolute values are low by construction; ordering is what matters.
- **BERTScore F1** (deberta-xlarge-mnli, tokenizer cap patched). ACI only.
- **Term recall / term precision**: lexicon terms shared by reference and draft, over reference terms / over draft terms. Precision is the hallucination guard.
- **Plan-item recall**: Qwen judge extracts plan items from the reference note, then checks each against the draft. Proxy for Abridge's completeness dimension. **Flaw found 12 Sep**: ~27% of reference items are never spoken; should count transcript-stated items only ([[plan-recall-gap]]).
- **Misattributions per note**: Qwen judge (Abridge's attribution dimension: four misattribution types) reads the draft against the *human* transcript. Third-party-as-patient counted separately on ACI.
- **Grounded**: share of note sentences the judge finds supported by the human transcript, whole transcript as evidence. **Linked**: share of sentences whose own cited lines (±1) support them; uncited notes score 0. **cited_frac**: share of sentences carrying a citation ([[citations-and-verifiability]]).
- **Edit effort**: character edits (Levenshtein) to turn the draft into the reference, per 100 reference words. **len_ratio**: draft words / reference words.
- **Verifier recall / clean-flag rate** on injected errors ([[verifier]]).
- Expert-iteration readouts: **passive_per_100w** (passive constructions), **agentless_frac** (sentences with no agent word; saturated at ~88% for bullet-style notes, so a weak detector) ([[expert-iteration]]).

## Composites
- **Loop composite** = 0.3 × term recall + 0.3 × term precision + 0.2 × ROUGE-L + 0.2 × plan recall − 40 × misattributions per note. Mine, not Abridge's; borrows two of their dimensions and adds two overlap/lexicon terms. No verifiability term (the flaw), no specialty stratification, no fairness term, judges not clinician-validated.
- **composite_v** = composite + 0.2 × grounded (added 12 Sep).
- **Expert-iteration reward** = composite + 0.2 × grounded + 0.1 × linked − 20 × max(0, |len_ratio − r0| − 0.6), r0 = round-0 mean len_ratio (2.06).
- **Clinician-experience scorecard** (framing, not a number): safety, completeness, fit, effort, trust.

Sources: `scribe_bench/asr_score.py`, `note_score.py`, `note_judge.py`, `verifier.py`, `autoresearch/loop.py` (`composite`), `autoresearch/verif_eval.py`, `autoresearch/expert_iter.py`.
