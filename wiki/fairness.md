# Fairness stratification

ACI-Bench, plain notes from the human transcript, Qwen3.8-27B, stratified by the patient metadata:
- **Gender**: ROUGE-L gaps within one point on every split.
- **Age**: patients under 40 score 57–61% medical-term recall vs 64–71% for other bands; patients 65+ score lowest on ROUGE-L but highest on term recall. Groups are 3–26 encounters → a lead to check on real data, not a finding.

The judge flags used for attribution/completeness include false positives and would need the clinician-alignment step the vendor evaluation literature describes before being treated as metrics ([[judges]]).

Sources: `runs/fairness.log`, artifact section 6.
