# scribe-bench results

### ASR — asr_score

| system | n | wer | med_term_err | der | sub | del | ins | wall_s |
|---|---|---|---|---|---|---|---|---|
| moss_plain | 57 | 10.34 | 8.39 | 11.36 | 2992 | 3706 | 1715 | 4015.1 |
| moss_hot | 57 | 10.41 | 8.21 | 11.44 | 3014 | 3724 | 1731 | 1914.2 |
| nemotron/offline | 57 | 11.79 | 12.46 |  | 3208 | 4963 | 1425 |  |
| streaming_70_13/streaming_out_nemotron-speech-streaming-en-0_manifest | 57 | 11.78 | 12.69 |  | 3209 | 4959 | 1420 |  |

### ASR — asr_score_nvidia

| system | n | wer | med_term_err | der | sub | del | ins | wall_s |
|---|---|---|---|---|---|---|---|---|
| nano_omni | 57 | 27.2 | 15.74 | 88.28 | 5812 | 6451 | 9876 | 8090.3 |
| sortformer_parakeet | 57 | 11.21 | 15.11 | 11.12 | 3535 | 3663 | 1924 | 181.0 |
| parakeet_v3/pred | 57 | 11.21 | 15.11 |  | 3535 | 3663 | 1924 |  |
| canary_qwen/pred | 57 | 11.56 | 9.59 |  | 3341 | 4327 | 1742 |  |

### Notes — note_score_test1

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 34.31 | 58.85 | 25.28 | 71.03 | 65.25 | 66.89 |  | 314 | 409 |
| asrcorr | 22 | 34.65 | 58.93 | 25.6 | 71.14 | 65.91 | 66.45 |  | 315 | 409 |
| humantrans | 22 | 34.46 | 58.37 | 25.5 | 71.05 | 65.25 | 67.0 |  | 307 | 409 |

### Notes — note_score_test2

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 33.94 | 59.46 | 25.4 | 70.57 | 68.45 | 66.87 |  | 344 | 416 |
| asrcorr | 22 | 34.04 | 59.44 | 25.57 | 70.57 | 67.19 | 66.36 |  | 333 | 416 |
| humantrans | 22 | 34.08 | 59.65 | 25.63 | 70.69 | 67.82 | 65.85 |  | 344 | 416 |

### Notes — note_score_test3

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 34.57 | 58.97 | 25.34 | 70.69 | 66.57 | 66.08 |  | 339 | 400 |
| asrcorr | 22 | 34.52 | 58.87 | 25.12 | 70.67 | 66.86 | 66.96 |  | 339 | 400 |
| humantrans | 22 | 34.02 | 58.65 | 24.91 | 70.66 | 67.01 | 66.42 |  | 339 | 400 |

### Notes — note_score_cite_test1

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 33.64 | 58.94 | 24.62 | 70.23 | 65.42 | 71.35 | 0.996 | 350 | 409 |
| asrcorr | 22 | 33.28 | 58.11 | 24.33 | 70.12 | 65.09 | 71.0 | 0.996 | 351 | 409 |
| humantrans | 22 | 33.39 | 58.01 | 24.43 | 70.33 | 64.44 | 70.92 | 0.996 | 335 | 409 |

### Notes — note_score_cite_test2

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 32.97 | 57.3 | 23.75 | 69.67 | 65.93 | 71.79 | 0.992 | 340 | 416 |
| asrcorr | 22 | 33.07 | 57.44 | 24.15 | 69.93 | 66.88 | 71.0 | 0.994 | 345 | 416 |
| humantrans | 22 | 33.16 | 57.24 | 23.99 | 69.79 | 67.19 | 71.57 | 0.994 | 341 | 416 |

### Notes — note_score_cite_test3

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 33.51 | 58.0 | 24.1 | 70.02 | 67.46 | 71.81 | 0.996 | 362 | 400 |
| asrcorr | 22 | 34.02 | 57.75 | 24.13 | 70.21 | 67.31 | 71.99 | 0.994 | 342 | 400 |
| humantrans | 22 | 33.77 | 57.81 | 24.27 | 69.93 | 65.83 | 71.09 | 0.996 | 344 | 400 |

### Notes — note_score_sft_test1

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 41.19 | 62.83 | 35.6 | 73.64 | 70.96 | 76.86 |  | 570 | 409 |
| asrcorr | 22 | 42.89 | 64.8 | 37.15 | 75.03 | 71.45 | 75.78 |  | 530 | 409 |
| humantrans | 22 | 41.74 | 63.27 | 35.99 | 73.73 | 70.64 | 77.74 |  | 561 | 409 |

### Notes — note_score_sft_test2

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 44.03 | 62.44 | 37.69 | 73.97 | 71.43 | 79.96 |  | 525 | 416 |
| asrcorr | 22 | 43.66 | 62.4 | 37.55 | 73.78 | 71.74 | 80.32 |  | 523 | 416 |
| humantrans | 22 | 43.78 | 62.36 | 37.33 | 73.53 | 72.21 | 79.04 |  | 527 | 416 |

### Notes — note_score_sft_test3

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 44.16 | 63.48 | 37.01 | 73.78 | 74.11 | 79.4 |  | 537 | 400 |
| asrcorr | 22 | 43.52 | 63.23 | 36.59 | 73.84 | 74.11 | 78.9 |  | 536 | 400 |
| humantrans | 22 | 44.03 | 64.31 | 37.45 | 73.62 | 74.56 | 79.62 |  | 525 | 400 |

### Notes — note_score_nano_test1

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 27.57 | 53.24 | 19.44 | 66.86 | 59.05 | 50.99 |  | 369 | 409 |
| asrcorr | 22 | 28.26 | 54.28 | 19.91 | 67.04 | 59.71 | 51.05 |  | 370 | 409 |
| humantrans | 22 | 27.76 | 53.69 | 19.5 | 66.91 | 58.56 | 51.14 |  | 365 | 409 |

### Notes — note_score_nano_test2

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 27.07 | 52.92 | 19.59 | 66.88 | 59.03 | 50.27 |  | 382 | 416 |
| asrcorr | 22 | 27.69 | 54.34 | 20.01 | 66.99 | 60.44 | 51.06 |  | 370 | 416 |
| humantrans | 22 | 26.37 | 52.51 | 18.82 | 66.27 | 56.83 | 48.85 |  | 374 | 416 |

### Notes — note_score_nano_test3

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| asr | 22 | 28.09 | 54.03 | 19.56 | 67.09 | 63.46 | 50.65 |  | 391 | 400 |
| asrcorr | 22 | 27.07 | 52.92 | 18.91 | 66.73 | 61.09 | 49.82 |  | 426 | 400 |
| humantrans | 22 | 27.78 | 53.1 | 18.96 | 66.69 | 60.65 | 51.38 |  | 386 | 400 |

### Notes — note_score_primock

| system | n | rougeL | rouge1 | rouge2 | bertscore_f1 | term_recall | term_precision | cited_frac | hyp_words | ref_words |
|---|---|---|---|---|---|---|---|---|---|---|
| moss_hot | 57 | 23.21 | 39.56 | 10.23 | 60.05 | 51.1 | 27.71 |  | 250 | 136 |
| moss_plain | 57 | 23.17 | 39.14 | 10.36 | 59.97 | 50.81 | 27.46 |  | 253 | 136 |
| nemotron_offline_dir | 57 | 22.97 | 39.22 | 10.45 | 59.98 | 50.07 | 27.0 |  | 244 | 136 |
| reference | 57 | 23.0 | 39.41 | 10.41 | 60.29 | 52.42 | 29.26 |  | 246 | 136 |

### Verifier — test1_report

```json
{
 "n": 905,
 "labels": {
  "SUPPORTED": 642,
  "UNSUPPORTED": 224,
  "PARTIAL": 39
 },
 "injected": 232,
 "recall_on_injected": 0.987,
 "flag_rate_on_clean": 0.055,
 "recall_by_kind": {
  "third_party": 1.0,
  "finding": 1.0,
  "negation": 0.85,
  "number": 0.88
 },
 "leak_flag_on_clean": 0.004
}
```

### Verifier — test2_report

```json
{
 "n": 968,
 "labels": {
  "SUPPORTED": 702,
  "UNSUPPORTED": 229,
  "PARTIAL": 37
 },
 "injected": 247,
 "recall_on_injected": 0.996,
 "flag_rate_on_clean": 0.035,
 "recall_by_kind": {
  "third_party": 1.0,
  "finding": 1.0,
  "negation": 0.93,
  "number": 1.0,
  "drug": 1.0
 },
 "leak_flag_on_clean": 0.007
}
```

### Verifier — test3_report

```json
{
 "n": 891,
 "labels": {
  "SUPPORTED": 636,
  "UNSUPPORTED": 216,
  "PARTIAL": 39
 },
 "injected": 229,
 "recall_on_injected": 0.987,
 "flag_rate_on_clean": 0.044,
 "recall_by_kind": {
  "third_party": 1.0,
  "finding": 0.99,
  "negation": 0.9,
  "number": 1.0,
  "drug": 1.0
 },
 "leak_flag_on_clean": 0.008
}
```
