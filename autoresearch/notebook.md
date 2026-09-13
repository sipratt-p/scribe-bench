# Autoresearch notebook

Started 2026-09-12 00:51. Dev = 20 PriMock consultations, test = 37. Cached ASR variants: ['canary_qwen', 'moss_hot150', 'moss_hotcc', 'moss_plain']. Composite = 0.3 term_recall + 0.3 term_precision + 0.2 ROUGE-L + 0.2 plan_recall - 40 misattrib/note.

| iter | asr | corr | role | prompt | extra | cite | gate | dev score | ΔASR-vs-human | test score | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | moss_plain | 0 | none | base [qwen27b] |  | 0 | none | 44.26 (TR 51.86, TP 33.41, RL 28.13, plan 75.27, mis 0.05) | -4.24 (human 48.5) | 39.62 | ACCEPTED as best (judge2 44.76) |
| 2 | moss_plain | 0 | llm | base [qwen27b] |  | 0 | none | 45.74 (TR 51.44, TP 31.54, RL 27.66, plan 76.6, mis 0.0) | -2.76 (human 48.5) | 41.79 | dev+test gain, but judge 2 disagreed (44.65 vs 44.76) |
| 3 | moss_plain | 0 | llm | attrib [qwen27b] |  | 0 | none | 44.59 (TR 52.71, TP 30.84, RL 26.35, plan 71.28, mis 0.0) | -1.03 (human 45.62) |  |  |
| 4 | moss_plain | 0 | llm | base [qwen27b] |  | 0 | none | 45.74 (TR 51.44, TP 31.54, RL 27.66, plan 76.6, mis 0.0) | -2.76 (human 48.5) | 41.79 | ACCEPTED as best (judge2 44.65) |
| 5 | moss_plain | 1 | llm | attrib [qwen27b] |  | 0 | none | 44.42 (TR 52.87, TP 30.81, RL 26.37, plan 70.21, mis 0.0) | -1.2 (human 45.62) |  |  |
| 6 | moss_plain | 0 | llm | attrib [qwen27b] |  | 1 | none | 44.06 (TR 53.55, TP 33.95, RL 25.65, plan 73.4, mis 0.05) | -3.42 (human 47.48) |  |  |
| 7 | moss_plain | 0 | llm | attrib [qwen27b] |  | 1 | drop_flagged | 41.62 (TR 47.83, TP 38.11, RL 25.36, plan 63.83, mis 0.05) | -4.74 (human 46.36) |  |  |
| 8 | moss_plain | 0 | llm | attrib_strict [qwen27b] |  | 1 | drop_flagged | 39.72 (TR 42.31, TP 36.56, RL 22.85, plan 57.45, mis 0.0) | +0.28 (human 39.44) |  |  |
| 9 | moss_plain | 1 | llm | attrib_strict [qwen27b] |  | 1 | drop_flagged | 39.71 (TR 43.49, TP 34.16, RL 22.51, plan 59.57, mis 0.0) | +0.27 (human 39.44) |  |  |
| 10 | moss_plain | 0 | llm | attrib +scaffold [qwen27b] |  | 1 | none | 44.41 (TR 49.78, TP 32.96, RL 24.52, plan 73.4, mis 0.0) | +0.54 (human 43.87) |  |  |
| 11 | moss_plain | 0 | llm | attrib_strict +scaffold [qwen27b] |  | 1 | drop_flagged | 40.29 (TR 45.41, TP 36.54, RL 22.56, plan 65.96, mis 0.05) | -2.95 (human 43.24) |  |  |
| 12 | moss_plain | 0 | llm | attrib [dsv4flash] |  | 0 | none | 47.3 (TR 50.85, TP 34.21, RL 28.92, plan 80.0, mis 0.0) | -1.36 (human 48.66) | 44.35 | ACCEPTED as best (judge2 46.22) |
| 13 | canary_qwen | 0 | llm | attrib [qwen27b] |  | 0 | none | 43.1 (TR 52.06, TP 29.61, RL 26.39, plan 76.6, mis 0.05) | -2.52 (human 45.62) |  |  |
| 14 | moss_hot150 | 0 | llm | attrib [qwen27b] |  | 0 | none | 44.89 (TR 51.88, TP 31.08, RL 26.62, plan 73.4, mis 0.0) | -0.73 (human 45.62) |  |  |
| 15 | moss_hotcc | 0 | llm | attrib [qwen27b] |  | 0 | none | 44.61 (TR 54.75, TP 33.28, RL 27.63, plan 73.4, mis 0.05) | -1.01 (human 45.62) |  |  |
| 16 | moss_hotcc | 0 | llm | attrib [dsv4flash] |  | 0 | none | 46.81 (TR 54.32, TP 36.32, RL 28.1, plan 80.0, mis 0.05) | -1.85 (human 48.66) |  |  |
| 17 | moss_plain | 1 | llm | attrib [dsv4flash] |  | 0 | none | 46.71 (TR 52.73, TP 35.77, RL 28.7, plan 82.11, mis 0.05) | -1.95 (human 48.66) |  |  |
| 18 | moss_plain | 0 | llm | attrib [dsv4flash] | Ensure all plan items are verbatim quote | 0 | none | 47.46 (TR 51.01, TP 36.14, RL 25.51, plan 81.05, mis 0.0) | -0.27 (human 47.73) |  |  |
| 19 | moss_hotcc | 0 | llm | attrib [dsv4flash] | Ensure all plan items are verbatim quote | 0 | none | 47.29 (TR 51.23, TP 35.8, RL 23.99, plan 81.91, mis 0.0) | -1.35 (human 48.64) |  |  |
| 20 | moss_hotcc | 1 | llm | attrib [dsv4flash] | Ensure all plan items are verbatim quote | 0 | none | 45.15 (TR 50.22, TP 35.55, RL 25.18, plan 81.91, mis 0.05) | -3.49 (human 48.64) |  |  |
| 21 | moss_hotcc | 0 | llm | attrib +scaffold [dsv4flash] | Ensure all plan items are verbatim quote | 1 | none | 44.93 (TR 49.39, TP 36.27, RL 21.97, plan 84.21, mis 0.05) | +1.4 (human 43.53) |  |  |
| 22 | moss_plain | 0 | llm | base [dsv4flash] |  | 0 | none | 47.61 (TR 50.34, TP 35.27, RL 29.85, plan 79.79, mis 0.0) | +2.85 (human 44.76) |  |  |
| 23 | moss_plain | 0 | llm | base +scaffold [dsv4flash] |  | 0 | none | 47.31 (TR 50.82, TP 36.99, RL 29.05, plan 75.79, mis 0.0) | -0.62 (human 47.93) |  |  |
| 24 | moss_hotcc | 0 | llm | base +scaffold [dsv4flash] |  | 0 | none | 44.88 (TR 48.34, TP 36.55, RL 29.43, plan 77.66, mis 0.05) | -3.05 (human 47.93) |  |  |
| 25 | moss_plain | 0 | llm | base [flashnext] |  | 0 | none | 44.29 (TR 52.81, TP 31.53, RL 25.46, plan 69.47, mis 0.0) | -1.2 (human 45.49) |  |  |
| 26 | moss_plain | 0 | llm | attrib [flashnext] |  | 0 | none | 43.85 (TR 52.48, TP 32.47, RL 25.14, plan 66.67, mis 0.0) | -0.25 (human 44.1) |  |  |
| 27 | {'asr_variant': 'moss_plain', 'asr_correct': 0, 'role_map': 'llm', 'prompt': 'base', 'extra': '', 'cite': 0, 'gate': 'none', 'scaffold': 1, 'note_model': 'flashnext'} | | | | | | | ERROR APIConnectionError: Connection error. | | | |
| 28 | moss_plain | 0 | llm | base +scaffold [qwen27b] |  | 0 | none | 47.6 (TR 53.89, TP 36.37, RL 28.14, plan 74.47, mis 0.0) | +2.29 (human 45.31) |  |  |
| 29 | moss_hotcc | 0 | llm | base [dsv4flash] |  | 0 | none | 47.5 (TR 51.8, TP 36.1, RL 29.07, plan 76.6, mis 0.0) | +2.74 (human 44.76) |  |  |
| 30 | moss_plain | 0 | llm | base +scaffold [qwen27b] |  | 0 | none | 47.6 (TR 53.89, TP 36.37, RL 28.14, plan 74.47, mis 0.0) | +2.29 (human 45.31) | 40.02 | dev gain did not hold on test |
| 31 | moss_plain | 0 | llm | base [dsv4flash] |  | 0 | none | 47.61 (TR 50.34, TP 35.27, RL 29.85, plan 79.79, mis 0.0) | +2.85 (human 44.76) | 44.57 | ACCEPTED as best (judge2 46.42) |
| 32 | moss_plain | 0 | llm | base +scaffold [dsv4flash] |  | 0 | none | 47.31 (TR 50.82, TP 36.99, RL 29.05, plan 75.79, mis 0.0) | -0.62 (human 47.93) |  |  |
| 33 | moss_plain | 0 | llm | attrib [dsv4flash] | Ensure all plan items are verbatim quote | 0 | none | 45.18 (TR 48.59, TP 36.12, RL 26.74, plan 82.11, mis 0.05) | -2.43 (human 47.61) |  |  |
| 34 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. I | 0 | none | 48.45 (TR 52.62, TP 36.08, RL 29.43, plan 79.79, mis 0.0) | +0.15 (human 48.3) | 44.61 | dev+test gain, but judge 2 disagreed (45.64 vs 46.42) |
| 35 | moss_plain | 0 | llm | base [dsv4flash] | Extract the plan section strictly from t | 0 | none | 48.32 (TR 51.11, TP 41.19, RL 27.61, plan 75.53, mis 0.0) | -0.05 (human 48.37) | 43.3 | dev gain did not hold on test |
| 36 | moss_hotcc | 0 | llm | base [dsv4flash] | Focus on capturing all distinct medical  | 0 | none | 45.52 (TR 53.56, TP 34.61, RL 28.76, plan 76.6, mis 0.05) | -2.7 (human 48.22) |  |  |
| 37 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. I | 0 | none | 45.74 (TR 51.53, TP 35.15, RL 28.9, plan 79.79, mis 0.05) | -1.09 (human 46.83) |  |  |
| 38 | moss_plain | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. I | 0 | none | 45.47 (TR 50.93, TP 35.77, RL 28.33, plan 78.95, mis 0.05) | -1.36 (human 46.83) |  |  |
| 39 | moss_hotcc | 0 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. I | 0 | none | 48.19 (TR 50.65, TP 35.92, RL 28.96, plan 82.11, mis 0.0) | +1.19 (human 47.0) | 44.52 | dev gain did not hold on test |
| 40 | moss_plain | 0 | llm | base [dsv4flash] |  | 0 | drop_flagged | 47.61 (TR 50.34, TP 35.27, RL 29.85, plan 79.79, mis 0.0) | +2.85 (human 44.76) | 44.57 | dev gain did not hold on test |
| 41 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. I | 0 | none | 45.85 (TR 47.4, TP 35.34, RL 28.29, plan 76.84, mis 0.0) | -2.47 (human 48.32) |  |  |
| 42 | moss_plain | 0 | llm | base [dsv4flash] | Extract the plan section strictly from t | 0 | none | 47.29 (TR 50.01, TP 39.23, RL 26.02, plan 76.6, mis 0.0) | -0.96 (human 48.25) |  |  |
| 43 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.66 (TR 49.86, TP 37.1, RL 29.15, plan 78.72, mis 0.05) | -2.76 (human 48.42) |  |  |
| 44 | moss_plain | 0 | llm | base [qwen27b] |  | 0 | none | 45.74 (TR 51.44, TP 31.54, RL 27.66, plan 76.6, mis 0.0) | -2.76 (human 48.5) |  |  |
| 45 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. I | 0 | none | 42.35 (TR 49.69, TP 35.33, RL 28.7, plan 75.53, mis 0.1) | -3.83 (human 46.18) |  |  |
| 46 | moss_plain | 0 | llm | base [dsv4flash] | Extract the plan section strictly from t | 0 | none | 47.38 (TR 49.62, TP 37.77, RL 29.0, plan 76.84, mis 0.0) | +0.03 (human 47.35) |  |  |
| 47 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.44 (TR 53.32, TP 36.35, RL 28.74, plan 78.95, mis 0.0) | +0.24 (human 48.2) | 45.01 | ACCEPTED as best (judge2 46.8) |
| 48 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.1 (TR 49.84, TP 36.18, RL 29.61, plan 76.84, mis 0.0) | +0.84 (human 46.26) |  |  |
| 49 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 1 | drop_flagged | 41.07 (TR 40.83, TP 37.92, RL 27.02, plan 70.21, mis 0.05) | -1.31 (human 42.38) |  |  |
| 50 | moss_hotcc | 0 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.37 (TR 52.45, TP 37.32, RL 28.49, plan 78.72, mis 0.0) | +1.45 (human 46.92) | 42.75 | dev gain did not hold on test |
| 51 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.99 (TR 48.9, TP 38.68, RL 26.74, plan 76.84, mis 0.05) | -0.69 (human 45.68) |  |  |
| 52 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.9 (TR 54.66, TP 35.28, RL 28.53, plan 81.05, mis 0.0) | +0.69 (human 48.21) | 42.14 | dev gain did not hold on test |
| 53 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.18 (TR 49.07, TP 34.15, RL 30.29, plan 75.79, mis 0.05) | -2.98 (human 47.16) |  |  |
| 54 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.76 (TR 54.31, TP 34.79, RL 28.23, plan 81.91, mis 0.0) | +1.48 (human 47.28) | 40.19 | dev gain did not hold on test |
| 55 | moss_hotcc | 0 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 46.86 (TR 50.37, TP 34.59, RL 24.94, plan 81.91, mis 0.0) | -2.6 (human 49.46) |  |  |
| 56 | moss_hotcc | 0 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.34 (TR 49.06, TP 34.81, RL 28.23, plan 77.66, mis 0.05) | -3.24 (human 47.58) |  |  |
| 57 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.72 (TR 53.26, TP 35.4, RL 27.93, plan 77.66, mis 0.05) | -2.22 (human 47.94) |  |  |
| 58 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.67 (TR 48.23, TP 38.09, RL 25.12, plan 78.72, mis 0.05) | -1.85 (human 46.52) |  |  |
| 59 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. F | 0 | none | 46.36 (TR 55.46, TP 34.79, RL 28.51, plan 77.89, mis 0.05) | -1.36 (human 47.72) |  |  |
| 60 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 46.19 (TR 48.67, TP 35.91, RL 29.33, plan 74.74, mis 0.0) | -0.71 (human 46.9) |  |  |
| 61 | moss_hotcc | 0 | llm | base [qwen27b] | Prioritize high-precision terminology. E | 0 | none | 46.45 (TR 53.07, TP 32.73, RL 29.07, plan 74.47, mis 0.0) | -0.56 (human 47.01) |  |  |
| 62 | moss_plain | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.26 (TR 50.89, TP 36.75, RL 28.27, plan 76.6, mis 0.0) | -0.94 (human 48.2) |  |  |
| 63 | moss_hotcc | 0 | none | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.56 (TR 50.04, TP 37.03, RL 28.85, plan 78.35, mis 0.0) | -0.64 (human 48.2) |  |  |
| 64 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 1 | none | 44.48 (TR 45.46, TP 38.32, RL 29.07, plan 77.66, mis 0.05) | -3.11 (human 47.59) |  |  |
| 65 | moss_plain | 0 | llm | base [qwen27b] | Prioritize high-precision terminology. E | 0 | none | 46.91 (TR 52.35, TP 33.46, RL 28.16, plan 77.66, mis 0.0) | -0.1 (human 47.01) |  |  |
| 66 | moss_hotcc | 0 | llm | base [dsv4flash] | Do not hallucinate follow-up instruction | 0 | none | 46.01 (TR 50.86, TP 38.63, RL 28.97, plan 76.84, mis 0.05) | -2.8 (human 48.81) |  |  |
| 67 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. | 0 | none | 43.82 (TR 52.41, TP 35.08, RL 28.9, plan 78.95, mis 0.1) | -4.21 (human 48.03) |  |  |
| 68 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.39 (TR 44.2, TP 37.3, RL 27.02, plan 77.66, mis 0.0) | -0.84 (human 46.23) |  |  |
| 69 | moss_hotcc | 0 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.85 (TR 49.1, TP 34.37, RL 27.97, plan 81.05, mis 0.05) | -2.02 (human 46.87) |  |  |
| 70 | moss_hot150 | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.22 (TR 52.81, TP 36.04, RL 29.12, plan 78.72, mis 0.0) | +0.01 (human 48.21) |  |  |
| 71 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.75 (TR 51.6, TP 36.94, RL 29.34, plan 76.6, mis 0.0) | +0.11 (human 47.64) |  |  |
| 72 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. U | 0 | none | 46.44 (TR 49.77, TP 34.56, RL 29.12, plan 76.6, mis 0.0) | -0.81 (human 47.25) |  |  |
| 73 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.37 (TR 51.22, TP 34.63, RL 29.34, plan 78.72, mis 0.05) | -0.8 (human 46.17) |  |  |
| 74 | moss_hot150 | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 43.86 (TR 48.42, TP 36.39, RL 25.26, plan 76.84, mis 0.05) | -3.06 (human 46.92) |  |  |
| 75 | moss_hotcc | 0 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. F | 0 | none | 46.34 (TR 52.21, TP 37.39, RL 28.61, plan 78.72, mis 0.05) | -1.41 (human 47.75) |  |  |
| 76 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.9 (TR 51.84, TP 32.63, RL 24.66, plan 78.12, mis 0.0) | +0.31 (human 45.59) |  |  |
| 77 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. W | 0 | none | 44.81 (TR 48.31, TP 37.16, RL 29.26, plan 76.6, mis 0.05) | -2.87 (human 47.68) |  |  |
| 78 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. D | 0 | none | 45.62 (TR 52.34, TP 35.91, RL 29.94, plan 75.79, mis 0.05) | -1.69 (human 47.31) |  |  |
| 79 | moss_hot150 | 0 | llm | base +scaffold [dsv4flash] | Extract atomic propositions from the tra | 0 | none | 45.09 (TR 50.36, TP 36.65, RL 29.41, plan 75.53, mis 0.05) | -2.04 (human 47.13) |  |  |
| 80 | moss_hotcc | 0 | llm | base [dsv4flash] | Identify the speaker for each utterance. | 0 | none | 45.83 (TR 49.19, TP 37.05, RL 29.78, plan 80.0, mis 0.05) | -1.57 (human 47.4) |  |  |
| 81 | moss_hotcc | 0 | llm | base [dsv4flash] | When listing plan items, use the exact p | 0 | none | 46.67 (TR 48.21, TP 37.71, RL 28.93, plan 75.53, mis 0.0) | +0.72 (human 45.95) |  |  |
| 82 | moss_hot150 | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.27 (TR 50.36, TP 35.35, RL 30.13, plan 77.66, mis 0.05) | -1.6 (human 46.87) |  |  |
| 83 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.41 (TR 51.14, TP 36.04, RL 29.7, plan 76.6, mis 0.0) | +1.95 (human 45.46) |  |  |
| 84 | moss_hotcc | 0 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.11 (TR 49.89, TP 36.04, RL 28.97, plan 77.66, mis 0.0) | +0.67 (human 46.44) |  |  |
| 85 | moss_hotcc | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 1 | drop_flagged | 45.72 (TR 48.01, TP 38.65, RL 29.15, plan 69.47, mis 0.0) | +0.98 (human 44.74) |  |  |
| 86 | moss_hotcc | 0 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.35 (TR 50.35, TP 35.58, RL 29.15, plan 78.72, mis 0.05) | -1.45 (human 46.8) |  |  |
| 87 | moss_hot150 | 0 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.71 (TR 50.78, TP 35.78, RL 29.23, plan 74.47, mis 0.05) | -2.09 (human 46.8) |  |  |
| 88 | moss_hot150 | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 1 | drop_flagged | 43.71 (TR 48.73, TP 38.99, RL 29.09, plan 77.89, mis 0.1) | -1.03 (human 44.74) |  |  |
| 89 | moss_hot150 | 0 | llm | attrib_strict [dsv4flash] | Strictly attribute every clinical findin | 0 | none | 46.87 (TR 50.0, TP 36.58, RL 27.87, plan 76.6, mis 0.0) | -0.26 (human 47.13) |  |  |
| 90 | moss_hot150 | 0 | llm | base [qwen27b] | Prioritize high-precision terminology. E | 0 | none | 45.48 (TR 51.19, TP 30.86, RL 27.5, plan 76.84, mis 0.0) | -1.69 (human 47.17) |  |  |
| 91 | canary_qwen | 0 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 42.82 (TR 49.75, TP 35.43, RL 29.48, plan 76.84, mis 0.1) | -5.38 (human 48.2) |  |  |
| 92 | moss_hot150 | 0 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.26 (TR 50.43, TP 34.82, RL 28.61, plan 79.79, mis 0.0) | -1.4 (human 48.66) |  |  |
| 93 | moss_hotcc | 0 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 43.93 (TR 52.98, TP 36.48, RL 29.65, plan 75.79, mis 0.1) | -4.73 (human 48.66) |  |  |
| 94 | moss_hot150 | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.28 (TR 51.15, TP 34.96, RL 28.5, plan 78.72, mis 0.0) | -1.38 (human 48.66) |  |  |
| 95 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.78 (TR 55.67, TP 35.78, RL 29.88, plan 76.84, mis 0.0) | +1.91 (human 46.87) | 42.51 | dev gain did not hold on test |
| 96 | moss_hotcc | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 43.53 (TR 53.7, TP 35.51, RL 29.11, plan 74.74, mis 0.1) | -5.13 (human 48.66) |  |  |
| 97 | moss_hot150 | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.63 (TR 51.0, TP 35.08, RL 30.32, plan 78.72, mis 0.05) | -1.24 (human 46.87) |  |  |
| 98 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 1 | drop_flagged | 41.73 (TR 48.02, TP 37.64, RL 28.87, plan 71.28, mis 0.1) | -3.01 (human 44.74) |  |  |
| 99 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.42 (TR 49.76, TP 33.73, RL 28.98, plan 77.89, mis 0.05) | -2.4 (human 46.82) |  |  |
| 100 | moss_hot150 | 1 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.15 (TR 51.13, TP 34.86, RL 27.8, plan 78.95, mis 0.05) | -2.57 (human 47.72) |  |  |
| 101 | moss_hotcc | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 46.95 (TR 50.1, TP 35.53, RL 28.43, plan 77.89, mis 0.0) | -0.67 (human 47.62) |  |  |
| 102 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 1 | none | 45.22 (TR 50.18, TP 37.52, RL 29.01, plan 75.53, mis 0.05) | -0.19 (human 45.41) |  |  |
| 103 | moss_hotcc | 1 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.26 (TR 51.51, TP 35.99, RL 29.25, plan 75.79, mis 0.05) | -1.54 (human 46.8) |  |  |
| 104 | moss_hotcc | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.27 (TR 50.45, TP 35.41, RL 27.02, plan 75.53, mis 0.05) | -2.95 (human 47.22) |  |  |
| 105 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.4 (TR 52.81, TP 38.0, RL 29.51, plan 71.28, mis 0.05) | -2.89 (human 48.29) |  |  |
| 106 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.72 (TR 52.66, TP 36.12, RL 29.38, plan 81.05, mis 0.0) | +1.59 (human 47.13) | 41.88 | dev gain did not hold on test |
| 107 | moss_hotcc | 1 | llm | attrib_strict +scaffold [dsv4flash] | Extract atomic propositions first. For e | 0 | none | 47.33 (TR 52.33, TP 38.03, RL 27.96, plan 83.16, mis 0.05) | +0.85 (human 46.48) |  |  |
| 108 | moss_hot150 | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. W | 0 | none | 45.87 (TR 50.15, TP 35.82, RL 30.4, plan 80.0, mis 0.05) | -3.08 (human 48.95) |  |  |
| 109 | moss_hotcc | 1 | llm | strict +scaffold [dsv4flash] | Extract atomic propositions from the tra | 0 | none | 43.23 (TR 49.5, TP 37.21, RL 26.49, plan 79.57, mis 0.1) | -2.08 (human 45.31) |  |  |
| 110 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.13 (TR 51.06, TP 35.52, RL 28.94, plan 76.84, mis 0.05) | -2.44 (human 47.57) |  |  |
| 111 | moss_hot150 | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. W | 0 | none | 44.04 (TR 50.31, TP 35.79, RL 27.66, plan 73.4, mis 0.05) | -4.02 (human 48.06) |  |  |
| 112 | moss_hotcc | 1 | llm | base +scaffold [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 44.93 (TR 50.96, TP 36.13, RL 28.21, plan 75.79, mis 0.05) | -1.59 (human 46.52) |  |  |
| 113 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.24 (TR 55.09, TP 34.92, RL 28.27, plan 77.89, mis 0.0) | -0.47 (human 48.71) |  |  |
| 114 | moss_hotcc | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. W | 0 | none | 47.09 (TR 49.9, TP 35.89, RL 29.1, plan 77.66, mis 0.0) | +0.77 (human 46.32) |  |  |
| 115 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. D | 0 | none | 43.27 (TR 53.19, TP 35.64, RL 28.63, plan 74.47, mis 0.1) | -3.65 (human 46.92) |  |  |
| 116 | moss_hotcc | 1 | llm | attrib_strict +scaffold [dsv4flash] | Extract atomic propositions first. For e | 0 | none | 46.24 (TR 49.12, TP 34.84, RL 27.38, plan 77.89, mis 0.0) | +0.51 (human 45.73) |  |  |
| 117 | moss_hot150 | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. W | 0 | none | 45.66 (TR 50.98, TP 38.48, RL 28.58, plan 75.53, mis 0.05) | -3.0 (human 48.66) |  |  |
| 118 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 48.76 (TR 52.89, TP 37.32, RL 28.48, plan 80.0, mis 0.0) | +1.29 (human 47.47) | 42.57 | dev gain did not hold on test |
| 119 | moss_hotcc | 1 | llm | base [qwen27b] | Prioritize high-precision terminology. E | 0 | none | 44.74 (TR 53.07, TP 29.75, RL 25.02, plan 74.47, mis 0.0) | -1.11 (human 45.85) |  |  |
| 120 | moss_hotcc | 1 | llm | attrib_strict +scaffold [dsv4flash] | Extract atomic propositions first. For e | 0 | none | 44.76 (TR 49.52, TP 36.72, RL 26.53, plan 77.89, mis 0.05) | -1.96 (human 46.72) |  |  |
| 121 | moss_hot150 | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. W | 0 | none | 41.66 (TR 47.55, TP 36.12, RL 29.1, plan 73.68, mis 0.1) | -5.92 (human 47.58) |  |  |
| 122 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.44 (TR 49.83, TP 34.8, RL 29.38, plan 80.85, mis 0.05) | -2.16 (human 47.6) |  |  |
| 123 | moss_hotcc | 1 | llm | attrib_strict +scaffold [dsv4flash] | Extract atomic propositions first. For e | 0 | none | 47.0 (TR 50.22, TP 35.51, RL 27.69, plan 78.72, mis 0.0) | +0.25 (human 46.75) |  |  |
| 124 | moss_hot150 | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. U | 0 | none | 45.68 (TR 52.21, TP 34.73, RL 29.03, plan 78.95, mis 0.05) | +1.01 (human 44.67) |  |  |
| 125 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 45.52 (TR 51.81, TP 35.7, RL 28.67, plan 77.66, mis 0.05) | -2.1 (human 47.62) |  |  |
| 126 | moss_hotcc | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. D | 0 | none | 44.1 (TR 48.44, TP 35.12, RL 28.57, plan 76.6, mis 0.05) | -3.09 (human 47.19) |  |  |
| 127 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. U | 0 | none | 46.74 (TR 50.89, TP 33.42, RL 28.28, plan 78.95, mis 0.0) | +2.23 (human 44.51) |  |  |
| 128 | moss_hotcc | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 47.34 (TR 50.15, TP 35.86, RL 27.7, plan 80.0, mis 0.0) | -0.46 (human 47.8) |  |  |
| 129 | moss_hotcc | 1 | llm | base [dsv4flash] | Prioritize high-precision terminology. I | 0 | none | 46.86 (TR 51.51, TP 35.08, RL 27.83, plan 76.6, mis 0.0) | -0.09 (human 46.95) |  |  |
| 130 | moss_hot150 | 1 | llm | attrib_strict [dsv4flash] | Prioritize high-precision terminology. E | 0 | none | 46.64 (TR 48.69, TP 36.0, RL 28.5, plan 77.66, mis 0.0) | -1.19 (human 47.83) |  |  |
| 131 | moss_hotcc | 1 | llm | base [dsv4flash] | Explicitly list any plan items using the | 0 | none | 45.7 (TR 50.31, TP 38.46, RL 28.76, plan 76.6, mis 0.05) | -1.54 (human 47.24) |  |  |
