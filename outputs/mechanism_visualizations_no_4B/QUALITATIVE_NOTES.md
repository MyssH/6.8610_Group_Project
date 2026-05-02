# Qualitative Mechanism Notes

These notes are automatically derived from the generated summaries and are intended as prompts for inspection, not final-report claims.

## Exp A
- Layer 6: lowest mean LID is `templatic_repetition` on 32B (1.38); highest is `regular_gsm8k_nothink` on 14B (16.21).
- Layer 13: lowest mean LID is `templatic_repetition` on 32B (1.51); highest is `regular_gsm8k_nothink` on 32B (13.70).
- Layer 20: lowest mean LID is `templatic_repetition` on 1.7B (2.89); highest is `regular_gsm8k_nothink` on 32B (12.94).

## Exp B
- Layer 6 average ordering across scale: canonical (14.77) < repetitive_filler (14.80) < irrelevant_context (14.93).
- Layer 13 average ordering across scale: repetitive_filler (12.13) < canonical (12.16) < irrelevant_context (12.26).
- Layer 20 average ordering across scale: canonical (12.14) < repetitive_filler (12.21) < irrelevant_context (12.41).

## Exp C
- Layer 6 mean segment ordering across scale: no_think_answer (14.82) < think_answer (15.27) < thinking (15.46).
- Layer 13 mean segment ordering across scale: thinking (11.86) < no_think_answer (12.19) < think_answer (12.57).
- Layer 20 mean segment ordering across scale: think_answer (12.21) < no_think_answer (12.29) < thinking (13.57).

## Paired Deltas
- `think_answer_minus_no_think_answer`: mean over model/layer summaries 0.25; average fraction below zero 0.44.
- `thinking_minus_no_think_answer`: mean over model/layer summaries 0.53; average fraction below zero 0.52.
- `thinking_minus_think_answer`: mean over model/layer summaries 0.28; average fraction below zero 0.55.

## Trajectory Dynamics
- `min_lid` phase ordering in sampled token diagnostics: thinking (2.70) < think_answer (4.16) < no_think_answer (4.24).
- `variance_lid` phase ordering in sampled token diagnostics: no_think_answer (87.22) < think_answer (95.02) < thinking (3388.98).
- `slope_lid` phase ordering in sampled token diagnostics: think_answer (-5.60) < thinking (-5.46) < no_think_answer (-4.44).
- `early_late_diff` phase ordering in sampled token diagnostics: thinking (-4.52) < think_answer (-4.28) < no_think_answer (-3.58).
- `fraction_below_low_threshold` phase ordering in sampled token diagnostics: no_think_answer (0.04) < think_answer (0.04) < thinking (0.12).

## Correctness
- `no_think_answer` correct-minus-wrong mean LID gap: 0.28.
- `think_answer` correct-minus-wrong mean LID gap: 0.77.
- `thinking` correct-minus-wrong mean LID gap: 3.69.

