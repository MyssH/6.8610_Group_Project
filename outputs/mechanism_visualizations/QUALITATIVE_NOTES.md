# Qualitative Mechanism Notes

These notes are automatically derived from the generated summaries and are intended as prompts for inspection, not final-report claims.

## Exp A
- Layer 6: lowest mean LID is `templatic_repetition` on 32B (1.38); highest is `regular_gsm8k_nothink` on 14B (16.21).
- Layer 13: lowest mean LID is `templatic_repetition` on 32B (1.51); highest is `regular_gsm8k_nothink` on 32B (13.70).
- Layer 20: lowest mean LID is `templatic_repetition` on 1.7B (2.89); highest is `templatic_repetition` on 4B (20.13).

## Exp B
- Layer 6 average ordering across scale: canonical (14.41) < repetitive_filler (14.43) < irrelevant_context (14.62).
- Layer 13 average ordering across scale: repetitive_filler (12.00) < canonical (12.00) < irrelevant_context (12.12).
- Layer 20 average ordering across scale: canonical (12.01) < repetitive_filler (12.08) < irrelevant_context (12.24).

## Exp C
- Layer 6 mean segment ordering across scale: no_think_answer (14.42) < think_answer (14.92) < thinking (15.13).
- Layer 13 mean segment ordering across scale: thinking (11.63) < no_think_answer (12.05) < think_answer (12.47).
- Layer 20 mean segment ordering across scale: think_answer (12.07) < no_think_answer (12.13) < thinking (13.74).

## Paired Deltas
- `think_answer_minus_no_think_answer`: mean over model/layer summaries 0.28; average fraction below zero 0.44.
- `thinking_minus_no_think_answer`: mean over model/layer summaries 0.63; average fraction below zero 0.52.
- `thinking_minus_think_answer`: mean over model/layer summaries 0.35; average fraction below zero 0.55.

## Trajectory Dynamics
- `min_lid` phase ordering in sampled token diagnostics: thinking (2.70) < think_answer (4.15) < no_think_answer (4.21).
- `variance_lid` phase ordering in sampled token diagnostics: no_think_answer (76.38) < think_answer (87.27) < thinking (4457.46).
- `slope_lid` phase ordering in sampled token diagnostics: think_answer (-6.06) < no_think_answer (-4.31) < thinking (-4.12).
- `early_late_diff` phase ordering in sampled token diagnostics: think_answer (-4.75) < no_think_answer (-3.50) < thinking (-3.19).
- `fraction_below_low_threshold` phase ordering in sampled token diagnostics: no_think_answer (0.04) < think_answer (0.04) < thinking (0.12).

## Correctness
- `no_think_answer` correct-minus-wrong mean LID gap: 0.30.
- `think_answer` correct-minus-wrong mean LID gap: 0.44.
- `thinking` correct-minus-wrong mean LID gap: 3.11.
