# Qualitative Mechanism Notes

These notes are automatically derived from the generated summaries and are intended as prompts for inspection, not final-report claims.

## Exp A
- Layer 6: lowest mean LID is `templatic_repetition` on 32B (2.691e-04); highest is `regular_gsm8k_nothink` on 1.7B (7.011e-03).
- Layer 13: lowest mean LID is `templatic_repetition` on 32B (2.941e-04); highest is `regular_gsm8k_nothink` on 1.7B (5.341e-03).
- Layer 20: lowest mean LID is `templatic_repetition` on 32B (6.409e-04); highest is `templatic_repetition` on 4B (7.864e-03).

## Exp B
- Layer 6 average ordering across scale: canonical (4.270e-03) < repetitive_filler (4.297e-03) < irrelevant_context (4.336e-03).
- Layer 13 average ordering across scale: canonical (3.550e-03) < repetitive_filler (3.552e-03) < irrelevant_context (3.578e-03).
- Layer 20 average ordering across scale: canonical (3.585e-03) < repetitive_filler (3.607e-03) < irrelevant_context (3.647e-03).

## Exp C
- Layer 6 mean segment ordering across scale: thinking (4.250e-03) < no_think_answer (4.257e-03) < think_answer (4.405e-03).
- Layer 13 mean segment ordering across scale: thinking (3.364e-03) < no_think_answer (3.562e-03) < think_answer (3.687e-03).
- Layer 20 mean segment ordering across scale: think_answer (3.596e-03) < no_think_answer (3.612e-03) < thinking (3.974e-03).

## Paired Deltas
- `think_answer_minus_no_think_answer`: mean over model/layer summaries 8.559e-05; average fraction below zero 0.44.
- `thinking_minus_no_think_answer`: mean over model/layer summaries 5.236e-05; average fraction below zero 0.52.
- `thinking_minus_think_answer`: mean over model/layer summaries -3.323e-05; average fraction below zero 0.55.

## Trajectory Dynamics
- `min_lid` phase ordering in sampled token diagnostics: thinking (7.922e-04) < think_answer (1.286e-03) < no_think_answer (1.306e-03).
- `variance_lid` phase ordering in sampled token diagnostics: no_think_answer (7.037e-06) < think_answer (8.301e-06) < thinking (3.002e-04).
- `slope_lid` phase ordering in sampled token diagnostics: thinking (-1.703e-03) < think_answer (-1.361e-03) < no_think_answer (-9.230e-04).
- `early_late_diff` phase ordering in sampled token diagnostics: thinking (-1.336e-03) < think_answer (-1.040e-03) < no_think_answer (-7.341e-04).
- `fraction_below_low_threshold` phase ordering in sampled token diagnostics: no_think_answer (0.03) < think_answer (0.04) < thinking (0.12).

## Correctness
- `no_think_answer` correct-minus-wrong mean LID gap: -6.899e-04.
- `think_answer` correct-minus-wrong mean LID gap: -8.647e-04.
- `thinking` correct-minus-wrong mean LID gap: 3.258e-05.

