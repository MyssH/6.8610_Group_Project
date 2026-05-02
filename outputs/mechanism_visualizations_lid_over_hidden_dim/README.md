# Mechanism Visualization Outputs

This directory contains exploratory mechanism-discovery plots generated from the multi-model output folders.

- Models: Qwen_Qwen3-1.7B, Qwen_Qwen3-4B, Qwen_Qwen3-8B, Qwen_Qwen3-14B, Qwen_Qwen3-32B
- Layers: 6, 13, 20
- LID k: 10
- LID value normalization: hidden_dim
- Token-level recomputation device: cpu
- Max tokens per token-level LID recomputation: 256
- Max Exp C examples per model for token-level segment diagnostics: 30

Aggregate scale/correctness/delta plots use the complete existing `sample_metrics.csv` and `pair_metrics.csv` tables.
Token-level trajectory, heatmap, boundary, and token-category plots are recomputed from hidden states and cached under each experiment's `figures_mechanism_lid_over_hidden_dim/cache/` directory.

Main subdirectories:

- `exp_a/mean_lid_comparisons/`: scale comparisons for repetition and regular GSM8K families.
- `exp_b/mean_lid_comparisons/`: corrected Exp B prompt-condition answer LID comparisons.
- `exp_c/mean_lid_comparisons/`: thinking, think-answer, and no-think-answer segment comparisons.
- `exp_c/delta_distributions/`: paired delta histograms, ECDFs, violins, and delta summaries.
- `exp_c/accuracy_by_model/`: Exp C answer accuracy by model size using each model's own kept examples.
- `exp_c/boundary_anchored/`: trajectories aligned to thinking start and thinking-to-answer transition.
- `exp_c/normalized_trajectories/`: average segment-shape plots after normalizing token position.
- `exp_c/sample_token_heatmaps/`: sample-by-token normalized heatmaps sorted by correctness, mean LID, and paired delta.
- `exp_c/trajectory_statistics/`: minimum, variance, slope, early-vs-late, low-LID fraction, AUC, and boundary-jump views. Aggregate correctness plots use the common kept-example subset across the included models; `by_model_correctness/` uses each model's own kept examples.
- `exp_c/token_category_diagnostics/`: token-category LID summaries when the tokenizer is available locally.
- `data/`: joined input metrics and derived CSV summaries, including `exp_c_common_kept_token_trajectory_stats.csv` and `exp_c_accuracy_by_model_kept_per_model.csv`.

Per-model raw trajectory examples are saved in each model folder under `<model>/<exp>/figures_mechanism_lid_over_hidden_dim/per_sample_trajectories/`.
