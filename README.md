# Qwen3 LID Experiment Suite

This repository runs local activation-space Local Intrinsic Dimensionality (LID) experiments for Qwen3 models.
It supports degenerate generation controls, GSM8K think vs no-think comparisons, and GSM8K prompt-structure controls.

## Setup On macOS

Use the provided conda environment, then install the package requirements:

```bash
conda activate ds311
pip install -r requirements.txt
```

The code prefers MPS on Apple Silicon, falls back to CUDA when available in other environments, and otherwise uses CPU.
Model loading uses `float16` on MPS/CUDA and `float32` on CPU.
Supported model ids are `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, and `Qwen/Qwen3-14B`.

## Bootstrap Assets

Download the Qwen3 model snapshot and cache GSM8K under the project:

```bash
python scripts/bootstrap_assets.py
python scripts/bootstrap_assets.py --model-id Qwen/Qwen3-1.7B --dataset-name openai/gsm8k
```

The model is stored under `data/caches/models/` and GSM8K is saved to `data/raw/gsm8k/`.
Later runs reuse those local paths when present.

## Run Experiments

Experiment A:

```bash
python scripts/run_exp_a.py
python scripts/run_exp_a.py --n-baseline 30 --k 10 --layers 6 13 20
```

Experiment B now runs the prompt-structure control in no-thinking mode:

```bash
python scripts/run_exp_b.py
python scripts/run_exp_b.py --n-samples 100 --k 10 --layers 6 13 20 --sampling-profile official_recommended
```

Experiment C now runs the canonical think vs no-think comparison:

```bash
python scripts/run_exp_c.py
python scripts/run_exp_c.py --n-samples 200 --k 10 --layers 6 13 20 --sampling-profile official_recommended
```

Each runner resumes from saved JSONL and NPZ artifacts unless `--no-resume` is passed.
Use `--max-new-tokens` for short smoke runs.
Use `--model-id` on any experiment runner to switch among supported Qwen3 models.

## Outputs

Each experiment writes to its own output directory:

- `outputs/exp_a/`
- `outputs/exp_b/`
- `outputs/exp_c/`

Artifacts include:

- `raw_generations.jsonl`
- `hidden_states/*.npz`
- `sample_metrics.csv`
- `pair_metrics.csv`
- `summary_metrics.csv`
- `run_manifest.json`
- `figures/*.png` and `figures/*.pdf`

## Replot Only

Regenerate plots from saved CSV artifacts without rerunning inference:

```bash
python scripts/plot_all.py
```

Overview copies are placed under `outputs/plots/`.

## Implementation Notes

Generation uses a custom autoregressive loop with KV cache. It samples one token at a time, stores generated token ids, and captures selected-layer hidden states aligned to generated tokens.
The default sampling profile follows Qwen's recommended parameters: thinking uses temperature `0.6`, top-p `0.95`, top-k `20`, min-p `0`; non-thinking uses temperature `0.7`, top-p `0.8`, top-k `20`, min-p `0`.
The default total generation budget is `1024` tokens. Thinking-mode generation may emit `</think>` at any time, but the loop forces `</think>` once the thinking segment reaches `512` tokens.
Thinking segmentation uses Qwen's generated-token boundary for the final `</think>` token id, then falls back to text parsing only when token-level data is unavailable.
Experiment C metrics and plots filter out paired examples when the think-mode `thinking_segment` is too short to produce valid LID on every selected layer; both the think and no-think rows are excluded from the main analysis for those examples.
The main LID path uses float32 Euclidean distances with no hidden-state normalization; `--normalize-hidden-states` enables the optional sensitivity path.

## Colab

Use [colab_qwen_lid_experiments.ipynb](/Users/myssh/Desktop/NLP_Final/notebooks/colab_qwen_lid_experiments.ipynb) on an A100 runtime to download GSM8K, download a selected Qwen3 model, run Experiments A/B/C with corrected names, and copy the full output folder to Google Drive.
