# LID-of-Thought: A Geometric Marker of Correct Reasoning in Qwen3

> Local Intrinsic Dimensionality (LID) of hidden-state trajectories during
> autoregressive generation, measured in Qwen3 models from **1.7B to 32B** on
> GSM8K. The thinking-phase LID of *correct* reasoning trajectories is
> reliably higher than that of *wrong* ones — at every scale we tested.

<p align="center">
  <img src="report/Images/Allmodel_expC_auc_correctness_layer13.png" width="78%" alt="Across Qwen3 scales, correct thinking trajectories have higher LID than incorrect ones, and the correctness gap is largest in the thinking phase.">
</p>

## TL;DR

Two controls and one main experiment:

- **(A) Validity.** LID cleanly separates degenerate repetitive generation
  from ordinary GSM8K generation at every Qwen3 scale.
- **(B) Robustness.** Padding the prompt with irrelevant context or with
  *repetitive boilerplate* does not move the answer-segment LID. LID tracks
  the task, not the prompt's surface form.
- **(C) Reasoning signature.** Paired think vs. no-think comparison on
  matched GSM8K problems yields two findings:
  - **C.1.** The relative ordering of mean LID across the *thinking*,
    *think-answer*, and *no-think-answer* segments **is not stable across
    model scale** — paired Wilcoxon `thinking − answer` differences flip
    sign between 8B and 14B.
  - **C.2 (main).** Stratifying by final-answer correctness, the thinking
    segment of correct trajectories has **higher** mean LID than the
    thinking segment of wrong trajectories — **at every scale**, one-sided
    Mann–Whitney `p ≤ 0.05`. The gap inside the thinking segment is several
    times larger than the gap in either answer segment.

We read LID along the thinking trajectory not as a fixed signature of the
thinking *mode*, but as a geometric correlate of *successful* reasoning.

The full write-up is in [`report/main.tex`](report/main.tex).

## Repository layout

```
.
├── README.md
├── pyproject.toml          installable package (qwen-lid)
├── requirements.txt
├── configs/                YAML configs for the three experiments
│   ├── common.yaml         models, layers, k, sampling profile
│   ├── exp_a.yaml          degenerate-control parameters
│   ├── exp_b.yaml          prompt-structure parameters
│   └── exp_c.yaml          think vs. no-think parameters
├── src/qwen_lid/           library code
│   ├── lid.py              Levina–Bickel k-NN LID estimator (torch/numpy)
│   ├── generation.py       custom autoregressive loop with KV cache
│   ├── segmentation.py     thinking vs. answer span identification
│   ├── prompts.py          GSM8K visible-answer template + Exp A/B variants
│   ├── mechanism_visualization.py
│   │                       multi-model trajectory plots and tables
│   └── experiments/        per-experiment drivers (exp_a, exp_b, exp_c)
├── scripts/                CLI entry points (see below)
├── tests/                  pytest suite (LID, prompts, segmentation, …)
├── notebooks/              Colab notebook for A100 runs
├── outputs/                per-model CSV metrics + figures (large; see notes)
└── report/                 ICML-style paper, bib, sty, and all figures
    ├── main.tex
    ├── reference.bib
    └── Images/             every figure cited by the paper
```

## Setup

```bash
# Recommended: a fresh Python 3.11 environment
pip install -r requirements.txt
# (optional) editable install of the library
pip install -e .
```

Apple Silicon (MPS), CUDA, and CPU are all supported; `float16` is used on
MPS/CUDA and `float32` on CPU. Supported model ids:
`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`,
`Qwen/Qwen3-32B`.

## Quick start

```bash
# 1. cache GSM8K and a model checkpoint under data/
python scripts/bootstrap_assets.py --model-id Qwen/Qwen3-1.7B

# 2. run any of the three experiments (CLI flags override configs/*.yaml)
python scripts/run_exp_a.py    # validity (degenerate controls)
python scripts/run_exp_b.py    # robustness (prompt structure)
python scripts/run_exp_c.py    # think vs. no-think (main)

# 3. build the multi-model trajectory plots and cross-model tables
python scripts/run_mechanism_visualizations.py

# 4. paired Wilcoxon / Mann–Whitney tests reported in the paper
python scripts/run_wilcoxon_tests.py
```

Each experiment runner resumes from previously saved JSONL/NPZ artifacts
unless `--no-resume` is passed. Use `--max-new-tokens` for short smoke runs
and `--model-id` to switch models. Plots can be regenerated from saved
metrics with `python scripts/plot_all.py`.

## Reproducing the figures and tables in the paper

The figures in `report/Images/` and the tables in `report/main.tex` come
from the small set of CSV / PNG artifacts under `outputs/`. Concretely:

| Paper element                          | Source |
| -------------------------------------- | ------ |
| Exp. A figures (Fig. 1, A-layer 6/20)  | `outputs/Qwen_Qwen3-*/exp_a/figures/` + `mechanism_visualizations/exp_a/` |
| Exp. B figures (Fig. 2, B-layer 6/20)  | `outputs/Qwen_Qwen3-*/exp_b/figures/` + `mechanism_visualizations/exp_b/` |
| Exp. C, C.1 segment plots              | `outputs/Qwen_Qwen3-1.7B/exp_c/figures/` + `mechanism_visualizations/exp_c/trajectory_statistics/` |
| Exp. C, C.2 correctness plots          | `outputs/Qwen_Qwen3-*/exp_c/figures_mechanism/correctness_stratified/` |
| Table 1 (C.1 Wilcoxon)                 | `outputs/mechanism_visualizations/data/wilcoxon_exp_c_c1.csv` |
| Table 2 (C.2 Mann–Whitney)             | `outputs/mechanism_visualizations/data/wilcoxon_exp_c_c2.csv` |

The Wilcoxon CSVs are produced by `scripts/run_wilcoxon_tests.py` and are
computed from each model's *full* `sample_metrics.csv` (no subsampling).

## Building the paper

```bash
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
# produces report/main.pdf  (gitignored)
```

`icml2026.sty`, `icml2026.bst`, and `reference.bib` are all included in
`report/`.

## Tests

```bash
pytest -q
```

The suite covers the LID estimator, GSM8K answer parsing, prompt
construction, and think/no-think segmentation.

## Implementation notes

- LID uses the Levina–Bickel k-NN maximum-likelihood estimator with `k=10`,
  computed on raw Euclidean distances over `float32` hidden states (no
  normalization). The `--normalize-hidden-states` flag enables an optional
  unit-norm sensitivity path.
- Generation is a custom autoregressive loop with a KV cache, sampling one
  token at a time, storing only generated-token hidden states (no prompt /
  padding rows). Sampling follows Qwen's recommended parameters: think
  mode `T=0.6, top-p=0.95, top-k=20`, no-think `T=0.7, top-p=0.8, top-k=20`.
- The thinking budget is capped at 512 tokens, total at 1024. Thinking
  segments are bounded by Qwen's `</think>` token id (`151668`) when
  available, with a text-level fallback.
- Exp. C filters out paired examples whose thinking segment is too short to
  produce a valid LID estimate on every selected layer; both the think and
  no-think rows are dropped from the main analysis for those examples.

## Citation

If you use this code or analysis, please cite:

```bibtex
@unpublished{huang2026lidthinking,
  title  = {Local Intrinsic Dimensionality of Hidden-State Trajectories:
            A Geometric Marker of Correct Reasoning in Qwen3},
  author = {Huang, Kaiyuan and Shen, Gefei and Feng, Qiuyang},
  year   = {2026},
  note   = {Harvard / MIT; under review.}
}
```
