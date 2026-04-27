# Build a Local, Maintainable Qwen3-1.7B Experiment Suite on macOS

You are implementing a **local research codebase** for activation-space Local Intrinsic Dimensionality (LID) experiments with **Qwen/Qwen3-1.7B** on **macOS**.

This document is the **single source of truth**. Follow it exactly. Do not ask for clarification unless something is impossible. Make reasonable engineering decisions consistent with this spec.

## 0. Goal

Build a Python project that can run three experiments locally on a Mac:

- **Experiment A**: degenerate-control sanity checks
- **Experiment B**: GSM8K canonical prompt, `think` vs `no_think`
- **Experiment C**: GSM8K prompt-structure control (`canonical`, `irrelevant_context`, `repetitive_filler`) crossed with `think` vs `no_think`

The system must:

1. **Download the Qwen model first**.
2. **Download the GSM8K dataset first**.
3. Keep each experiment behind a **separate entry point**.
4. Extract hidden states and compute token-level / sample-level **LID**.
5. Save raw outputs, metrics, summaries, and plots.
6. Be reasonably maintainable and easy to resume after interruption.

Use **English** everywhere in code, comments, docstrings, CLI help text, README text, plot labels, and file names.

---

## 1. Non-negotiable constraints

### 1.1 Platform

- Target environment: **macOS**, ideally Apple Silicon.
- Prefer **PyTorch + MPS** when available.
- Must gracefully fall back to CPU.
- Support CUDA, this will be tested in a CUDA environment but should not be the primary development path.

### 1.2 Model

- Use **`Qwen/Qwen3-1.7B`** from Hugging Face.
- Use the official Hugging Face `transformers` integration.
- Use chat-template generation through `tokenizer.apply_chat_template(...)`.
- Support both:
  - `enable_thinking=True`
  - `enable_thinking=False`

### 1.3 Dataset

- Use **`openai/gsm8k`** from Hugging Face Datasets.
- Default split for experiments B and C: **`test`**.
- Save a local cached copy under the project so reruns do not redownload.

### 1.4 Maintainability

Implement this as a real package, not a single script dump.

Required properties:

- `src/` package layout
- shared utilities and configs
- experiment-specific entry points
- resumable outputs
- deterministic seeding where possible
- unit tests for at least the fragile logic

### 1.5 Plotting

Each experiment must generate its own plots **automatically after metrics finish**.

Also provide one optional top-level command to regenerate all plots from saved CSV/JSONL artifacts without rerunning inference.

---

## 2. Scientific scope to implement

### 2.1 Main metric

For a segment of hidden states from one layer,

- collect hidden vectors for generated tokens in that segment,
- compute token-level Local Intrinsic Dimensionality with the **Levina–Bickel MLE estimator**,
- then compute sample-level mean LID for that segment and layer.

Formula to implement:

For token representation `h_t`, let `R_j(h_t)` be the distance to its `j`-th nearest neighbor among tokens in the same local comparison set. Then

`d_hat(t) = [ (1 / (k - 1)) * sum_{j=1}^{k-1} log( R_k(h_t) / R_j(h_t) ) ]^{-1}`

The sample-level summary is the mean over valid tokens in the segment.

### 2.2 Distance and numerical behavior

Implement the main metric as:

- hidden states converted to `float32`
- Euclidean distance
- no hidden-state normalization in the main path

Also support an optional config flag for L2-normalized hidden states as a sensitivity analysis, but do **not** make it the default.

### 2.3 Selected layers

For `Qwen3-1.7B`, use three default transformer block indices:

- `6`
- `13`
- `20`

These are 0-based block indices. Make this configurable.

### 2.4 Neighbor counts

Default `k = 10`.

Support `k in {5, 10, 20}` via config and CLI override.

### 2.5 Minimum segment length

A segment must have at least `k + 2` generated tokens to compute LID. If too short:

- mark the sample-layer-segment record as invalid,
- store the reason,
- do not crash.

---

## 3. Exact experiments to build

## 3.1 Experiment A — Degenerate-control sanity check

Purpose: verify that LID drops for geometrically collapsed or low-complexity generated trajectories.

Run only in **non-thinking mode**.

Families to include:

### A.constant_repetition

Prompt template:

```text
Output exactly the token {TOKEN} 64 times, separated by single spaces.
Do not add any other words, punctuation, or explanation.
```

Use these variants:

- `DATA`
- `ZERO`
- `TOKEN`
- `APPLE`
- `NULL`

### A.short_cycle_repetition

Prompt template:

```text
Repeat the sequence "{SEQ}" exactly 16 times.
Separate all tokens with single spaces.
Do not add anything else.
```

Use these variants:

- `RED BLUE GREEN YELLOW`
- `A B C D`
- `CAT DOG BIRD FISH`
- `ONE TWO THREE FOUR`
- `LEFT RIGHT UP DOWN`

### A.templatic_repetition

Prompt template:

```text
Repeat the exact string
"{TEMPLATE}"
exactly 8 times, separated by single spaces.
Do not add anything else.
```

Use these variants:

- `Summary: same. Equation: 1+1=2. Final answer: 2.`
- `Summary: same. Equation: 2+2=4. Final answer: 4.`
- `Summary: same. Equation: 3+3=6. Final answer: 6.`
- `Summary: same. Equation: 5-5=0. Final answer: 0.`
- `Summary: same. Equation: 5+5=10. Final answer: 10.`

### A.baseline_regular_generation

Add a small regular-generation baseline so plots are not degenerate-only.

Use the **canonical GSM8K prompt format from Experiment B** on a small subset of GSM8K in `no_think` mode.

Default count: `n_baseline = 30` GSM8K test examples.

### A outputs

Compute and save:

- generated text
- generated token ids
- segment = `full_output`
- mean LID per sample and layer
- aggregate family summaries

### A plots

Create at least:

1. boxplot or violin plot of sample mean LID by family and layer
2. strip/scatter overlay of individual sample mean LID values
3. summary bar chart with mean and bootstrap CI per family

Families on x-axis:

- `constant_repetition`
- `short_cycle_repetition`
- `templatic_repetition`
- `regular_gsm8k_nothink`

---

## 3.2 Experiment B — GSM8K canonical, think vs no_think

Purpose: paired comparison on the same GSM8K problems.

### B.prompt template

Use exactly this visible-answer template for **both** conditions.

Thinking condition uses `enable_thinking=True`.
Non-thinking condition uses `enable_thinking=False`.

User content:

```text
Solve the following grade-school math word problem.

Do not provide detailed step-by-step reasoning.
Return exactly three lines in this format:
Summary: one short sentence describing the key calculation.
Equation: one equation only.
Final answer: one number only.

Problem: {QUESTION}
```

Important:

- Do **not** add "Let's think step by step".
- Do **not** ask for chain-of-thought.
- The visible output format must be the same across modes.

### B.segments to analyze

For `enable_thinking=True`:

- `thinking_segment`: generated text inside `<think> ... </think>`
- `answer_segment`: generated text after `</think>`
- `full_output`

For `enable_thinking=False`:

- `answer_segment`: the whole generated output
- `full_output`

Primary comparison:

- `answer_segment` under `think` vs `answer_segment` under `no_think`

Secondary summaries:

- `thinking_segment` alone for think mode
- `full_output` for both modes

### B.dataset size

Default local run:

- `n_samples = 200` from GSM8K test split
- configurable via CLI

Use deterministic subsampling with a seed.

### B.correctness

Implement answer extraction and correctness evaluation.

For GSM8K labels:

- parse the gold numeric answer from the `answer` field after the final `####`
- normalize commas and whitespace
- compare normalized numeric strings

For model outputs:

- parse the numeric answer from the line beginning with `Final answer:`
- if that fails, attempt a conservative fallback numeric parse from the last line
- store parse status and correctness status

### B.outputs

Store per sample, per mode:

- prompt
- generated text
- generated token ids
- extracted segments
- correctness
- generated token count per segment
- mean LID per selected layer and segment

Also create a paired summary table keyed by GSM8K example id.

### B.plots

Create at least:

1. paired scatter plot of `answer_segment` mean LID: `think` vs `no_think`
2. histogram or KDE-style line plot of paired differences: `LID_think_answer - LID_nothink_answer`
3. boxplot of `thinking_segment`, `think_answer_segment`, `nothink_answer_segment`
4. accuracy bar chart: `think` vs `no_think`
5. token-length comparison plot for answer segments

Plot separately for each selected layer and also provide one aggregated figure across layers.

---

## 3.3 Experiment C — Prompt-structure control on GSM8K

Purpose: test whether prompt structure itself changes hidden-state geometry.

Run on a smaller GSM8K subset.

### C.dataset size

Default local run:

- `n_samples = 100`
- deterministic subset from GSM8K test split

### C.conditions

Prompt variants:

- `canonical`
- `irrelevant_context`
- `repetitive_filler`

Modes:

- `think`
- `no_think`

This yields six conditions per problem.

### C.visible answer format

Use the **same visible output format** as Experiment B.

### C.canonical prompt

```text
Solve the following grade-school math word problem.

Do not provide detailed step-by-step reasoning.
Return exactly three lines in this format:
Summary: one short sentence describing the key calculation.
Equation: one equation only.
Final answer: one number only.

Problem: {QUESTION}
```

### C.irrelevant_context prompt

Use this exact structure:

```text
Solve the following grade-school math word problem.

Do not provide detailed step-by-step reasoning.
Return exactly three lines in this format:
Summary: one short sentence describing the key calculation.
Equation: one equation only.
Final answer: one number only.

Background:
{DISTRACTOR_SNIPPET}

Problem: {QUESTION}
```

Use a fixed distractor library of exactly 10 snippets. Create the library in source control as data or config, not inline hard-coded inside a notebook.

Use these snippets:

1.
`At a library, 14 red chairs were moved to one room and 9 blue chairs to another. The chairs were cleaned on Friday afternoon.`

2.
`Nina packed 6 postcards and 11 stickers for a trip last month. She kept them in a small yellow box near the window.`

3.
`A pet shop sold 8 collars on Monday and 13 bowls on Tuesday. The owner painted the front door green the next day.`

4.
`Owen counted 12 pinecones and 7 smooth stones during a walk. He placed them on a shelf beside two candles.`

5.
`A music room had 5 drums and 16 flutes stored after class. The teacher locked the cabinet before lunch.`

6.
`Tara bought 9 markers and 4 notebooks on Saturday. She wrote her name on each notebook with purple ink.`

7.
`The bakery displayed 15 muffins and 6 pies in the front case. A new sign was taped to the glass door.`

8.
`Leo folded 10 paper stars and 3 paper cranes for decoration. He hung them above a wooden desk.`

9.
`A garden shed contained 11 clay pots and 8 small shovels. Rain started just after the tools were arranged.`

10.
`Mira sorted 7 ribbons and 14 buttons into separate jars. She left the jars on a round table near the lamp.`

Assign distractors deterministically by example index unless the user overrides with randomization.

### C.repetitive_filler prompt

Use this exact structure:

```text
Solve the following grade-school math word problem.

Do not provide detailed step-by-step reasoning.
Return exactly three lines in this format:
Summary: one short sentence describing the key calculation.
Equation: one equation only.
Final answer: one number only.

Prefix:
DATA NOTE TEXT NULL DATA NOTE TEXT NULL
DATA NOTE TEXT NULL DATA NOTE TEXT NULL
DATA NOTE TEXT NULL DATA NOTE TEXT NULL

Problem: {QUESTION}
```

### C.outputs

Same output schema as Experiment B, plus prompt variant metadata.

### C.plots

Create at least:

1. grouped boxplot of `answer_segment` mean LID by prompt variant and mode
2. paired difference plots within each prompt variant: `think - no_think`
3. token-length comparison by prompt variant and mode
4. accuracy comparison by prompt variant and mode

---

## 4. Decoding and generation policy

### 4.1 Main comparison policy

To avoid confounding the main scientific comparison with different decoding profiles, implement **two sampling profiles**:

- `matched_main` (default for experiments B and C)
- `official_recommended`

### 4.2 matched_main

Use the same decoding parameters for both `think` and `no_think`:

- `do_sample=True`
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `max_new_tokens=256`
- `repetition_penalty=1.0`
- `presence_penalty` not required unless the implementation path supports it cleanly

### 4.3 official_recommended

Implement an alternative profile:

For `think`:

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`

For `no_think`:

- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`

### 4.4 Experiment A generation

Default to `matched_main` here too unless you discover extreme compliance issues. Do not introduce unnecessary special cases.

### 4.5 Autoregressive loop requirement

Do **not** rely on a black-box `generate(..., output_hidden_states=True)` workflow if it makes code brittle.

Implement a maintainable custom autoregressive generation loop that:

1. tokenizes the prompt,
2. runs step-by-step generation with KV cache,
3. samples the next token,
4. stores per-step generated token id,
5. stores the hidden state for that generated token for the selected layers,
6. stops on EOS or `max_new_tokens`.

This gives direct control over token-aligned hidden states.

### 4.6 Segment extraction

Implement robust text and token segmentation.

For `think` mode, detect `<think>` and `</think>` in decoded text. Also support token-level segment boundary detection if feasible.

Required behavior:

- if `<think>...</think>` exists, split into `thinking_segment` and `answer_segment`
- if think mode yields an empty think block, allow empty `thinking_segment`
- if formatting is malformed, store a parse status and continue

You must preserve token alignment well enough to compute hidden states for each segment. The cleanest acceptable solution is:

- decode incremental outputs after each new token,
- track when the decoded running text crosses the segment markers,
- map generated token indices to segments.

This is slightly slower but acceptable for 1.7B on a local machine.

---

## 5. Repository structure to create

Create this structure exactly or very close to it:

```text
project_root/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  configs/
    common.yaml
    exp_a.yaml
    exp_b.yaml
    exp_c.yaml
  data/
    raw/
    processed/
    caches/
  outputs/
    exp_a/
    exp_b/
    exp_c/
    plots/
  scripts/
    bootstrap_assets.py
    run_exp_a.py
    run_exp_b.py
    run_exp_c.py
    plot_all.py
  src/
    qwen_lid/
      __init__.py
      config.py
      paths.py
      logging_utils.py
      seeding.py
      device.py
      prompts.py
      distractors.py
      gsm8k.py
      answer_parsing.py
      model_loader.py
      generation.py
      segmentation.py
      hidden_states.py
      lid.py
      stats.py
      plotting.py
      io_utils.py
      schemas.py
      experiments/
        __init__.py
        exp_a.py
        exp_b.py
        exp_c.py
  tests/
    test_lid.py
    test_answer_parsing.py
    test_segmentation.py
    test_prompts.py
```

---

## 6. Bootstrapping step — implement first

Create `scripts/bootstrap_assets.py` first.

Responsibilities:

1. create required directories,
2. download model/tokenizer assets for `Qwen/Qwen3-1.7B`,
3. download/cache `openai/gsm8k`,
4. optionally save a small metadata manifest,
5. print clear success/failure messages.

### 6.1 Model bootstrap details

Use Hugging Face APIs suitable for local download and later reuse.

Acceptable options:

- `transformers` `from_pretrained(...)` once, or
- `huggingface_hub.snapshot_download(...)`

Prefer a solution that keeps later loading simple.

### 6.2 Dataset bootstrap details

Use:

- `datasets.load_dataset("openai/gsm8k", "main")`

Then save to disk under `data/raw/gsm8k/` using `save_to_disk(...)`.

Also keep a helper that can reload from local disk if present, otherwise fetch from Hugging Face.

### 6.3 Bootstrap CLI

Support at least:

```bash
python scripts/bootstrap_assets.py
python scripts/bootstrap_assets.py --model-id Qwen/Qwen3-1.7B --dataset-name openai/gsm8k
```

---

## 7. Shared implementation details

## 7.1 Device selection

Implement:

- prefer `mps` if available
- else CPU

For model dtype:

- use `float16` on MPS if stable
- otherwise use `float32`
- avoid brittle assumptions

Add a small helper that logs device and dtype.

## 7.2 Seeding

Implement a central seed helper that seeds:

- `random`
- `numpy`
- `torch`

## 7.3 GSM8K loading

Implement a helper that:

- loads local saved dataset if available,
- otherwise downloads and caches,
- supports deterministic subsetting by seed,
- returns examples with stable ids.

Stable id scheme: if the dataset has no explicit id, create one like `gsm8k_test_000123` based on split index.

## 7.4 Prompt builders

Create dedicated prompt-builder functions for:

- Experiment A degenerate prompts
- Experiment B canonical prompt
- Experiment C canonical prompt
- Experiment C irrelevant context prompt
- Experiment C repetitive filler prompt

Do not duplicate prompt strings in multiple files.

## 7.5 Model wrapper

Create a model wrapper that exposes:

- `load_model(...)`
- `build_chat_prompt(user_content, enable_thinking)`
- `generate_with_hidden_states(...)`

The generation method must return a structured object with:

- prompt text
- prompt token ids
- generated token ids
- decoded text
- per-layer generated hidden states aligned to generated tokens
- metadata including decoding config and stop reason

---

## 8. LID implementation requirements

## 8.1 Core API

Implement something close to:

- `compute_token_lid(hidden_states: np.ndarray, k: int) -> np.ndarray`
- `compute_mean_lid(hidden_states: np.ndarray, k: int) -> float | None`

Where `hidden_states` has shape `[num_tokens, hidden_dim]`.

## 8.2 Numerical rules

- exclude self from nearest neighbors
- handle duplicate points safely
- if any denominator would be zero or invalid, mark token invalid instead of crashing
- return per-token validity mask if useful

## 8.3 Tests

Include tests for:

- simple synthetic data with known valid output shape
- too-short segments returning invalid / `None`
- duplicated rows not crashing

---

## 9. Statistics to implement

Implement lightweight stats utilities:

- group means
- medians
- bootstrap confidence intervals
- paired differences
- Wilcoxon signed-rank test when applicable

Use `scipy` if needed.

### Experiment A summaries

Per family and layer:

- sample count
- mean sample mean LID
- median sample mean LID
- bootstrap 95% CI

### Experiment B summaries

Per layer:

- count of valid paired samples for `answer_segment`
- mean paired difference: `think_answer - nothink_answer`
- median paired difference
- bootstrap 95% CI
- Wilcoxon p-value
- accuracy per mode

### Experiment C summaries

Per prompt variant and layer:

- same paired-difference summary as B
- accuracy by mode
- answer token length by mode

---

## 10. File formats and artifacts

Use simple, inspectable file formats.

### 10.1 Raw generations

Save per-sample generations as **JSONL**.

At minimum include:

- example id
- experiment name
- mode
- prompt variant
- prompt text
- generated text
- generated token ids
- segment texts
- segment token index ranges
- correctness metadata
- decoding config
- runtime metadata

### 10.2 Hidden states

Do not dump giant unstructured pickle files.

Use one of these:

- compressed `.npz` files per sample, or
- a sharded arrangement under experiment output directories

Store only selected layers, generated-token hidden states, and enough metadata to reattach them to the generation record.

### 10.3 Metrics tables

Save metrics as **CSV** and optionally Parquet.

Suggested tables:

- `sample_metrics.csv`
- `pair_metrics.csv`
- `summary_metrics.csv`

### 10.4 Manifest files

For each experiment run, create a small `run_manifest.json` containing:

- timestamp
- git commit if available
- model id
- device
- dtype
- config
- sample counts
- selected layers
- seed

---

## 11. Plotting requirements

Centralize plotting code in `src/qwen_lid/plotting.py`.

Each experiment script should call its own plotting function after metrics finish.

### 11.1 Style requirements

- use `matplotlib`
- use readable axis labels and titles
- include layer ids in titles or facet labels
- save `.png` and `.pdf`
- use consistent naming

### 11.2 Required output locations

Save experiment-specific plots to:

- `outputs/exp_a/figures/`
- `outputs/exp_b/figures/`
- `outputs/exp_c/figures/`

Also allow `scripts/plot_all.py` to copy or regenerate overview plots under:

- `outputs/plots/`

---

## 12. CLI entry points to create

These files must exist and be runnable:

### 12.1 Bootstrap

```bash
python scripts/bootstrap_assets.py
```

### 12.2 Experiment A

```bash
python scripts/run_exp_a.py
python scripts/run_exp_a.py --n-baseline 30 --k 10 --layers 6 13 20
```

### 12.3 Experiment B

```bash
python scripts/run_exp_b.py
python scripts/run_exp_b.py --n-samples 200 --k 10 --layers 6 13 20 --sampling-profile matched_main
```

### 12.4 Experiment C

```bash
python scripts/run_exp_c.py
python scripts/run_exp_c.py --n-samples 100 --k 10 --layers 6 13 20 --sampling-profile matched_main
```

### 12.5 Replot all

```bash
python scripts/plot_all.py
```

Each script must provide `--help` output with clear descriptions.

---

## 13. Recommended implementation order

Follow this order exactly.

### Step 1
Create the project skeleton and `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`.

### Step 2
Implement `scripts/bootstrap_assets.py` and verify model/dataset download works.

### Step 3
Implement shared utilities:

- paths
- logging
- seeding
- device selection
- config loading

### Step 4
Implement GSM8K loader and answer parsing.

### Step 5
Implement prompt builders and distractor library.

### Step 6
Implement model loading and the custom autoregressive generation loop.

### Step 7
Implement segmentation logic for think / answer segments.

### Step 8
Implement LID computation and unit tests.

### Step 9
Implement Experiment A end-to-end.

### Step 10
Implement Experiment B end-to-end.

### Step 11
Implement Experiment C end-to-end.

### Step 12
Implement plotting and `plot_all.py`.

### Step 13
Write a concise README with setup and run instructions.

Do not start with notebooks. Notebooks are optional and should not be required.

---

## 14. Dependencies

Use a minimal, standard stack.

Required packages likely include:

- `torch`
- `transformers`
- `datasets`
- `huggingface_hub`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `pyyaml`
- `tqdm`
- `pytest`

Keep dependencies conservative. Do not add heavy orchestration frameworks unless truly necessary.

---

## 15. Testing requirements

At minimum, `pytest` should cover:

- LID computation edge cases
- answer parsing from GSM8K labels and model outputs
- segmentation of `<think>...</think>` vs answer content
- prompt-builder output invariants

If practical, add one tiny smoke test for generation utilities that can run without downloading the full model by mocking the generator output.

---

## 16. README requirements

The repository README must include:

1. project purpose
2. local setup on macOS
3. how to bootstrap assets
4. how to run experiments A/B/C
5. where outputs are stored
6. how to rerun only plotting
7. key implementation notes about MPS and CPU fallback

---

## 17. Quality bar

This is research code, but it should still be clean.

Required quality bar:

- modular functions
- clear names
- type hints where reasonable
- docstrings on nontrivial functions
- no giant monolithic script with duplicated logic
- no notebook-only workflow
- no hidden magic constants; keep them in config or constants

---

## 18. Acceptance checklist

Consider the task complete only if all items below are true.

### Core build

- [ ] project skeleton exists
- [ ] `bootstrap_assets.py` downloads model and dataset
- [ ] model loads locally on macOS with MPS or CPU fallback
- [ ] GSM8K loads from local saved copy

### Experiment functionality

- [ ] Experiment A runs from its own script and writes outputs + plots
- [ ] Experiment B runs from its own script and writes outputs + plots
- [ ] Experiment C runs from its own script and writes outputs + plots
- [ ] each experiment can resume or skip already-computed examples when sensible

### Metrics

- [ ] hidden states are extracted for selected layers
- [ ] LID is computed per segment and layer
- [ ] paired summaries are produced for B and C
- [ ] correctness metrics are produced for B and C

### Plotting

- [ ] plots are generated automatically after each experiment
- [ ] `plot_all.py` can regenerate overview plots from saved artifacts

### Maintainability

- [ ] shared utilities are factored into `src/qwen_lid/`
- [ ] tests exist and pass
- [ ] README exists and is usable

---

## 19. Final note

When in doubt, prefer:

- correctness over premature optimization
- inspectable artifacts over clever but opaque storage
- modularity over shortcut scripts
- explicit segment metadata over heuristic guessing hidden inside plotting code

Implement a codebase that another researcher can run locally, inspect, and extend.
