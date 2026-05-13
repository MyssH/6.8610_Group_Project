from __future__ import annotations

import json
import math
import os
import re
import string
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "qwen_lid_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from qwen_lid.lid import compute_token_lid_with_mask
from qwen_lid.stats import bootstrap_ci


EXPERIMENTS = ("exp_a", "exp_b", "exp_c")
SEGMENT_LABELS = {
    ("think", "thinking_segment"): "thinking",
    ("think", "answer_segment"): "think_answer",
    ("no_think", "answer_segment"): "no_think_answer",
    ("no_think", "full_output"): "no_think_full",
    ("think", "full_output"): "think_full",
}
MODEL_COLOR_ORDER = ["1.7B", "4B", "8B", "14B", "32B"]
HIDDEN_DIMS_BY_MODEL_LABEL = {
    "1.7B": 2048,
    "2B": 2048,
    "4B": 2560,
    "8B": 4096,
    "14B": 5120,
    "32B": 5120,
}
FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "with",
}
REASONING_MARKERS = {
    "because",
    "therefore",
    "so",
    "thus",
    "hence",
    "first",
    "second",
    "third",
    "next",
    "then",
    "finally",
    "answer",
    "summary",
    "equation",
    "calculate",
    "total",
    "difference",
}
TOKEN_CATEGORY_CACHE: dict[int, str] = {}


@dataclass(frozen=True)
class ModelRun:
    name: str
    label: str
    root: Path
    sort_key: float


@dataclass(frozen=True)
class MechanismConfig:
    outputs_root: Path
    output_dir_name: str = "mechanism_visualizations"
    per_model_dir_name: str = "figures_mechanism"
    exclude_model_labels: tuple[str, ...] = ()
    exclude_model_names: tuple[str, ...] = ()
    lid_value_normalization: str = "none"
    layers: tuple[int, ...] = (6, 13, 20)
    k: int = 10
    normalize_hidden_states: bool = False
    lid_device: str | None = "auto"
    random_seed: int = 1234
    max_lid_tokens: int | None = 512
    max_shared_samples: int = 3
    max_exp_c_token_examples: int = 80
    boundary_window: int = 80
    normalized_bins: int = 60
    refresh_cache: bool = False
    token_categories: bool = True


def run_mechanism_visualizations(config: MechanismConfig) -> dict[str, Any]:
    sns.set_theme(style="whitegrid", context="talk")
    outputs_root = config.outputs_root.resolve()
    out_root = outputs_root / config.output_dir_name
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    model_runs = discover_model_runs(outputs_root)
    model_runs = filter_model_runs(model_runs, config)
    if not model_runs:
        raise FileNotFoundError(f"No model output folders found under {outputs_root}")

    lid_device = choose_lid_device(config.lid_device)
    print(f"Discovered {len(model_runs)} model folders: {', '.join(m.name for m in model_runs)}")
    print(f"LID recomputation device: {lid_device or 'cpu'}")

    sample_metrics, pair_metrics = load_all_metric_tables(model_runs, config)
    sample_metrics.to_csv(data_root / "all_sample_metrics.csv", index=False)
    pair_metrics.to_csv(data_root / "all_pair_metrics.csv", index=False)

    key_stats = make_mean_comparison_plots(sample_metrics, out_root)
    for exp, df in key_stats.items():
        df.to_csv(data_root / f"{exp}_key_stats.csv", index=False)

    selected_token_rows: list[pd.DataFrame] = []
    plot_per_sample_trajectories(model_runs, config, lid_device, selected_token_rows)
    plot_cross_model_aligned_trajectories(model_runs, config, lid_device, out_root, selected_token_rows)
    if selected_token_rows:
        selected_df = pd.concat(selected_token_rows, ignore_index=True)
        selected_df.to_csv(data_root / "selected_token_trajectories.csv", index=False)
        selected_stats = trajectory_stats_from_token_rows(selected_df)
        selected_stats.to_csv(data_root / "selected_trajectory_stats.csv", index=False)
        selected_token_count = int(len(selected_df))
    else:
        selected_stats = pd.DataFrame()
        selected_token_count = 0

    delta_df, delta_summary = make_exp_c_correctness_and_delta_plots(sample_metrics, pair_metrics, model_runs, out_root, config)
    delta_df.to_csv(data_root / "exp_c_paired_deltas.csv", index=False)
    delta_summary.to_csv(data_root / "exp_c_paired_delta_summary.csv", index=False)

    tokenizer = load_qwen_tokenizer(outputs_root) if config.token_categories else None
    exp_c_common_examples = common_exp_c_kept_examples(model_runs)
    exp_c_accuracy = build_exp_c_accuracy_by_model(model_runs)
    exp_c_accuracy.to_csv(data_root / "exp_c_accuracy_by_model_kept_per_model.csv", index=False)
    plot_exp_c_accuracy_by_model(exp_c_accuracy, out_root)

    exp_c_token_df = build_exp_c_token_dataset(model_runs, config, lid_device, tokenizer, exp_c_common_examples)
    exp_c_token_path = data_root / "exp_c_token_lid_segments_sampled.csv"
    exp_c_token_df.to_csv(exp_c_token_path, index=False)

    exp_c_stats = trajectory_stats_from_token_rows(exp_c_token_df)
    exp_c_stats = add_exp_c_boundary_stats(exp_c_stats, exp_c_token_df)
    exp_c_stats["sample_scope"] = "per_model_kept_independent_plus_common_union"
    exp_c_stats.to_csv(data_root / "exp_c_token_trajectory_stats.csv", index=False)

    if "selected_for_aggregate_common" in exp_c_token_df:
        exp_c_common_token_df = exp_c_token_df[exp_c_token_df["selected_for_aggregate_common"].astype(bool)].copy()
    else:
        exp_c_common_token_df = exp_c_token_df.iloc[0:0].copy()
    exp_c_common_stats = trajectory_stats_from_token_rows(exp_c_common_token_df)
    exp_c_common_stats = add_exp_c_boundary_stats(exp_c_common_stats, exp_c_common_token_df)
    exp_c_common_stats["sample_scope"] = "common_kept_across_included_models"
    exp_c_common_stats.to_csv(data_root / "exp_c_common_kept_token_trajectory_stats.csv", index=False)
    exp_c_correctness_stats = summarize_exp_c_correctness_statistics(exp_c_stats, aggregate_stats=exp_c_common_stats)
    exp_c_correctness_stats.to_csv(data_root / "exp_c_correct_incorrect_trajectory_statistics.csv", index=False)

    plot_exp_c_boundary_anchored(exp_c_token_df, out_root, config)
    plot_exp_c_normalized_trajectories(exp_c_token_df, out_root, config)
    plot_exp_c_heatmaps(exp_c_token_df, exp_c_stats, out_root, config)
    plot_exp_c_trajectory_statistics(exp_c_stats, out_root, aggregate_stats=exp_c_common_stats)
    plot_exp_c_token_category_summaries(exp_c_token_df, out_root)
    plot_accuracy_colored_trajectories(exp_c_token_df, out_root, config)

    write_readme(out_root, model_runs, config, lid_device)
    write_qualitative_notes(out_root, key_stats, delta_summary, exp_c_stats, sample_metrics)

    return {
        "output_root": str(out_root),
        "models": [m.name for m in model_runs],
        "sample_metrics_rows": int(len(sample_metrics)),
        "pair_metrics_rows": int(len(pair_metrics)),
        "exp_c_token_rows": int(len(exp_c_token_df)),
        "selected_token_rows": selected_token_count,
    }


def discover_model_runs(outputs_root: Path) -> list[ModelRun]:
    runs: list[ModelRun] = []
    for path in sorted(outputs_root.iterdir()):
        if not path.is_dir():
            continue
        if not (path.name.startswith("Qwen_Qwen3-") or path.name.startswith("Qwen--Qwen3-")):
            continue
        if all((path / exp / "raw_generations.jsonl").exists() for exp in EXPERIMENTS):
            label = model_label_from_name(path.name)
            runs.append(ModelRun(name=path.name, label=label, root=path, sort_key=model_sort_key(label)))
    if not runs and all((outputs_root / exp).is_dir() for exp in EXPERIMENTS):
        runs.append(ModelRun(name=outputs_root.name, label=outputs_root.name, root=outputs_root, sort_key=0.0))
    return sorted(runs, key=lambda run: run.sort_key)


def filter_model_runs(model_runs: list[ModelRun], config: MechanismConfig) -> list[ModelRun]:
    excluded_labels = {normalize_model_filter(value) for value in config.exclude_model_labels}
    excluded_names = {normalize_model_filter(value) for value in config.exclude_model_names}
    filtered: list[ModelRun] = []
    for run in model_runs:
        label = normalize_model_filter(run.label)
        name = normalize_model_filter(run.name)
        if label in excluded_labels or name in excluded_names:
            continue
        filtered.append(run)
    return filtered


def normalize_model_filter(value: str) -> str:
    return str(value).strip().lower().replace("qwen/qwen3-", "").replace("qwen_qwen3-", "").replace("qwen--qwen3-", "")


def model_label_from_name(name: str) -> str:
    match = re.search(r"Qwen[_-]+Qwen3[-_](.+)$", name)
    if match:
        return match.group(1)
    match = re.search(r"Qwen3[-_](.+)$", name)
    return match.group(1) if match else name


def model_sort_key(label: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)B", label)
    if match:
        return float(match.group(1))
    return math.inf


def choose_lid_device(requested: str | None) -> str | None:
    if requested in (None, "", "cpu"):
        return None
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return None
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_all_metric_tables(model_runs: list[ModelRun], config: MechanismConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_tables: list[pd.DataFrame] = []
    pair_tables: list[pd.DataFrame] = []
    for run in model_runs:
        for exp in EXPERIMENTS:
            exp_dir = run.root / exp
            sample_path = exp_dir / "sample_metrics.csv"
            if sample_path.exists():
                df = pd.read_csv(sample_path)
                df.insert(0, "model", run.name)
                df.insert(1, "model_label", run.label)
                df["model_sort_key"] = run.sort_key
                df["hidden_dim"] = hidden_dim_for_model_label(run.label)
                if "mean_lid" in df:
                    df["raw_mean_lid"] = df["mean_lid"]
                    df["mean_lid"] = normalize_lid_series(df["mean_lid"], run.label, config)
                df["lid_value_normalization"] = config.lid_value_normalization
                df["condition"] = df.apply(lambda row: condition_from_metric_row(exp, row), axis=1)
                df["segment_label"] = df.apply(lambda row: segment_label(row.get("mode"), row.get("segment")), axis=1)
                sample_tables.append(df)
            pair_path = exp_dir / "pair_metrics.csv"
            if pair_path.exists():
                pair = pd.read_csv(pair_path)
                pair.insert(0, "model", run.name)
                pair.insert(1, "model_label", run.label)
                pair["model_sort_key"] = run.sort_key
                pair["hidden_dim"] = hidden_dim_for_model_label(run.label)
                for col in ["think_answer_lid", "no_think_answer_lid", "answer_lid_diff"]:
                    if col in pair:
                        pair[f"raw_{col}"] = pair[col]
                        pair[col] = normalize_lid_series(pair[col], run.label, config)
                pair["lid_value_normalization"] = config.lid_value_normalization
                pair_tables.append(pair)
    sample = pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame()
    pair = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    return sample, pair


def hidden_dim_for_model_label(label: str) -> int:
    normalized = str(label).strip()
    if normalized in HIDDEN_DIMS_BY_MODEL_LABEL:
        return HIDDEN_DIMS_BY_MODEL_LABEL[normalized]
    compact = normalized.lower().replace("qwen/qwen3-", "").replace("qwen_qwen3-", "").replace("qwen--qwen3-", "")
    for known, hidden_dim in HIDDEN_DIMS_BY_MODEL_LABEL.items():
        if compact == known.lower():
            return hidden_dim
    raise KeyError(f"No hidden dimension configured for model label {label!r}")


def normalize_lid_series(values: pd.Series, model_label: str, config: MechanismConfig) -> pd.Series:
    if config.lid_value_normalization == "none":
        return values
    if config.lid_value_normalization == "hidden_dim":
        return values / hidden_dim_for_model_label(model_label)
    raise ValueError(f"Unknown LID value normalization: {config.lid_value_normalization}")


def normalize_lid_array(values: np.ndarray, model_label: str, config: MechanismConfig) -> np.ndarray:
    if config.lid_value_normalization == "none":
        return values
    if config.lid_value_normalization == "hidden_dim":
        return values / float(hidden_dim_for_model_label(model_label))
    raise ValueError(f"Unknown LID value normalization: {config.lid_value_normalization}")


def condition_from_metric_row(exp: str, row: pd.Series) -> str:
    if exp == "exp_a":
        value = row.get("family")
    elif exp == "exp_b":
        value = row.get("prompt_variant")
    elif exp == "exp_c":
        value = row.get("mode")
    else:
        value = None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    return str(value)


def condition_from_record(record: dict[str, Any], exp: str) -> str:
    if exp == "exp_a":
        return str(record.get("family") or "unknown")
    if exp == "exp_b":
        return str(record.get("prompt_variant") or "unknown")
    if exp == "exp_c":
        return str(record.get("mode") or "unknown")
    return "unknown"


def segment_label(mode: Any, segment: Any) -> str:
    mode_str = str(mode) if mode is not None and not pd.isna(mode) else ""
    segment_str = str(segment) if segment is not None and not pd.isna(segment) else ""
    return SEGMENT_LABELS.get((mode_str, segment_str), segment_str or "unknown")


def valid_metric_rows(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    if sample_metrics.empty:
        return sample_metrics
    df = sample_metrics.copy()
    if "valid" in df:
        df = df[df["valid"].astype(bool)]
    df = df[np.isfinite(df["mean_lid"])]
    df = df.sort_values(["model_sort_key", "model_label", "layer"])
    return df


def make_mean_comparison_plots(sample_metrics: pd.DataFrame, out_root: Path) -> dict[str, pd.DataFrame]:
    valid = valid_metric_rows(sample_metrics)
    summaries: dict[str, pd.DataFrame] = {}
    if valid.empty:
        return summaries

    exp_a = valid[(valid["experiment"] == "exp_a") & (valid["segment"] == "full_output")]
    summaries["exp_a"] = group_summary(exp_a, ["model", "model_label", "model_sort_key", "family", "layer"])
    if not exp_a.empty:
        plot_metric_family(
            exp_a,
            out_root / "exp_a" / "mean_lid_comparisons",
            condition_col="family",
            title_prefix="Exp A mean full-output LID",
            heatmap_col="family",
        )

    exp_b = valid[(valid["experiment"] == "exp_b") & (valid["segment"] == "answer_segment")]
    summaries["exp_b"] = group_summary(exp_b, ["model", "model_label", "model_sort_key", "prompt_variant", "layer"])
    if not exp_b.empty:
        plot_metric_family(
            exp_b,
            out_root / "exp_b" / "mean_lid_comparisons",
            condition_col="prompt_variant",
            title_prefix="Exp B mean answer LID",
            heatmap_col="prompt_variant",
        )

    exp_c = valid[
        (valid["experiment"] == "exp_c")
        & (
            ((valid["mode"] == "think") & (valid["segment"].isin(["thinking_segment", "answer_segment"])))
            | ((valid["mode"] == "no_think") & (valid["segment"] == "answer_segment"))
        )
    ].copy()
    exp_c["phase"] = exp_c.apply(lambda row: segment_label(row["mode"], row["segment"]), axis=1)
    summaries["exp_c"] = group_summary(exp_c, ["model", "model_label", "model_sort_key", "phase", "layer"])
    if not exp_c.empty:
        plot_metric_family(
            exp_c,
            out_root / "exp_c" / "mean_lid_comparisons",
            condition_col="phase",
            title_prefix="Exp C mean segment LID",
            heatmap_col="phase",
        )
    return summaries


def group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["n", "mean", "median", "ci_low", "ci_high"])
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["mean_lid"].dropna().to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(values, seed=1234)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": int(values.size),
                "mean": float(np.mean(values)) if values.size else np.nan,
                "median": float(np.median(values)) if values.size else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values([col for col in group_cols if col in ["model_sort_key", "layer"]] or group_cols)


def plot_metric_family(
    df: pd.DataFrame,
    out_dir: Path,
    condition_col: str,
    title_prefix: str,
    heatmap_col: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = model_order(df)
    condition_order = sorted(str(v) for v in df[condition_col].dropna().unique())

    for layer in sorted(df["layer"].dropna().unique()):
        layer_df = df[df["layer"] == layer].copy()
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.pointplot(
            data=layer_df,
            x="model_label",
            y="mean_lid",
            hue=condition_col,
            order=order,
            hue_order=condition_order,
            errorbar=("ci", 95),
            dodge=0.35,
            markers="o",
            ax=ax,
        )
        ax.set_title(f"{title_prefix}, layer {int(layer)}")
        ax.set_xlabel("model size")
        ax.set_ylabel(f"mean {lid_value_label(layer_df)}")
        ax.legend(title=condition_col, bbox_to_anchor=(1.02, 1), loc="upper left")
        save_figure(fig, out_dir / f"point_ci_layer_{int(layer)}")

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.violinplot(
            data=layer_df,
            x="model_label",
            y="mean_lid",
            hue=condition_col,
            order=order,
            hue_order=condition_order,
            inner="quartile",
            cut=0,
            ax=ax,
        )
        ax.set_title(f"{title_prefix} distributions, layer {int(layer)}")
        ax.set_xlabel("model size")
        ax.set_ylabel(f"mean {lid_value_label(layer_df)}")
        ax.legend(title=condition_col, bbox_to_anchor=(1.02, 1), loc="upper left")
        save_figure(fig, out_dir / f"violin_layer_{int(layer)}")

        pivot = (
            layer_df.groupby(["model_label", heatmap_col], observed=True)["mean_lid"]
            .mean()
            .unstack(heatmap_col)
            .reindex(order)
        )
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(pivot.columns)), 5))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
            ax.set_title(f"{title_prefix} heatmap, layer {int(layer)}")
            ax.set_xlabel(heatmap_col)
            ax.set_ylabel("model size")
            save_figure(fig, out_dir / f"heatmap_layer_{int(layer)}")


def model_order(df: pd.DataFrame) -> list[str]:
    order_df = df[["model_label", "model_sort_key"]].drop_duplicates().sort_values("model_sort_key")
    return order_df["model_label"].tolist()


def lid_value_label(df: pd.DataFrame | None = None, config: MechanismConfig | None = None) -> str:
    normalization = config.lid_value_normalization if config is not None else "none"
    if df is not None and "lid_value_normalization" in df and df["lid_value_normalization"].notna().any():
        values = df["lid_value_normalization"].dropna().astype(str).unique().tolist()
        normalization = values[0] if len(values) == 1 else "mixed"
    return "LID / hidden dim" if normalization == "hidden_dim" else "LID"


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(f"{path_without_suffix}.png"), dpi=180, bbox_inches="tight")
    fig.savefig(Path(f"{path_without_suffix}.pdf"), bbox_inches="tight")
    plt.close(fig)


def safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "unknown"


def exp_c_kept_examples(exp_dir: Path) -> set[str]:
    path = exp_dir / "kept_examples.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "keep" in df:
            df = df[df["keep"].astype(bool)]
        return set(df["example_id"].astype(str))
    filter_path = exp_dir / "thinking_segment_filter.csv"
    if filter_path.exists():
        df = pd.read_csv(filter_path)
        if "keep" in df:
            df = df[df["keep"].astype(bool)]
        return set(df["example_id"].astype(str))
    return set()


def common_exp_c_kept_examples(model_runs: list[ModelRun]) -> set[str]:
    common: set[str] | None = None
    for run in model_runs:
        kept = exp_c_kept_examples(run.root / "exp_c")
        common = kept if common is None else common & kept
    return common or set()


def build_exp_c_accuracy_by_model(model_runs: list[ModelRun]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in model_runs:
        records = filtered_records(run, "exp_c")
        by_mode: dict[str, list[bool]] = {}
        for record in records:
            mode = str(record.get("mode") or "unknown")
            correct = correctness_value(record)
            if correct is None:
                continue
            by_mode.setdefault(mode, []).append(bool(correct))
        for mode, values in sorted(by_mode.items()):
            array = np.asarray(values, dtype=float)
            ci_low, ci_high = bootstrap_ci(array, seed=1234)
            rows.append(
                {
                    "model": run.name,
                    "model_label": run.label,
                    "model_sort_key": run.sort_key,
                    "hidden_dim": hidden_dim_for_model_label(run.label),
                    "mode": mode,
                    "n": int(array.size),
                    "accuracy": float(array.mean()) if array.size else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "sample_scope": "kept_for_each_model_independently",
                }
            )
    return pd.DataFrame(rows).sort_values(["model_sort_key", "mode"]) if rows else pd.DataFrame()


def plot_exp_c_accuracy_by_model(accuracy: pd.DataFrame, out_root: Path) -> None:
    if accuracy.empty:
        return
    out_dir = out_root / "exp_c" / "accuracy_by_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    order = model_order(accuracy)
    sns.pointplot(
        data=accuracy,
        x="model_label",
        y="accuracy",
        hue="mode",
        order=order,
        errorbar=None,
        markers="o",
        dodge=0.2,
        ax=ax,
    )
    for _, row in accuracy.iterrows():
        x = order.index(row["model_label"])
        if row["mode"] == "think":
            x += 0.08
        elif row["mode"] == "no_think":
            x -= 0.08
        ax.vlines(x, row["ci_low"], row["ci_high"], color="black", alpha=0.35, linewidth=1)
    ax.set_title("Exp C accuracy by model size, using each model's kept examples")
    ax.set_xlabel("model size")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend(title="mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, out_dir / "exp_c_accuracy_by_model_kept_per_model")


def filtered_records(run: ModelRun, exp: str) -> list[dict[str, Any]]:
    records = load_jsonl(run.root / exp / "raw_generations.jsonl")
    if exp != "exp_c":
        return records
    kept = exp_c_kept_examples(run.root / exp)
    if not kept:
        return records
    return [record for record in records if str(record.get("example_id")) in kept]


def plot_per_sample_trajectories(
    model_runs: list[ModelRun],
    config: MechanismConfig,
    lid_device: str | None,
    selected_token_rows: list[pd.DataFrame],
) -> None:
    rng = np.random.default_rng(config.random_seed)
    for run in model_runs:
        for exp in EXPERIMENTS:
            records = filtered_records(run, exp)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                grouped.setdefault(condition_from_record(record, exp), []).append(record)
            for condition, condition_records in sorted(grouped.items()):
                chosen = condition_records[int(rng.integers(0, len(condition_records)))]
                out_dir = run.root / exp / config.per_model_dir_name / "per_sample_trajectories"
                title = f"{run.label} {exp} {condition} {chosen.get('example_id')}"
                base = out_dir / f"{safe_name(condition)}__{safe_name(chosen.get('example_id'))}"
                rows = plot_single_record_trajectory(
                    run,
                    exp,
                    chosen,
                    config,
                    lid_device,
                    title,
                    base,
                    scope="full_output",
                    source="per_sample",
                )
                if rows is not None:
                    selected_token_rows.append(rows)


def plot_cross_model_aligned_trajectories(
    model_runs: list[ModelRun],
    config: MechanismConfig,
    lid_device: str | None,
    out_root: Path,
    selected_token_rows: list[pd.DataFrame],
) -> None:
    for exp in EXPERIMENTS:
        groups = select_shared_record_groups(model_runs, exp, config.max_shared_samples)
        for group in groups:
            rows_by_model: dict[str, pd.DataFrame] = {}
            for run, record in group["records"]:
                token_rows = record_full_lid_rows(run, exp, record, config, lid_device, source="cross_model")
                selected_token_rows.append(token_rows)
                rows_by_model[run.label] = token_rows
            for layer in config.layers:
                fig, ax = plt.subplots(figsize=(12, 6))
                for run in model_runs:
                    df = rows_by_model.get(run.label)
                    if df is None:
                        continue
                    layer_df = df[(df["layer"] == layer) & np.isfinite(df["lid"])]
                    if layer_df.empty:
                        continue
                    ax.plot(
                        layer_df["token_index"],
                        layer_df["lid"],
                        label=run.label,
                        linewidth=1.5,
                        alpha=0.9,
                    )
                ax.set_title(
                    f"{exp} shared trajectory, {group['condition']}, {group['example_id']}, layer {layer}"
                )
                ax.set_xlabel("generated token index")
                ax.set_ylabel(f"token {lid_value_label(layer_df)}")
                ax.legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left")
                save_figure(
                    fig,
                    out_root
                    / exp
                    / "aligned_trajectories"
                    / f"layer_{layer}__{safe_name(group['condition'])}__{safe_name(group['example_id'])}",
                )


def select_shared_record_groups(
    model_runs: list[ModelRun], exp: str, max_shared_samples: int
) -> list[dict[str, Any]]:
    per_model: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    condition_for_key: dict[tuple[str, str], str] = {}
    for run in model_runs:
        mapping: dict[tuple[str, str], dict[str, Any]] = {}
        for record in filtered_records(run, exp):
            condition = condition_from_record(record, exp)
            key = shared_key(record, exp, condition)
            if key is None:
                continue
            mapping[key] = record
            condition_for_key[key] = condition
        per_model[run.name] = mapping

    common_keys: set[tuple[str, str]] | None = None
    for mapping in per_model.values():
        keys = set(mapping)
        common_keys = keys if common_keys is None else common_keys & keys
    if not common_keys:
        return []

    by_condition: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(common_keys):
        by_condition.setdefault(condition_for_key.get(key, key[0]), []).append(key)

    groups: list[dict[str, Any]] = []
    for condition, keys in sorted(by_condition.items()):
        for key in keys[:max_shared_samples]:
            records = [(run, per_model[run.name][key]) for run in model_runs]
            groups.append(
                {
                    "condition": condition,
                    "example_id": key[1],
                    "records": records,
                }
            )
    return groups


def shared_key(record: dict[str, Any], exp: str, condition: str) -> tuple[str, str] | None:
    example_id = str(record.get("example_id") or "")
    if not example_id:
        return None
    if exp == "exp_c":
        return (str(record.get("mode") or condition), example_id)
    return (condition, example_id)


def plot_single_record_trajectory(
    run: ModelRun,
    exp: str,
    record: dict[str, Any],
    config: MechanismConfig,
    lid_device: str | None,
    title: str,
    base: Path,
    scope: str,
    source: str,
) -> pd.DataFrame | None:
    token_rows = record_full_lid_rows(run, exp, record, config, lid_device, source=source)
    fig, ax = plt.subplots(figsize=(12, 6))
    for layer in config.layers:
        layer_df = token_rows[(token_rows["layer"] == layer) & np.isfinite(token_rows["lid"])]
        if layer_df.empty:
            continue
        ax.plot(layer_df["token_index"], layer_df["lid"], label=f"layer {layer}", linewidth=1.4)
    add_segment_markers(ax, record)
    ax.set_title(title)
    ax.set_xlabel("generated token index")
    ax.set_ylabel(f"token {lid_value_label(token_rows)}")
    ax.legend(title="layer", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, base)
    return token_rows


def add_segment_markers(ax: plt.Axes, record: dict[str, Any]) -> None:
    segments = record.get("segments") or {}
    thinking = segments.get("thinking_segment")
    answer = segments.get("answer_segment")
    if thinking:
        start = int(thinking.get("token_start", 0))
        end = int(thinking.get("token_end", start))
        ax.axvspan(start, end, color="#8ecae6", alpha=0.12, label="thinking span")
        ax.axvline(end, color="#277da1", linestyle="--", linewidth=1, alpha=0.7)
    if answer:
        start = int(answer.get("token_start", 0))
        end = int(answer.get("token_end", start))
        if start > 0:
            ax.axvspan(start, end, color="#ffb703", alpha=0.10, label="answer span")
            ax.axvline(start, color="#fb8500", linestyle=":", linewidth=1.2, alpha=0.9)


def record_full_lid_rows(
    run: ModelRun,
    exp: str,
    record: dict[str, Any],
    config: MechanismConfig,
    lid_device: str | None,
    source: str,
) -> pd.DataFrame:
    return record_segment_lid_rows(
        run,
        exp,
        record,
        segment_name="full_output",
        config=config,
        lid_device=lid_device,
        source=source,
        tokenizer=None,
    )


def record_segment_lid_rows(
    run: ModelRun,
    exp: str,
    record: dict[str, Any],
    segment_name: str,
    config: MechanismConfig,
    lid_device: str | None,
    source: str,
    tokenizer: Any | None,
) -> pd.DataFrame:
    segment = (record.get("segments") or {}).get(segment_name)
    if segment is None:
        return pd.DataFrame()
    start = int(segment.get("token_start", 0))
    end = int(segment.get("token_end", len(record.get("generated_token_ids", []))))
    token_ids = record.get("generated_token_ids", [])
    rows: list[pd.DataFrame] = []
    for layer in config.layers:
        token_indices, values, valid_mask = load_or_compute_lid_values(
            run,
            exp,
            record,
            layer,
            start,
            end,
            segment_name,
            config,
            lid_device,
        )
        raw_values = values
        values = normalize_lid_array(raw_values, run.label, config)
        n = max(end - start, 1)
        rel = token_indices - start
        data = pd.DataFrame(
            {
                "model": run.name,
                "model_label": run.label,
                "model_sort_key": run.sort_key,
                "hidden_dim": hidden_dim_for_model_label(run.label),
                "lid_value_normalization": config.lid_value_normalization,
                "experiment": exp,
                "condition_id": record.get("condition_id"),
                "example_id": record.get("example_id"),
                "condition": condition_from_record(record, exp),
                "mode": record.get("mode"),
                "prompt_variant": record.get("prompt_variant"),
                "family": record.get("family"),
                "correct": correctness_value(record),
                "segment": segment_name,
                "segment_label": segment_label(record.get("mode"), segment_name),
                "segment_token_count": end - start,
                "layer": layer,
                "token_index": token_indices,
                "token_offset": rel,
                "normalized_pos": rel / max(n - 1, 1),
                "lid": values,
                "raw_lid": raw_values,
                "valid_lid": valid_mask,
                "source": source,
            }
        )
        if token_ids:
            ids = [int(token_ids[int(i)]) if 0 <= int(i) < len(token_ids) else -1 for i in token_indices]
            data["token_id"] = ids
            if tokenizer is not None:
                data["token_category"] = [category_for_token(tokenizer, token_id) for token_id in ids]
            else:
                data["token_category"] = "not_decoded"
        rows.append(data)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def correctness_value(record: dict[str, Any]) -> bool | None:
    correctness = record.get("correctness") or {}
    value = correctness.get("correct")
    if value is None:
        return None
    return bool(value)


def resolve_hidden_state_path(run: ModelRun, exp: str, record: dict[str, Any]) -> Path:
    raw = record.get("hidden_state_path")
    candidates: list[Path] = []
    if raw:
        path = Path(str(raw))
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(Path.cwd() / path)
        candidates.append(run.root / exp / "hidden_states" / path.name)
    condition_name = f"{record.get('condition_id')}.npz"
    candidates.append(run.root / exp / "hidden_states" / condition_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve hidden states for {run.name}/{exp}/{record.get('condition_id')}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def load_or_compute_lid_values(
    run: ModelRun,
    exp: str,
    record: dict[str, Any],
    layer: int,
    start: int,
    end: int,
    scope: str,
    config: MechanismConfig,
    lid_device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_dir = run.root / exp / config.per_model_dir_name / "cache" / "token_lid"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"{safe_name(record.get('condition_id'))}__{safe_name(scope)}__layer_{layer}"
        f"__k{config.k}__norm{int(config.normalize_hidden_states)}__max{config.max_lid_tokens or 'all'}.npz"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists() and not config.refresh_cache:
        with np.load(cache_path, allow_pickle=False) as loaded:
            return loaded["token_indices"], loaded["values"], loaded["valid_mask"].astype(bool)

    hidden_path = resolve_hidden_state_path(run, exp, record)
    with np.load(hidden_path, allow_pickle=False) as loaded:
        key = f"layer_{layer}"
        if key not in loaded.files:
            raise KeyError(f"{hidden_path} does not contain {key}")
        hidden = loaded[key][start:end].astype(np.float32, copy=False)

    local_indices = select_lid_indices(hidden.shape[0], config.max_lid_tokens)
    selected_hidden = hidden[local_indices]
    result = compute_token_lid_with_mask(
        selected_hidden,
        k=config.k,
        normalize=config.normalize_hidden_states,
        device=lid_device,
    )
    token_indices = local_indices.astype(np.int32) + start
    values = result.values.astype(np.float32, copy=False)
    valid_mask = result.valid_mask.astype(bool, copy=False)
    np.savez_compressed(
        cache_path,
        token_indices=token_indices,
        values=values,
        valid_mask=valid_mask,
        start=np.asarray([start], dtype=np.int32),
        end=np.asarray([end], dtype=np.int32),
    )
    return token_indices, values, valid_mask


def select_lid_indices(n_tokens: int, max_lid_tokens: int | None) -> np.ndarray:
    if n_tokens <= 0:
        return np.asarray([], dtype=np.int32)
    if max_lid_tokens is None or n_tokens <= max_lid_tokens:
        return np.arange(n_tokens, dtype=np.int32)
    return np.unique(np.round(np.linspace(0, n_tokens - 1, max_lid_tokens)).astype(np.int32))


def make_exp_c_correctness_and_delta_plots(
    sample_metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    model_runs: list[ModelRun],
    out_root: Path,
    config: MechanismConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_c = valid_metric_rows(sample_metrics)
    exp_c = exp_c[exp_c["experiment"] == "exp_c"].copy()
    if exp_c.empty:
        return pd.DataFrame(), pd.DataFrame()
    exp_c["phase"] = exp_c.apply(lambda row: segment_label(row["mode"], row["segment"]), axis=1)
    phase_df = exp_c[exp_c["phase"].isin(["thinking", "think_answer", "no_think_answer"])].copy()
    phase_df["correct_label"] = phase_df["correct"].map({True: "correct", False: "wrong"}).fillna("unknown")

    for run in model_runs:
        run_df = phase_df[phase_df["model"] == run.name]
        out_dir = run.root / "exp_c" / config.per_model_dir_name / "correctness_stratified"
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in sorted(run_df["layer"].dropna().unique()):
            layer_df = run_df[run_df["layer"] == layer]
            if layer_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(
                data=layer_df,
                x="phase",
                y="mean_lid",
                hue="correct_label",
                order=["thinking", "think_answer", "no_think_answer"],
                hue_order=["correct", "wrong", "unknown"],
                cut=0,
                inner="quartile",
                ax=ax,
            )
            ax.set_title(f"{run.label} Exp C correctness-stratified LID, layer {int(layer)}")
            ax.set_xlabel("phase")
            ax.set_ylabel(f"mean {lid_value_label(layer_df)}")
            ax.tick_params(axis="x", rotation=20)
            save_figure(fig, out_dir / f"violin_correctness_layer_{int(layer)}")

    delta_df = build_exp_c_delta_table(exp_c)
    delta_summary = summarize_delta_table(delta_df)
    if not delta_df.empty:
        out_dir = out_root / "exp_c" / "delta_distributions"
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in sorted(delta_df["layer"].dropna().unique()):
            layer_df = delta_df[delta_df["layer"] == layer]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(
                data=layer_df,
                x="model_label",
                y="delta",
                hue="delta_name",
                order=model_order(layer_df),
                cut=0,
                inner="quartile",
                ax=ax,
            )
            ax.axhline(0, color="black", linewidth=1, linestyle="--")
            ax.set_title(f"Exp C paired delta distributions, layer {int(layer)}")
            ax.set_xlabel("model size")
            ax.set_ylabel(f"{lid_value_label(layer_df)} delta")
            ax.legend(title="delta", bbox_to_anchor=(1.02, 1), loc="upper left")
            save_figure(fig, out_dir / f"violin_deltas_layer_{int(layer)}")

            fig, ax = plt.subplots(figsize=(12, 6))
            for (label, delta_name), group in layer_df.groupby(["model_label", "delta_name"], dropna=False):
                values = np.sort(group["delta"].dropna().to_numpy(dtype=float))
                if values.size == 0:
                    continue
                y = np.arange(1, values.size + 1) / values.size
                ax.step(values, y, where="post", linewidth=1.2, alpha=0.75, label=f"{label}: {delta_name}")
            ax.axvline(0, color="black", linewidth=1, linestyle="--")
            ax.set_title(f"Exp C paired delta ECDFs, layer {int(layer)}")
            ax.set_xlabel(f"{lid_value_label(layer_df)} delta")
            ax.set_ylabel("ECDF")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
            save_figure(fig, out_dir / f"ecdf_deltas_layer_{int(layer)}")

            g = sns.FacetGrid(
                layer_df,
                row="delta_name",
                col="model_label",
                col_order=model_order(layer_df),
                sharex=False,
                sharey=False,
                height=2.3,
                aspect=1.2,
            )
            g.map_dataframe(sns.histplot, x="delta", bins=20, kde=True)
            for ax in g.axes.flat:
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            g.fig.suptitle(f"Exp C paired delta histograms, layer {int(layer)}", y=1.03)
            save_facet(g, out_dir / f"hist_deltas_layer_{int(layer)}")
    return delta_df, delta_summary


def save_facet(grid: sns.FacetGrid, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(Path(f"{path_without_suffix}.png"), dpi=180, bbox_inches="tight")
    grid.fig.savefig(Path(f"{path_without_suffix}.pdf"), bbox_inches="tight")
    plt.close(grid.fig)


def build_exp_c_delta_table(exp_c: pd.DataFrame) -> pd.DataFrame:
    subset = exp_c[exp_c["segment"].isin(["thinking_segment", "answer_segment"])].copy()
    if subset.empty:
        return pd.DataFrame()
    pivot = subset.pivot_table(
        index=["model", "model_label", "model_sort_key", "example_id", "layer"],
        columns=["mode", "segment"],
        values="mean_lid",
        aggfunc="first",
    )
    correctness = (
        subset.pivot_table(
            index=["model", "model_label", "model_sort_key", "example_id", "layer"],
            columns=["mode"],
            values="correct",
            aggfunc="first",
        )
        if "correct" in subset
        else pd.DataFrame(index=pivot.index)
    )
    rows: list[dict[str, Any]] = []
    for idx, row in pivot.iterrows():
        model, label, sort_key, example_id, layer = idx
        values = {
            "thinking": get_pivot_value(row, ("think", "thinking_segment")),
            "think_answer": get_pivot_value(row, ("think", "answer_segment")),
            "no_think_answer": get_pivot_value(row, ("no_think", "answer_segment")),
        }
        deltas = {
            "think_answer_minus_no_think_answer": values["think_answer"] - values["no_think_answer"],
            "thinking_minus_think_answer": values["thinking"] - values["think_answer"],
            "thinking_minus_no_think_answer": values["thinking"] - values["no_think_answer"],
        }
        corr_row = correctness.loc[idx] if idx in correctness.index else pd.Series(dtype=object)
        for name, delta in deltas.items():
            if pd.isna(delta):
                continue
            rows.append(
                {
                    "model": model,
                    "model_label": label,
                    "model_sort_key": sort_key,
                    "example_id": example_id,
                    "layer": layer,
                    "delta_name": name,
                    "delta": float(delta),
                    "think_correct": bool(corr_row.get("think")) if "think" in corr_row else None,
                    "no_think_correct": bool(corr_row.get("no_think")) if "no_think" in corr_row else None,
                }
            )
    return pd.DataFrame(rows)


def get_pivot_value(row: pd.Series, key: tuple[str, str]) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def summarize_delta_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if delta_df.empty:
        return pd.DataFrame()
    group_cols = ["model", "model_label", "model_sort_key", "layer", "delta_name"]
    for keys, group in delta_df.groupby(group_cols, dropna=False):
        values = group["delta"].dropna().to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(values, seed=1234)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": int(values.size),
                "mean": float(np.mean(values)) if values.size else np.nan,
                "median": float(np.median(values)) if values.size else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "fraction_below_zero": float(np.mean(values < 0)) if values.size else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_sort_key", "layer", "delta_name"])


def load_qwen_tokenizer(outputs_root: Path) -> Any | None:
    candidates = [
        Path("data/caches/models/Qwen--Qwen3-1.7B"),
        outputs_root / "Qwen--Qwen3-1.7B",
        outputs_root / "Qwen_Qwen3-1.7B",
        Path("Qwen/Qwen3-1.7B"),
    ]
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    for candidate in candidates:
        try:
            if candidate.exists() or "/" in str(candidate):
                return AutoTokenizer.from_pretrained(str(candidate), local_files_only=True)
        except Exception:
            continue
    return None


def decode_one_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception:
        return ""


def category_for_token(tokenizer: Any, token_id: int) -> str:
    if token_id not in TOKEN_CATEGORY_CACHE:
        text = decode_one_token(tokenizer, token_id)
        TOKEN_CATEGORY_CACHE[token_id] = categorize_token(text, token_id)
    return TOKEN_CATEGORY_CACHE[token_id]


def categorize_token(text: str, token_id: int) -> str:
    stripped = text.strip()
    lower = stripped.lower()
    if token_id in {151644, 151645, 151667, 151668} or "<|" in text or "think" in lower and "<" in text:
        return "special_delimiter"
    if not stripped or "\n" in text or stripped in {"\\n", "Ċ"}:
        return "newline_formatting"
    math_chars = set("0123456789+-*/=<>×÷^%$.,")
    if any(ch.isdigit() for ch in stripped) or (set(stripped) <= math_chars and any(ch in math_chars for ch in stripped)):
        return "digits_math_symbols"
    punctuation_chars = set(string.punctuation) | {"“", "”", "‘", "’", "…", "–", "—", "×", "÷"}
    if set(stripped) <= punctuation_chars:
        return "punctuation"
    if lower in FUNCTION_WORDS or (len(lower) <= 3 and lower.isalpha()):
        return "short_function_word"
    if lower.rstrip(":") in REASONING_MARKERS:
        return "reasoning_discourse_marker"
    return "content_word"


def build_exp_c_token_dataset(
    model_runs: list[ModelRun],
    config: MechanismConfig,
    lid_device: str | None,
    tokenizer: Any | None,
    common_kept_examples: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(config.random_seed)
    common_kept_examples = common_kept_examples or set()
    common_candidates = sorted(common_kept_examples)
    if config.max_exp_c_token_examples and len(common_candidates) > config.max_exp_c_token_examples:
        aggregate_common_examples = set(
            rng.choice(common_candidates, size=config.max_exp_c_token_examples, replace=False).tolist()
        )
    else:
        aggregate_common_examples = set(common_candidates)
    for run in model_runs:
        exp = "exp_c"
        records = filtered_records(run, exp)
        by_key = {(str(record.get("example_id")), str(record.get("mode"))): record for record in records}
        kept_examples = sorted({example for example, mode in by_key if mode == "think" and (example, "no_think") in by_key})
        wrong_examples = sorted(
            example
            for example in kept_examples
            if correctness_value(by_key[(example, "think")]) is False
        )
        correct_examples = sorted(set(kept_examples) - set(wrong_examples))
        cap = config.max_exp_c_token_examples
        if cap and len(kept_examples) > cap:
            keep_wrong = set(wrong_examples)
            remaining = max(cap - len(keep_wrong), 0)
            if remaining and len(correct_examples) > remaining:
                keep_correct = set(
                    rng.choice(correct_examples, size=remaining, replace=False).tolist()
                )
            else:
                keep_correct = set(correct_examples)
            per_model_selected = keep_wrong | keep_correct
        else:
            per_model_selected = set(kept_examples)
        selected = sorted(per_model_selected | aggregate_common_examples)
        print(
            f"{run.label} Exp C token-level segment sample: {len(selected)} examples "
            f"({len(per_model_selected)} per-model [keeps all {len(wrong_examples)} wrong], "
            f"{len(aggregate_common_examples)} common-kept aggregate)"
        )
        for example_id in selected:
            for mode, segment_names in {
                "think": ["thinking_segment", "answer_segment"],
                "no_think": ["answer_segment"],
            }.items():
                record = by_key.get((example_id, mode))
                if record is None:
                    continue
                for segment_name in segment_names:
                    segment_rows = record_segment_lid_rows(
                        run,
                        exp,
                        record,
                        segment_name=segment_name,
                        config=config,
                        lid_device=lid_device,
                        source="exp_c_segment_sample",
                        tokenizer=tokenizer,
                    )
                    if not segment_rows.empty:
                        segment_rows["selected_for_per_model"] = example_id in per_model_selected
                        segment_rows["kept_by_all_models"] = example_id in common_kept_examples
                        segment_rows["selected_for_aggregate_common"] = example_id in aggregate_common_examples
                        rows.append(segment_rows)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def trajectory_stats_from_token_rows(token_df: pd.DataFrame) -> pd.DataFrame:
    if token_df.empty:
        return pd.DataFrame()
    df = token_df.copy()
    thresholds = (
        df[np.isfinite(df["lid"])]
        .groupby(["model", "experiment", "layer"], dropna=False)["lid"]
        .quantile(0.10)
        .rename("low_lid_threshold")
        .reset_index()
    )
    df = df.merge(thresholds, on=["model", "experiment", "layer"], how="left")
    group_cols = [
        "model",
        "model_label",
        "model_sort_key",
        "hidden_dim",
        "lid_value_normalization",
        "experiment",
        "condition",
        "example_id",
        "mode",
        "prompt_variant",
        "family",
        "correct",
        "segment",
        "segment_label",
        "layer",
        "source",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        values = group["lid"].to_numpy(dtype=float)
        pos = group["normalized_pos"].to_numpy(dtype=float)
        mask = np.isfinite(values)
        clean = values[mask]
        clean_pos = pos[mask]
        row = dict(zip(group_cols, keys))
        row["n_tokens"] = int(len(values))
        row["valid_tokens"] = int(mask.sum())
        row["low_lid_threshold"] = float(group["low_lid_threshold"].dropna().iloc[0]) if group["low_lid_threshold"].notna().any() else np.nan
        if clean.size:
            row.update(
                {
                    "mean_lid": float(np.mean(clean)),
                    "min_lid": float(np.min(clean)),
                    "max_lid": float(np.max(clean)),
                    "variance_lid": float(np.var(clean)),
                    "auc_lid": float(np.trapz(clean, clean_pos) / max(np.ptp(clean_pos), 1e-12))
                    if clean.size > 1
                    else float(clean[0]),
                    "fraction_below_low_threshold": float(np.mean(clean < row["low_lid_threshold"]))
                    if np.isfinite(row["low_lid_threshold"])
                    else np.nan,
                }
            )
            if clean.size >= 2 and np.ptp(clean_pos) > 0:
                row["slope_lid"] = float(np.polyfit(clean_pos, clean, 1)[0])
            else:
                row["slope_lid"] = np.nan
            early = clean[clean_pos <= 0.25]
            late = clean[clean_pos >= 0.75]
            row["early_late_diff"] = (
                float(np.mean(late) - np.mean(early)) if early.size and late.size else np.nan
            )
        else:
            row.update(
                {
                    "mean_lid": np.nan,
                    "min_lid": np.nan,
                    "max_lid": np.nan,
                    "variance_lid": np.nan,
                    "auc_lid": np.nan,
                    "fraction_below_low_threshold": np.nan,
                    "slope_lid": np.nan,
                    "early_late_diff": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_exp_c_boundary_stats(stats: pd.DataFrame, token_df: pd.DataFrame) -> pd.DataFrame:
    if stats.empty or token_df.empty:
        return stats
    boundary_rows: list[dict[str, Any]] = []
    think = token_df[(token_df["mode"] == "think") & (token_df["segment_label"].isin(["thinking", "think_answer"]))]
    for keys, group in think.groupby(["model", "example_id", "layer"], dropna=False):
        model, example_id, layer = keys
        thinking = group[(group["segment_label"] == "thinking") & np.isfinite(group["lid"])]
        answer = group[(group["segment_label"] == "think_answer") & np.isfinite(group["lid"])]
        if thinking.empty or answer.empty:
            continue
        last_thinking = thinking.sort_values("token_index").tail(5)["lid"].mean()
        first_answer = answer.sort_values("token_index").head(5)["lid"].mean()
        boundary_rows.append(
            {
                "model": model,
                "example_id": example_id,
                "layer": layer,
                "boundary_jump_answer_minus_thinking": float(first_answer - last_thinking),
            }
        )
    if not boundary_rows:
        stats["boundary_jump_answer_minus_thinking"] = np.nan
        return stats
    boundary = pd.DataFrame(boundary_rows)
    return stats.merge(boundary, on=["model", "example_id", "layer"], how="left")


def plot_exp_c_boundary_anchored(token_df: pd.DataFrame, out_root: Path, config: MechanismConfig) -> None:
    if token_df.empty:
        return
    out_dir = out_root / "exp_c" / "boundary_anchored"
    out_dir.mkdir(parents=True, exist_ok=True)
    window = config.boundary_window

    thinking = token_df[(token_df["segment_label"] == "thinking") & np.isfinite(token_df["lid"])].copy()
    if not thinking.empty:
        thinking = thinking[thinking["token_offset"] <= window]
        binned = thinking.groupby(["model_label", "model_sort_key", "layer", "token_offset"], dropna=False)["lid"].mean().reset_index()
        for layer in config.layers:
            layer_df = binned[binned["layer"] == layer]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=layer_df, x="token_offset", y="lid", hue="model_label", hue_order=model_order(layer_df), ax=ax)
            ax.set_title(f"Exp C thinking-start anchored mean LID, layer {layer}")
            ax.set_xlabel("tokens after <think> start")
            ax.set_ylabel(f"mean token {lid_value_label(config=config)}")
            save_figure(fig, out_dir / f"thinking_start_mean_layer_{layer}")

    transition = token_df[
        (token_df["mode"] == "think")
        & (token_df["segment_label"].isin(["thinking", "think_answer"]))
        & np.isfinite(token_df["lid"])
    ].copy()
    if transition.empty:
        return
    starts = (
        transition[transition["segment_label"] == "think_answer"]
        .groupby(["model", "example_id"])["token_index"]
        .min()
        .rename("answer_start")
        .reset_index()
    )
    transition = transition.merge(starts, on=["model", "example_id"], how="inner")
    transition["boundary_offset"] = transition["token_index"] - transition["answer_start"]
    transition = transition[(transition["boundary_offset"] >= -window) & (transition["boundary_offset"] <= window)]
    binned = transition.groupby(["model_label", "model_sort_key", "layer", "boundary_offset"], dropna=False)["lid"].mean().reset_index()
    for layer in config.layers:
        layer_df = binned[binned["layer"] == layer]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=layer_df, x="boundary_offset", y="lid", hue="model_label", hue_order=model_order(layer_df), ax=ax)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Exp C thinking-to-answer transition mean LID, layer {layer}")
        ax.set_xlabel("token offset from answer start")
        ax.set_ylabel(f"mean token {lid_value_label(config=config)}")
        save_figure(fig, out_dir / f"thinking_to_answer_transition_layer_{layer}")


def plot_exp_c_normalized_trajectories(token_df: pd.DataFrame, out_root: Path, config: MechanismConfig) -> None:
    if token_df.empty:
        return
    out_dir = out_root / "exp_c" / "normalized_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    binned = normalized_bin_means(token_df, config.normalized_bins)
    phase_order = ["thinking", "think_answer", "no_think_answer"]
    for layer in config.layers:
        for phase in phase_order:
            layer_phase = binned[(binned["layer"] == layer) & (binned["segment_label"] == phase)]
            if layer_phase.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=layer_phase, x="pos_bin_center", y="lid", hue="model_label", hue_order=model_order(layer_phase), ax=ax)
            ax.set_title(f"Exp C normalized {phase} trajectory, layer {layer}")
            ax.set_xlabel("normalized segment position")
            ax.set_ylabel(f"mean token {lid_value_label(config=config)}")
            save_figure(fig, out_dir / f"{phase}_by_model_layer_{layer}")

        layer_df = binned[binned["layer"] == layer]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=layer_df,
            x="pos_bin_center",
            y="lid",
            hue="segment_label",
            style="model_label",
            hue_order=phase_order,
            ax=ax,
        )
        ax.set_title(f"Exp C normalized trajectories by phase and model, layer {layer}")
        ax.set_xlabel("normalized segment position")
        ax.set_ylabel(f"mean token {lid_value_label(config=config)}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        save_figure(fig, out_dir / f"all_phases_layer_{layer}")


def normalized_bin_means(token_df: pd.DataFrame, bins: int) -> pd.DataFrame:
    df = token_df[np.isfinite(token_df["lid"])].copy()
    if df.empty:
        return df
    bin_ids = np.floor(np.clip(df["normalized_pos"].to_numpy(dtype=float), 0, 0.999999) * bins).astype(int)
    df["pos_bin"] = bin_ids
    df["pos_bin_center"] = (df["pos_bin"] + 0.5) / bins
    return (
        df.groupby(["model", "model_label", "model_sort_key", "layer", "segment_label", "correct", "pos_bin", "pos_bin_center"], dropna=False)["lid"]
        .mean()
        .reset_index()
    )


def plot_exp_c_heatmaps(token_df: pd.DataFrame, stats: pd.DataFrame, out_root: Path, config: MechanismConfig) -> None:
    if token_df.empty or stats.empty:
        return
    out_dir = out_root / "exp_c" / "sample_token_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = np.linspace(0, 1, config.normalized_bins)
    delta_sort = build_delta_sort_from_stats(stats)
    for (model, label, layer, phase), group in token_df.groupby(
        ["model", "model_label", "layer", "segment_label"], dropna=False
    ):
        if phase not in {"thinking", "think_answer", "no_think_answer"}:
            continue
        matrix_rows: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []
        for example_id, sample in group.groupby("example_id", dropna=False):
            sample = sample[np.isfinite(sample["lid"])].sort_values("normalized_pos")
            if len(sample) < 2:
                continue
            x = sample["normalized_pos"].to_numpy(dtype=float)
            y = sample["lid"].to_numpy(dtype=float)
            unique_x, unique_indices = np.unique(x, return_index=True)
            if len(unique_x) < 2:
                continue
            interp = np.interp(grids, unique_x, y[unique_indices])
            matrix_rows.append(interp)
            meta_rows.append(
                {
                    "example_id": example_id,
                    "correct": bool(sample["correct"].iloc[0]) if not pd.isna(sample["correct"].iloc[0]) else False,
                    "mean_lid": float(np.nanmean(y)),
                    "delta_sort": delta_sort.get((model, example_id, layer), np.nan),
                }
            )
        if not matrix_rows:
            continue
        matrix = np.vstack(matrix_rows)
        meta = pd.DataFrame(meta_rows)
        sort_specs = {
            "correctness_then_mean_lid": meta.assign(correct_sort=~meta["correct"]).sort_values(["correct_sort", "mean_lid"]).index,
            "mean_lid": meta.sort_values("mean_lid").index,
            "paired_delta": meta.sort_values("delta_sort", na_position="last").index,
        }
        for sort_name, order in sort_specs.items():
            sorted_matrix = matrix[list(order)]
            sorted_meta = meta.iloc[list(order)]
            fig, ax = plt.subplots(figsize=(10, max(5, 0.08 * len(sorted_matrix))))
            sns.heatmap(sorted_matrix, cmap="mako", ax=ax, cbar_kws={"label": f"token {lid_value_label(config=config)}"})
            ax.set_title(f"{label} Exp C {phase} LID heatmap, layer {layer}, sorted by {sort_name}")
            ax.set_xlabel("normalized token position")
            ax.set_ylabel("samples")
            tick_positions = np.linspace(0, config.normalized_bins - 1, 6)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])
            if len(sorted_meta) <= 80:
                correct_positions = np.where(sorted_meta["correct"].to_numpy())[0]
                for pos in correct_positions:
                    ax.axhline(pos, color="white", linewidth=0.15, alpha=0.4)
            save_figure(fig, out_dir / f"{safe_name(label)}__layer_{layer}__{safe_name(phase)}__sort_{sort_name}")


def build_delta_sort_from_stats(stats: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    pivot = stats[stats["segment_label"].isin(["think_answer", "no_think_answer"])].pivot_table(
        index=["model", "example_id", "layer"],
        columns="segment_label",
        values="mean_lid",
        aggfunc="first",
    )
    result: dict[tuple[str, str, int], float] = {}
    for idx, row in pivot.iterrows():
        try:
            result[idx] = float(row["think_answer"] - row["no_think_answer"])
        except Exception:
            result[idx] = np.nan
    return result


def plot_exp_c_trajectory_statistics(
    stats: pd.DataFrame,
    out_root: Path,
    aggregate_stats: pd.DataFrame | None = None,
) -> None:
    if stats.empty:
        return
    out_dir = out_root / "exp_c" / "trajectory_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = stats[stats["segment_label"].isin(["thinking", "think_answer", "no_think_answer"])].copy()
    stats["correct_label"] = stats["correct"].apply(correctness_label)
    aggregate = aggregate_stats if aggregate_stats is not None and not aggregate_stats.empty else stats
    aggregate = aggregate[aggregate["segment_label"].isin(["thinking", "think_answer", "no_think_answer"])].copy()
    aggregate["correct_label"] = aggregate["correct"].apply(correctness_label)
    phase_order = ["thinking", "think_answer", "no_think_answer"]
    correctness_order = ["correct", "wrong", "unknown"]
    stat_cols = [
        "min_lid",
        "max_lid",
        "variance_lid",
        "slope_lid",
        "early_late_diff",
        "fraction_below_low_threshold",
        "auc_lid",
        "boundary_jump_answer_minus_thinking",
    ]
    for stat in stat_cols:
        if stat not in stats or not np.isfinite(stats[stat]).any():
            continue
        for layer in sorted(stats["layer"].dropna().unique()):
            layer_df = stats[(stats["layer"] == layer) & np.isfinite(stats[stat])]
            if layer_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(
                data=layer_df,
                x="model_label",
                y=stat,
                hue="segment_label",
                order=model_order(layer_df),
                cut=0,
                inner="quartile",
                ax=ax,
            )
            ax.set_title(f"Exp C trajectory statistic: {stat}, layer {int(layer)}")
            ax.set_xlabel("model size")
            ax.set_ylabel(stat)
            ax.legend(title="phase", bbox_to_anchor=(1.02, 1), loc="upper left")
            save_figure(fig, out_dir / f"{safe_name(stat)}__layer_{int(layer)}")

            aggregate_layer_df = aggregate[(aggregate["layer"] == layer) & np.isfinite(aggregate[stat])]
            if aggregate_layer_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                data=aggregate_layer_df,
                x="segment_label",
                y=stat,
                hue="correct_label",
                order=phase_order,
                hue_order=correctness_order,
                ax=ax,
            )
            ax.set_title(f"Exp C {stat} by correctness, common-kept examples across model sizes, layer {int(layer)}")
            ax.set_xlabel("phase")
            ax.set_ylabel(stat)
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="correctness", bbox_to_anchor=(1.02, 1), loc="upper left")
            save_figure(fig, out_dir / f"{safe_name(stat)}__correctness_layer_{int(layer)}")

            model_dir = out_dir / "by_model_correctness"
            for model_label, model_df in layer_df.groupby("model_label", dropna=False):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(
                    data=model_df,
                    x="segment_label",
                    y=stat,
                    hue="correct_label",
                    order=phase_order,
                    hue_order=correctness_order,
                    ax=ax,
                )
                ax.set_title(f"Exp C {stat} by correctness, {model_label}, layer {int(layer)}")
                ax.set_xlabel("phase")
                ax.set_ylabel(stat)
                ax.tick_params(axis="x", rotation=20)
                ax.legend(title="correctness", bbox_to_anchor=(1.02, 1), loc="upper left")
                save_figure(
                    fig,
                    model_dir / f"{safe_name(stat)}__{safe_name(model_label)}__correctness_layer_{int(layer)}",
                )


def summarize_exp_c_correctness_statistics(
    stats: pd.DataFrame,
    aggregate_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    stat_cols = [
        "mean_lid",
        "min_lid",
        "max_lid",
        "variance_lid",
        "slope_lid",
        "early_late_diff",
        "fraction_below_low_threshold",
        "auc_lid",
        "boundary_jump_answer_minus_thinking",
    ]
    phase_descriptions = {
        "thinking": "thinking_model_thinking_segment",
        "think_answer": "thinking_model_answer_segment",
        "no_think_answer": "no_thinking_model_answer_segment",
    }
    per_model_df = stats[stats["segment_label"].isin(phase_descriptions)].copy()
    aggregate_df = (
        aggregate_stats[aggregate_stats["segment_label"].isin(phase_descriptions)].copy()
        if aggregate_stats is not None and not aggregate_stats.empty
        else per_model_df.copy()
    )
    if per_model_df.empty and aggregate_df.empty:
        return pd.DataFrame()
    per_model_df["correct_label"] = per_model_df["correct"].apply(correctness_label)
    aggregate_df["correct_label"] = aggregate_df["correct"].apply(correctness_label)
    normalization_values = (
        per_model_df.get("lid_value_normalization", pd.Series(["unknown"]))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    aggregate_normalization = normalization_values[0] if len(normalization_values) == 1 else "mixed"
    rows: list[dict[str, Any]] = []

    scopes: list[tuple[str, pd.DataFrame, list[str]]] = [
        ("aggregate_common_kept_across_models", aggregate_df, ["layer", "segment_label", "correct_label"]),
        (
            "per_model",
            per_model_df,
            [
                "model",
                "model_label",
                "model_sort_key",
                "hidden_dim",
                "lid_value_normalization",
                "layer",
                "segment_label",
                "correct_label",
            ],
        ),
    ]
    for scope, scope_df, group_cols in scopes:
        if scope_df.empty:
            continue
        for keys, group in scope_df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_dict = dict(zip(group_cols, keys))
            base = {
                "scope": scope,
                "model": key_dict.get("model", "ALL_MODELS"),
                "model_label": key_dict.get("model_label", "ALL_MODELS"),
                "model_sort_key": key_dict.get("model_sort_key", np.nan),
                "hidden_dim": key_dict.get("hidden_dim", np.nan),
                "lid_value_normalization": key_dict.get("lid_value_normalization", aggregate_normalization),
                "layer": key_dict.get("layer"),
                "segment_label": key_dict.get("segment_label"),
                "phase_description": phase_descriptions.get(key_dict.get("segment_label"), key_dict.get("segment_label")),
                "correct_label": key_dict.get("correct_label"),
            }
            for stat in stat_cols:
                if stat not in group:
                    continue
                values = group[stat].dropna().to_numpy(dtype=float)
                if values.size == 0:
                    continue
                ci_low, ci_high = bootstrap_ci(values, seed=1234)
                rows.append(
                    {
                        **base,
                        "statistic": stat,
                        "n": int(values.size),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["scope", "model_sort_key", "layer", "segment_label", "correct_label", "statistic"])


def correctness_label(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "1.0", "correct"}:
            return "correct"
        if lower in {"false", "0", "0.0", "wrong", "incorrect"}:
            return "wrong"
        return "unknown"
    return "correct" if bool(value) else "wrong"


def plot_exp_c_token_category_summaries(token_df: pd.DataFrame, out_root: Path) -> None:
    if token_df.empty or "token_category" not in token_df or token_df["token_category"].eq("not_decoded").all():
        return
    out_dir = out_root / "exp_c" / "token_category_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = token_df[np.isfinite(token_df["lid"])].copy()
    summary = (
        df.groupby(["model", "model_label", "model_sort_key", "layer", "segment_label", "token_category"], dropna=False)
        .agg(mean_lid=("lid", "mean"), median_lid=("lid", "median"), n=("lid", "size"))
        .reset_index()
        .sort_values(["model_sort_key", "layer", "segment_label", "token_category"])
    )
    summary.to_csv(out_dir / "token_category_lid_summary.csv", index=False)
    for layer in sorted(summary["layer"].dropna().unique()):
        layer_df = summary[(summary["layer"] == layer) & (summary["segment_label"].isin(["thinking", "think_answer", "no_think_answer"]))]
        if layer_df.empty:
            continue
        g = sns.catplot(
            data=layer_df,
            x="token_category",
            y="mean_lid",
            hue="model_label",
            col="segment_label",
            kind="bar",
            col_order=["thinking", "think_answer", "no_think_answer"],
            height=5,
            aspect=1.2,
            sharey=False,
        )
        for ax in g.axes.flat:
            ax.tick_params(axis="x", rotation=70)
            ax.set_xlabel("")
        g.fig.suptitle(f"Exp C token-category mean {lid_value_label(layer_df)}, layer {int(layer)}", y=1.04)
        save_facet(g, out_dir / f"token_category_mean_lid_layer_{int(layer)}")


def plot_accuracy_colored_trajectories(token_df: pd.DataFrame, out_root: Path, config: MechanismConfig) -> None:
    if token_df.empty:
        return
    out_dir = out_root / "exp_c" / "accuracy_colored_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    binned = normalized_bin_means(token_df, config.normalized_bins)
    binned["correct_label"] = binned["correct"].map({True: "correct", False: "wrong"}).fillna("unknown")
    for layer in config.layers:
        for phase in ["thinking", "think_answer", "no_think_answer"]:
            layer_phase = binned[(binned["layer"] == layer) & (binned["segment_label"] == phase)]
            if layer_phase.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=layer_phase,
                x="pos_bin_center",
                y="lid",
                hue="correct_label",
                style="model_label",
                hue_order=["correct", "wrong", "unknown"],
                ax=ax,
            )
            ax.set_title(f"Exp C {phase} normalized trajectory by correctness, layer {layer}")
            ax.set_xlabel("normalized segment position")
            ax.set_ylabel(f"mean token {lid_value_label(config=config)}")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            save_figure(fig, out_dir / f"{safe_name(phase)}__correctness_layer_{layer}")


def write_readme(out_root: Path, model_runs: list[ModelRun], config: MechanismConfig, lid_device: str | None) -> None:
    lines = [
        "# Mechanism Visualization Outputs",
        "",
        "This directory contains exploratory mechanism-discovery plots generated from the multi-model output folders.",
        "",
        f"- Models: {', '.join(run.name for run in model_runs)}",
        f"- Layers: {', '.join(str(layer) for layer in config.layers)}",
        f"- LID k: {config.k}",
        f"- LID value normalization: {config.lid_value_normalization}",
        f"- Token-level recomputation device: {lid_device or 'cpu'}",
        f"- Max tokens per token-level LID recomputation: {config.max_lid_tokens or 'all'}",
        f"- Max Exp C examples per model for token-level segment diagnostics: {config.max_exp_c_token_examples or 'all'}",
        "",
        "Aggregate scale/correctness/delta plots use the complete existing `sample_metrics.csv` and `pair_metrics.csv` tables.",
        f"Token-level trajectory, heatmap, boundary, and token-category plots are recomputed from hidden states and cached under each experiment's `{config.per_model_dir_name}/cache/` directory.",
        "",
        "Main subdirectories:",
        "",
        "- `exp_a/mean_lid_comparisons/`: scale comparisons for repetition and regular GSM8K families.",
        "- `exp_b/mean_lid_comparisons/`: corrected Exp B prompt-condition answer LID comparisons.",
        "- `exp_c/mean_lid_comparisons/`: thinking, think-answer, and no-think-answer segment comparisons.",
        "- `exp_c/delta_distributions/`: paired delta histograms, ECDFs, violins, and delta summaries.",
        "- `exp_c/accuracy_by_model/`: Exp C answer accuracy by model size using each model's own kept examples.",
        "- `exp_c/boundary_anchored/`: trajectories aligned to thinking start and thinking-to-answer transition.",
        "- `exp_c/normalized_trajectories/`: average segment-shape plots after normalizing token position.",
        "- `exp_c/sample_token_heatmaps/`: sample-by-token normalized heatmaps sorted by correctness, mean LID, and paired delta.",
        "- `exp_c/trajectory_statistics/`: minimum, variance, slope, early-vs-late, low-LID fraction, AUC, and boundary-jump views. Aggregate correctness plots use the common kept-example subset across the included models; `by_model_correctness/` uses each model's own kept examples.",
        "- `exp_c/token_category_diagnostics/`: token-category LID summaries when the tokenizer is available locally.",
        "- `data/`: joined input metrics and derived CSV summaries, including `exp_c_common_kept_token_trajectory_stats.csv` and `exp_c_accuracy_by_model_kept_per_model.csv`.",
        "",
        f"Per-model raw trajectory examples are saved in each model folder under `<model>/<exp>/{config.per_model_dir_name}/per_sample_trajectories/`.",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text("\n".join(lines) + "\n")


def write_qualitative_notes(
    out_root: Path,
    key_stats: dict[str, pd.DataFrame],
    delta_summary: pd.DataFrame,
    exp_c_stats: pd.DataFrame,
    sample_metrics: pd.DataFrame,
) -> None:
    lines = ["# Qualitative Mechanism Notes", ""]
    lines.append("These notes are automatically derived from the generated summaries and are intended as prompts for inspection, not final-report claims.")
    lines.append("")

    exp_a = key_stats.get("exp_a", pd.DataFrame())
    if not exp_a.empty and "family" in exp_a:
        lines.append("## Exp A")
        for layer in sorted(exp_a["layer"].dropna().unique()):
            layer_df = exp_a[exp_a["layer"] == layer].sort_values("mean")
            if not layer_df.empty:
                low = layer_df.iloc[0]
                high = layer_df.iloc[-1]
                lines.append(
                    f"- Layer {int(layer)}: lowest mean LID is `{low['family']}` on {low['model_label']} ({format_stat_value(low['mean'])}); highest is `{high['family']}` on {high['model_label']} ({format_stat_value(high['mean'])})."
                )
        lines.append("")

    exp_b = key_stats.get("exp_b", pd.DataFrame())
    if not exp_b.empty and "prompt_variant" in exp_b:
        lines.append("## Exp B")
        for layer in sorted(exp_b["layer"].dropna().unique()):
            layer_df = exp_b[exp_b["layer"] == layer]
            by_variant = layer_df.groupby("prompt_variant")["mean"].mean().sort_values()
            if not by_variant.empty:
                lines.append(
                    f"- Layer {int(layer)} average ordering across scale: "
                    + " < ".join(f"{idx} ({format_stat_value(value)})" for idx, value in by_variant.items())
                    + "."
                )
        lines.append("")

    exp_c = key_stats.get("exp_c", pd.DataFrame())
    if not exp_c.empty and "phase" in exp_c:
        lines.append("## Exp C")
        for layer in sorted(exp_c["layer"].dropna().unique()):
            layer_df = exp_c[exp_c["layer"] == layer]
            phase_means = layer_df.groupby("phase")["mean"].mean().sort_values()
            if not phase_means.empty:
                lines.append(
                    f"- Layer {int(layer)} mean segment ordering across scale: "
                    + " < ".join(f"{idx} ({format_stat_value(value)})" for idx, value in phase_means.items())
                    + "."
                )
        lines.append("")

    if not delta_summary.empty:
        lines.append("## Paired Deltas")
        for delta_name, group in delta_summary.groupby("delta_name"):
            frac_negative = group["fraction_below_zero"].mean()
            mean_delta = group["mean"].mean()
            lines.append(
                f"- `{delta_name}`: mean over model/layer summaries {format_stat_value(mean_delta)}; average fraction below zero {format_stat_value(frac_negative)}."
            )
        lines.append("")

    if not exp_c_stats.empty:
        lines.append("## Trajectory Dynamics")
        for stat in ["min_lid", "variance_lid", "slope_lid", "early_late_diff", "fraction_below_low_threshold"]:
            if stat not in exp_c_stats or not np.isfinite(exp_c_stats[stat]).any():
                continue
            summary = (
                exp_c_stats.groupby("segment_label")[stat]
                .mean()
                .dropna()
                .sort_values()
            )
            if not summary.empty:
                lines.append(
                    f"- `{stat}` phase ordering in sampled token diagnostics: "
                    + " < ".join(f"{idx} ({format_stat_value(value)})" for idx, value in summary.items())
                    + "."
                )
        lines.append("")

    valid = valid_metric_rows(sample_metrics)
    if not valid.empty and "correct" in valid:
        c = valid[(valid["experiment"] == "exp_c") & (valid["segment_label"].isin(["thinking", "think_answer", "no_think_answer"]))]
        if not c.empty:
            lines.append("## Correctness")
            correctness_gap = (
                c.groupby(["segment_label", "correct"])["mean_lid"].mean().unstack("correct")
            )
            for phase, row in correctness_gap.iterrows():
                wrong = first_present(row, [False, 0, 0.0, "False", "false", "0", "0.0"])
                correct = first_present(row, [True, 1, 1.0, "True", "true", "1", "1.0"])
                if np.isfinite(wrong) and np.isfinite(correct):
                    lines.append(f"- `{phase}` correct-minus-wrong mean LID gap: {format_stat_value(correct - wrong)}.")
            lines.append("")

    (out_root / "QUALITATIVE_NOTES.md").write_text("\n".join(lines) + "\n")


def format_stat_value(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return "nan"
    if number != 0 and abs(number) < 0.01:
        return f"{number:.3e}"
    return f"{number:.2f}"


def first_present(row: pd.Series, keys: list[Any]) -> float:
    for key in keys:
        if key in row.index:
            try:
                return float(row[key])
            except Exception:
                return float("nan")
    return float("nan")
