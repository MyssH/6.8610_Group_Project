from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "qwen_lid_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


FAMILY_ORDER = [
    "constant_repetition",
    "short_cycle_repetition",
    "templatic_repetition",
    "regular_gsm8k_nothink",
]


def _save(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _valid_lid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"valid", "mean_lid"}.issubset(df.columns):
        return df
    return df[(df["valid"] == True) & pd.notna(df["mean_lid"])].copy()


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _box_scatter(ax: plt.Axes, groups: list[np.ndarray], labels: list[str], title: str, ylabel: str) -> None:
    positions = np.arange(1, len(groups) + 1)
    ax.boxplot(groups, positions=positions, showfliers=False)
    for pos, values in zip(positions, groups):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), pos) + jitter, values, s=18, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def plot_exp_a(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    sample_path = output_dir / "sample_metrics.csv"
    summary_path = output_dir / "summary_metrics.csv"
    created: list[Path] = []
    if not sample_path.exists():
        return created
    sample = _valid_lid(_read_csv_or_empty(sample_path))
    if sample.empty or "segment" not in sample:
        return created
    sample = sample[sample["segment"] == "full_output"]
    for layer in sorted(sample["layer"].dropna().unique()):
        layer_df = sample[sample["layer"] == layer]
        labels = [family for family in FAMILY_ORDER if family in set(layer_df["family"])]
        groups = [layer_df[layer_df["family"] == family]["mean_lid"].to_numpy() for family in labels]
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        _box_scatter(ax, groups, labels, f"Experiment A LID by family, layer {layer}", "Mean LID")
        base = figures_dir / f"exp_a_lid_by_family_layer_{int(layer)}"
        _save(fig, base)
        created.append(base.with_suffix(".png"))

    if summary_path.exists():
        summary = _read_csv_or_empty(summary_path)
        if summary.empty or "summary_type" not in summary:
            return created
        summary = summary[summary["summary_type"] == "family_layer"]
        for layer in sorted(summary["layer"].dropna().unique()):
            layer_summary = summary[summary["layer"] == layer]
            labels = [family for family in FAMILY_ORDER if family in set(layer_summary["family"])]
            if not labels:
                continue
            means = [layer_summary[layer_summary["family"] == family]["mean"].iloc[0] for family in labels]
            ci_low = [layer_summary[layer_summary["family"] == family]["ci_low"].iloc[0] for family in labels]
            ci_high = [layer_summary[layer_summary["family"] == family]["ci_high"].iloc[0] for family in labels]
            yerr = np.array([np.array(means) - np.array(ci_low), np.array(ci_high) - np.array(means)])
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(labels, means, yerr=yerr, capsize=4)
            ax.set_title(f"Experiment A mean LID with bootstrap CI, layer {layer}")
            ax.set_ylabel("Mean sample LID")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
            base = figures_dir / f"exp_a_summary_bar_layer_{int(layer)}"
            _save(fig, base)
            created.append(base.with_suffix(".png"))
    return created


def _plot_canonical_comparison(output_dir: str | Path, experiment_label: str, prefix: str) -> list[Path]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    sample_path = output_dir / "sample_metrics.csv"
    pair_path = output_dir / "pair_metrics.csv"
    created: list[Path] = []
    if not sample_path.exists():
        return created
    sample = _read_csv_or_empty(sample_path)
    if sample.empty:
        return created
    valid = _valid_lid(sample)

    if pair_path.exists():
        pair = _read_csv_or_empty(pair_path)
        if not pair.empty and "layer" in pair and "valid_pair" in pair:
            for layer in sorted(pair["layer"].dropna().unique()):
                layer_pair = pair[(pair["layer"] == layer) & (pair["valid_pair"] == True)]
                if not layer_pair.empty:
                    fig, ax = plt.subplots(figsize=(5.5, 5.5))
                    x = layer_pair["no_think_answer_lid"].to_numpy()
                    y = layer_pair["think_answer_lid"].to_numpy()
                    ax.scatter(x, y, alpha=0.75)
                    low = min(float(np.nanmin(x)), float(np.nanmin(y)))
                    high = max(float(np.nanmax(x)), float(np.nanmax(y)))
                    ax.plot([low, high], [low, high], color="black", linewidth=1)
                    ax.set_xlabel("No-think answer mean LID")
                    ax.set_ylabel("Think answer mean LID")
                    ax.set_title(f"{experiment_label} paired answer LID, layer {layer}")
                    ax.grid(alpha=0.25)
                    base = figures_dir / f"{prefix}_paired_scatter_layer_{int(layer)}"
                    _save(fig, base)
                    created.append(base.with_suffix(".png"))

                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.hist(layer_pair["answer_lid_diff"].dropna(), bins=25)
                    ax.axvline(0, color="black", linewidth=1)
                    ax.set_xlabel("Think answer LID - no-think answer LID")
                    ax.set_ylabel("Count")
                    ax.set_title(f"{experiment_label} paired differences, layer {layer}")
                    base = figures_dir / f"{prefix}_difference_hist_layer_{int(layer)}"
                    _save(fig, base)
                    created.append(base.with_suffix(".png"))

    condition_rows = []
    for label, mode, segment in [
        ("thinking_segment", "think", "thinking_segment"),
        ("think_answer_segment", "think", "answer_segment"),
        ("nothink_answer_segment", "no_think", "answer_segment"),
    ]:
        rows = valid[(valid["mode"] == mode) & (valid["segment"] == segment)].copy()
        rows["condition_label"] = label
        condition_rows.append(rows)
    if condition_rows:
        combined = pd.concat(condition_rows, ignore_index=True)
        for layer in sorted(combined["layer"].dropna().unique()):
            layer_df = combined[combined["layer"] == layer]
            labels = ["thinking_segment", "think_answer_segment", "nothink_answer_segment"]
            groups = [layer_df[layer_df["condition_label"] == label]["mean_lid"].to_numpy() for label in labels]
            if any(len(group) for group in groups):
                fig, ax = plt.subplots(figsize=(8, 5))
                _box_scatter(ax, groups, labels, f"{experiment_label} segment LID, layer {layer}", "Mean LID")
                base = figures_dir / f"{prefix}_segment_box_layer_{int(layer)}"
                _save(fig, base)
                created.append(base.with_suffix(".png"))

    _plot_accuracy_and_lengths(sample, figures_dir, prefix, created, title_prefix=experiment_label)
    return created


def _plot_prompt_structure(output_dir: str | Path, experiment_label: str, prefix: str) -> list[Path]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    sample_path = output_dir / "sample_metrics.csv"
    pair_path = output_dir / "pair_metrics.csv"
    created: list[Path] = []
    if not sample_path.exists():
        return created
    sample = _read_csv_or_empty(sample_path)
    if sample.empty:
        return created
    valid = _valid_lid(sample)
    answer = valid[valid["segment"] == "answer_segment"]
    for layer in sorted(answer["layer"].dropna().unique()):
        layer_df = answer[answer["layer"] == layer]
        labels = []
        groups = []
        present_modes = [mode for mode in ["no_think", "think"] if mode in set(layer_df["mode"])]
        for variant in ["canonical", "irrelevant_context", "repetitive_filler"]:
            for mode in present_modes:
                label = f"{variant}\n{mode}"
                values = layer_df[(layer_df["prompt_variant"] == variant) & (layer_df["mode"] == mode)][
                    "mean_lid"
                ].to_numpy()
                labels.append(label)
                groups.append(values)
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=(11, 5))
        _box_scatter(ax, groups, labels, f"{experiment_label} answer LID, layer {layer}", "Mean LID")
        base = figures_dir / f"{prefix}_answer_lid_layer_{int(layer)}"
        _save(fig, base)
        created.append(base.with_suffix(".png"))

    if pair_path.exists():
        pair = _read_csv_or_empty(pair_path)
        if not pair.empty and "layer" in pair and "valid_pair" in pair:
            for layer in sorted(pair["layer"].dropna().unique()):
                layer_pair = pair[(pair["layer"] == layer) & (pair["valid_pair"] == True)]
                if layer_pair.empty:
                    continue
                labels = ["canonical", "irrelevant_context", "repetitive_filler"]
                groups = [layer_pair[layer_pair["prompt_variant"] == label]["answer_lid_diff"].dropna().to_numpy() for label in labels]
                fig, ax = plt.subplots(figsize=(8, 5))
                _box_scatter(ax, groups, labels, f"{experiment_label} paired differences, layer {layer}", "Think - no-think LID")
                ax.axhline(0, color="black", linewidth=1)
                base = figures_dir / f"{prefix}_paired_differences_layer_{int(layer)}"
                _save(fig, base)
                created.append(base.with_suffix(".png"))

    _plot_accuracy_and_lengths(sample, figures_dir, prefix, created, title_prefix=experiment_label)
    return created


def plot_exp_b(output_dir: str | Path) -> list[Path]:
    return _plot_prompt_structure(output_dir, "Experiment B", "exp_b")


def plot_exp_c(output_dir: str | Path) -> list[Path]:
    return _plot_canonical_comparison(output_dir, "Experiment C", "exp_c")


def _plot_accuracy_and_lengths(
    sample: pd.DataFrame,
    figures_dir: Path,
    prefix: str,
    created: list[Path],
    title_prefix: str,
) -> None:
    answer_rows = sample[sample["segment"] == "answer_segment"].copy()
    if answer_rows.empty:
        return
    dedup_cols = ["example_id", "mode", "prompt_variant"]
    answer_once = answer_rows.drop_duplicates(dedup_cols)
    if "correct" in answer_once and answer_once["correct"].notna().any():
        acc = answer_once.groupby(["prompt_variant", "mode"], dropna=False)["correct"].mean().reset_index()
        labels = [f"{row.prompt_variant}\n{row.mode}" for row in acc.itertuples()]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, acc["correct"].to_numpy())
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title_prefix} accuracy")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        base = figures_dir / f"{prefix}_accuracy"
        _save(fig, base)
        created.append(base.with_suffix(".png"))

    if "token_count" in answer_once:
        labels = []
        groups = []
        for key, group in answer_once.groupby(["prompt_variant", "mode"], dropna=False):
            labels.append(f"{key[0]}\n{key[1]}")
            groups.append(group["token_count"].to_numpy())
        if groups:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _box_scatter(ax, groups, labels, f"{title_prefix} answer token length", "Generated tokens")
            base = figures_dir / f"{prefix}_answer_token_lengths"
            _save(fig, base)
            created.append(base.with_suffix(".png"))


def regenerate_all_plots(outputs_root: str | Path = "outputs") -> list[Path]:
    outputs_root = Path(outputs_root)
    created: list[Path] = []
    created.extend(plot_exp_a(outputs_root / "exp_a"))
    created.extend(plot_exp_b(outputs_root / "exp_b"))
    created.extend(plot_exp_c(outputs_root / "exp_c"))

    overview_dir = outputs_root / "plots"
    overview_dir.mkdir(parents=True, exist_ok=True)
    for figure in created:
        if figure.exists():
            target = overview_dir / figure.name
            shutil.copy2(figure, target)
    return created
