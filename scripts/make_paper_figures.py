"""Generate the paper-ready figures into report/Images/.

Design principles:
  - Use **mean LID** consistently (no AUC-LID, no compound metrics).
  - One paper figure per claim, with within-model and across-scale views
    composed as subplots of a single figure when natural.
  - Read each model's full sample_metrics.csv (no token-level subsampling),
    so wrong-answer examples are never dropped for high-accuracy models.

Inputs (read-only):
  outputs/mechanism_visualizations/data/all_sample_metrics.csv

Outputs (written):
  report/Images/fig_exp_a_layer{L}.png
  report/Images/fig_exp_b_layer{L}.png
  report/Images/fig_exp_c_segments_layer{L}.png      (C.1)
  report/Images/fig_exp_c_correctness_layer{L}.png   (C.2)
  report/Images/fig_exp_c_correctness_gap.png        (headline summary)

Layers generated: 6, 13, 20. The body of the paper uses layer 13; the
others go in the appendix.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["1.7B", "4B", "8B", "14B", "32B"]
SEGMENT_ORDER = ["thinking", "think_answer", "no_think_answer"]
SEGMENT_LABEL = {
    "thinking": "thinking",
    "think_answer": "think-answer",
    "no_think_answer": "no-think-answer",
}
FAMILY_ORDER = ["regular_gsm8k_nothink", "constant_repetition", "short_cycle_repetition", "templatic_repetition"]
FAMILY_LABEL = {
    "regular_gsm8k_nothink": "regular GSM8K",
    "constant_repetition": "constant rep.",
    "short_cycle_repetition": "short-cycle rep.",
    "templatic_repetition": "templatic rep.",
}
PROMPT_ORDER = ["canonical", "irrelevant_context", "repetitive_filler"]
PROMPT_LABEL = {
    "canonical": "canonical",
    "irrelevant_context": "irrelevant ctx.",
    "repetitive_filler": "repetitive filler",
}


def boot_ci(values: np.ndarray, n: int = 2000, seed: int = 1234) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def load(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    df = df[df["valid"].astype(str).str.lower() == "true"]
    df = df[df["mean_lid"].notna()]
    df["phase"] = df.apply(
        lambda r: (
            "thinking" if r["mode"] == "think" and r["segment"] == "thinking_segment"
            else "think_answer" if r["mode"] == "think" and r["segment"] == "answer_segment"
            else "no_think_answer" if r["mode"] == "no_think" and r["segment"] == "answer_segment"
            else None
        ),
        axis=1,
    )
    df["correct_bool"] = df["correct"].map({1.0: True, 0.0: False, True: True, False: False})
    return df


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
    })


# ------------------------------------------------------------- Exp A figure
def figure_exp_a(df: pd.DataFrame, layer: int, out_path: Path) -> None:
    exp_a = df[(df["experiment"] == "exp_a") & (df["layer"] == layer)].copy()
    if exp_a.empty:
        return
    exp_a = exp_a[exp_a["family"].isin(FAMILY_ORDER)]
    exp_a = exp_a[exp_a["segment"] == "full_output"]
    exp_a["family_label"] = exp_a["family"].map(FAMILY_LABEL)
    family_palette = {
        FAMILY_LABEL["regular_gsm8k_nothink"]: "#1f77b4",
        FAMILY_LABEL["constant_repetition"]: "#d62728",
        FAMILY_LABEL["short_cycle_repetition"]: "#e377c2",
        FAMILY_LABEL["templatic_repetition"]: "#9467bd",
    }
    order = [FAMILY_LABEL[f] for f in FAMILY_ORDER]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.1, 1.9]})

    # Left: within-model distribution (Qwen3-1.7B)
    m = exp_a[exp_a["model_label"] == "1.7B"]
    sns.violinplot(
        data=m, x="family_label", y="mean_lid",
        order=order, palette=family_palette, inner="box", cut=0,
        ax=ax_left,
    )
    ax_left.set_title("Qwen3-1.7B")
    ax_left.set_xlabel("")
    ax_left.set_ylabel("mean LID")
    ax_left.tick_params(axis="x", rotation=25)
    for tick in ax_left.get_xticklabels():
        tick.set_horizontalalignment("right")

    # Right: across-scale point + 95% CI
    rows = []
    for (model_label, fam), sub in exp_a.groupby(["model_label", "family"]):
        vals = sub["mean_lid"].to_numpy()
        if vals.size == 0:
            continue
        lo, hi = boot_ci(vals)
        rows.append({
            "model_label": model_label,
            "family": fam,
            "family_label": FAMILY_LABEL[fam],
            "mean": float(np.mean(vals)),
            "ci_low": lo, "ci_high": hi, "n": int(vals.size),
        })
    summary = pd.DataFrame(rows)

    for i, fam in enumerate(FAMILY_ORDER):
        fam_label = FAMILY_LABEL[fam]
        sub = summary[summary["family"] == fam]
        x = np.array([MODEL_ORDER.index(m) for m in sub["model_label"]], dtype=float) + (i - 1.5) * 0.13
        ax_right.errorbar(
            x, sub["mean"], yerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
            fmt="o", color=family_palette[fam_label], label=fam_label,
            markersize=5, elinewidth=1.2, capsize=2.5,
        )
    ax_right.set_xticks(range(len(MODEL_ORDER)))
    ax_right.set_xticklabels(MODEL_ORDER)
    ax_right.set_title(f"All Qwen3 sizes, layer {layer}")
    ax_right.set_xlabel("model size")
    ax_right.set_ylabel("mean LID")
    ax_right.legend(loc="lower right", ncol=2, fontsize=8)

    fig.suptitle(f"Experiment A: regular vs. degenerate generation (layer {layer})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- Exp B figure
def figure_exp_b(df: pd.DataFrame, layer: int, out_path: Path) -> None:
    exp_b = df[(df["experiment"] == "exp_b") & (df["layer"] == layer)].copy()
    if exp_b.empty:
        return
    exp_b = exp_b[exp_b["segment"] == "answer_segment"]
    exp_b = exp_b[exp_b["prompt_variant"].isin(PROMPT_ORDER)]
    exp_b["prompt_label"] = exp_b["prompt_variant"].map(PROMPT_LABEL)
    palette = {
        PROMPT_LABEL["canonical"]: "#1f77b4",
        PROMPT_LABEL["irrelevant_context"]: "#ff7f0e",
        PROMPT_LABEL["repetitive_filler"]: "#d62728",
    }
    order = [PROMPT_LABEL[p] for p in PROMPT_ORDER]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.1, 1.9]})

    m = exp_b[exp_b["model_label"] == "1.7B"]
    sns.violinplot(
        data=m, x="prompt_label", y="mean_lid",
        order=order, palette=palette, inner="box", cut=0,
        ax=ax_left,
    )
    ax_left.set_title("Qwen3-1.7B")
    ax_left.set_xlabel("")
    ax_left.set_ylabel("answer mean LID")
    ax_left.tick_params(axis="x", rotation=20)
    for tick in ax_left.get_xticklabels():
        tick.set_horizontalalignment("right")

    rows = []
    for (model_label, pv), sub in exp_b.groupby(["model_label", "prompt_variant"]):
        vals = sub["mean_lid"].to_numpy()
        if vals.size == 0:
            continue
        lo, hi = boot_ci(vals)
        rows.append({
            "model_label": model_label,
            "prompt_variant": pv,
            "prompt_label": PROMPT_LABEL[pv],
            "mean": float(np.mean(vals)),
            "ci_low": lo, "ci_high": hi, "n": int(vals.size),
        })
    summary = pd.DataFrame(rows)

    for i, pv in enumerate(PROMPT_ORDER):
        pv_label = PROMPT_LABEL[pv]
        sub = summary[summary["prompt_variant"] == pv]
        x = np.array([MODEL_ORDER.index(m) for m in sub["model_label"]], dtype=float) + (i - 1) * 0.15
        ax_right.errorbar(
            x, sub["mean"], yerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
            fmt="o", color=palette[pv_label], label=pv_label,
            markersize=5, elinewidth=1.2, capsize=2.5,
        )
    ax_right.set_xticks(range(len(MODEL_ORDER)))
    ax_right.set_xticklabels(MODEL_ORDER)
    ax_right.set_title(f"All Qwen3 sizes, layer {layer}")
    ax_right.set_xlabel("model size")
    ax_right.set_ylabel("answer mean LID")
    ax_right.legend(loc="lower right", ncol=3, fontsize=8)

    fig.suptitle(f"Experiment B: prompt-structure robustness (layer {layer})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------- Exp C C.1 figure
def figure_exp_c_segments(df: pd.DataFrame, layer: int, out_path: Path) -> None:
    exp_c = df[(df["experiment"] == "exp_c") & (df["layer"] == layer)].copy()
    if exp_c.empty:
        return
    exp_c = exp_c[exp_c["phase"].notna()]
    exp_c["phase_label"] = exp_c["phase"].map(SEGMENT_LABEL)
    palette = {
        SEGMENT_LABEL["thinking"]: "#2ca02c",
        SEGMENT_LABEL["think_answer"]: "#1f77b4",
        SEGMENT_LABEL["no_think_answer"]: "#ff7f0e",
    }
    order = [SEGMENT_LABEL[s] for s in SEGMENT_ORDER]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.1, 1.9]})

    # Left: Qwen3-1.7B per-segment distributions
    m = exp_c[exp_c["model_label"] == "1.7B"]
    sns.violinplot(
        data=m, x="phase_label", y="mean_lid",
        order=order, palette=palette, inner="box", cut=0,
        ax=ax_left,
    )
    ax_left.set_title("Qwen3-1.7B")
    ax_left.set_xlabel("")
    ax_left.set_ylabel("mean LID")
    ax_left.tick_params(axis="x", rotation=15)
    for tick in ax_left.get_xticklabels():
        tick.set_horizontalalignment("right")

    # Right: across-scale point + 95% CI per segment
    rows = []
    for (model_label, phase), sub in exp_c.groupby(["model_label", "phase"]):
        vals = sub["mean_lid"].to_numpy()
        if vals.size == 0:
            continue
        lo, hi = boot_ci(vals)
        rows.append({
            "model_label": model_label,
            "phase": phase,
            "phase_label": SEGMENT_LABEL[phase],
            "mean": float(np.mean(vals)),
            "ci_low": lo, "ci_high": hi, "n": int(vals.size),
        })
    summary = pd.DataFrame(rows)

    for i, phase in enumerate(SEGMENT_ORDER):
        ph_label = SEGMENT_LABEL[phase]
        sub = summary[summary["phase"] == phase]
        x = np.array([MODEL_ORDER.index(m) for m in sub["model_label"]], dtype=float) + (i - 1) * 0.16
        ax_right.errorbar(
            x, sub["mean"], yerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
            fmt="o", color=palette[ph_label], label=ph_label,
            markersize=5, elinewidth=1.2, capsize=2.5,
        )
    ax_right.set_xticks(range(len(MODEL_ORDER)))
    ax_right.set_xticklabels(MODEL_ORDER)
    ax_right.set_title(f"All Qwen3 sizes, layer {layer}")
    ax_right.set_xlabel("model size")
    ax_right.set_ylabel("mean LID")
    ax_right.legend(loc="lower right", ncol=3, fontsize=8)

    fig.suptitle(f"Experiment C (C.1): segment-level mean LID, layer {layer}", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------- Exp C C.2 figure
def figure_exp_c_correctness(df: pd.DataFrame, layer: int, out_path: Path) -> None:
    exp_c = df[(df["experiment"] == "exp_c") & (df["layer"] == layer)].copy()
    if exp_c.empty:
        return
    exp_c = exp_c[exp_c["phase"].notna() & exp_c["correct_bool"].notna()]
    exp_c["phase_label"] = exp_c["phase"].map(SEGMENT_LABEL)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    palette = {True: "#2ca02c", False: "#d62728"}
    legend_handles = []

    for ax, phase in zip(axes, SEGMENT_ORDER):
        ph_df = exp_c[exp_c["phase"] == phase]
        rows = []
        for model_label in MODEL_ORDER:
            for is_correct in [True, False]:
                vals = ph_df.loc[
                    (ph_df["model_label"] == model_label) & (ph_df["correct_bool"] == is_correct),
                    "mean_lid",
                ].to_numpy()
                if vals.size < 3:
                    rows.append({
                        "model_label": model_label, "correct": is_correct,
                        "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": int(vals.size),
                    })
                    continue
                lo, hi = boot_ci(vals)
                rows.append({
                    "model_label": model_label, "correct": is_correct,
                    "mean": float(np.mean(vals)), "ci_low": lo, "ci_high": hi, "n": int(vals.size),
                })
        sub = pd.DataFrame(rows)
        for is_correct in [True, False]:
            piece = sub[sub["correct"] == is_correct].dropna(subset=["mean"])
            if piece.empty:
                continue
            x = np.array([MODEL_ORDER.index(m) for m in piece["model_label"]], dtype=float)
            x = x + (0.12 if is_correct else -0.12)
            color = palette[is_correct]
            label = "correct" if is_correct else "wrong"
            h = ax.errorbar(
                x, piece["mean"],
                yerr=[piece["mean"] - piece["ci_low"], piece["ci_high"] - piece["mean"]],
                fmt="o", color=color, markersize=5, elinewidth=1.2, capsize=2.5, label=label,
            )
            if ax is axes[0]:
                legend_handles.append(h)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_title(SEGMENT_LABEL[phase])
        ax.set_xlabel("model size")

    axes[0].set_ylabel("mean LID")
    axes[0].legend(handles=legend_handles, labels=["correct", "wrong"], loc="lower right", fontsize=9)

    fig.suptitle(
        f"Experiment C (C.2): mean LID by final-answer correctness, layer {layer}",
        y=1.03, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- Exp C correctness gap headline
def figure_correctness_gap(df: pd.DataFrame, out_path: Path) -> None:
    """Headline: correct-minus-wrong median LID, by phase x model, layer 13."""
    layer = 13
    exp_c = df[(df["experiment"] == "exp_c") & (df["layer"] == layer)].copy()
    exp_c = exp_c[exp_c["phase"].notna() & exp_c["correct_bool"].notna()]

    rows = []
    for model_label in MODEL_ORDER:
        for phase in SEGMENT_ORDER:
            sub = exp_c[(exp_c["model_label"] == model_label) & (exp_c["phase"] == phase)]
            cval = sub.loc[sub["correct_bool"] == True, "mean_lid"].to_numpy()
            wval = sub.loc[sub["correct_bool"] == False, "mean_lid"].to_numpy()
            if cval.size < 3 or wval.size < 3:
                gap = np.nan
            else:
                gap = float(np.median(cval) - np.median(wval))
            rows.append({"model_label": model_label, "phase": phase, "gap": gap,
                         "n_correct": int(cval.size), "n_wrong": int(wval.size)})
    gaps = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    palette = {
        SEGMENT_LABEL["thinking"]: "#2ca02c",
        SEGMENT_LABEL["think_answer"]: "#1f77b4",
        SEGMENT_LABEL["no_think_answer"]: "#ff7f0e",
    }
    for i, phase in enumerate(SEGMENT_ORDER):
        sub = gaps[gaps["phase"] == phase].copy().dropna(subset=["gap"])
        if sub.empty:
            continue
        x = np.array([MODEL_ORDER.index(m) for m in sub["model_label"]], dtype=float) + (i - 1) * 0.18
        ax.bar(
            x, sub["gap"], width=0.16,
            color=palette[SEGMENT_LABEL[phase]],
            edgecolor="black", linewidth=0.4,
            label=SEGMENT_LABEL[phase],
        )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_xlabel("model size")
    ax.set_ylabel(r"median LID gap (correct $-$ wrong)")
    ax.set_title("Correctness gap by phase, layer 13")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics", type=Path,
        default=Path("outputs/mechanism_visualizations/data/all_sample_metrics.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("report/Images"))
    p.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    df = load(args.metrics)

    for layer in args.layers:
        figure_exp_a(df, layer, args.out_dir / f"fig_exp_a_layer{layer}.png")
        figure_exp_b(df, layer, args.out_dir / f"fig_exp_b_layer{layer}.png")
        figure_exp_c_segments(df, layer, args.out_dir / f"fig_exp_c_segments_layer{layer}.png")
        figure_exp_c_correctness(df, layer, args.out_dir / f"fig_exp_c_correctness_layer{layer}.png")
        print(f"layer {layer}: 4 figures written")

    figure_correctness_gap(df, args.out_dir / "fig_exp_c_correctness_gap.png")
    print(f"wrote {args.out_dir / 'fig_exp_c_correctness_gap.png'}")


if __name__ == "__main__":
    main()
