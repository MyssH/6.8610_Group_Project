"""Wilcoxon tests supporting Exp C C.1 and C.2 claims.

C.1: across model sizes the relative ordering of mean LID across the thinking,
think-answer, and no-think-answer segments is not stable. We run paired
Wilcoxon signed-rank tests per model on segment-wise differences, on the same
example_id (paired by problem, joined across modes).

C.2: thinking-phase mean LID is higher for correct than incorrect answers. We
run a one-sided Mann-Whitney U test per model.

Input: outputs/mechanism_visualizations/data/all_sample_metrics.csv
Output: outputs/mechanism_visualizations/data/wilcoxon_exp_c.csv
        outputs/mechanism_visualizations/data/wilcoxon_exp_c_summary.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu


def load_exp_c(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["experiment"] == "exp_c"].copy()
    df = df[df["valid"].astype(str).str.lower() == "true"]
    df = df[df["mean_lid"].notna()]
    df["correct_bool"] = df["correct"].map({1.0: True, 0.0: False, True: True, False: False})
    df["phase"] = df.apply(
        lambda row: (
            "thinking"
            if row["mode"] == "think" and row["segment"] == "thinking_segment"
            else "think_answer"
            if row["mode"] == "think" and row["segment"] == "answer_segment"
            else "no_think_answer"
            if row["mode"] == "no_think" and row["segment"] == "answer_segment"
            else None
        ),
        axis=1,
    )
    df = df[df["phase"].notna()]
    return df


def c1_segment_paired(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_label, layer), grp in df.groupby(["model_label", "layer"]):
        wide = grp.pivot_table(
            index="example_id", columns="phase", values="mean_lid", aggfunc="first"
        )
        for a, b in [
            ("thinking", "think_answer"),
            ("thinking", "no_think_answer"),
            ("think_answer", "no_think_answer"),
        ]:
            if a not in wide.columns or b not in wide.columns:
                continue
            paired = wide[[a, b]].dropna()
            if len(paired) < 5:
                continue
            diffs = paired[a].values - paired[b].values
            try:
                stat, p_two = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                stat, p_two = np.nan, np.nan
            median_diff = float(np.median(diffs))
            rows.append(
                {
                    "model_label": model_label,
                    "layer": int(layer),
                    "test": "C1_paired_wilcoxon",
                    "comparison": f"{a} - {b}",
                    "n_pairs": int(len(paired)),
                    "median_diff": median_diff,
                    "mean_diff": float(np.mean(diffs)),
                    "wilcoxon_W": float(stat) if stat is not None else np.nan,
                    "p_two_sided": float(p_two) if p_two is not None else np.nan,
                    "direction": "+" if median_diff > 0 else ("-" if median_diff < 0 else "0"),
                }
            )
    return pd.DataFrame(rows)


def c2_correctness_thinking(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_label, layer), grp in df.groupby(["model_label", "layer"]):
        for phase in ["thinking", "think_answer", "no_think_answer"]:
            phase_df = grp[(grp["phase"] == phase) & grp["correct_bool"].notna()]
            correct = phase_df.loc[phase_df["correct_bool"] == True, "mean_lid"].dropna().values
            wrong = phase_df.loc[phase_df["correct_bool"] == False, "mean_lid"].dropna().values
            if len(correct) < 3 or len(wrong) < 3:
                rows.append(
                    {
                        "model_label": model_label,
                        "layer": int(layer),
                        "test": "C2_mannwhitney_u_one_sided",
                        "phase": phase,
                        "n_correct": int(len(correct)),
                        "n_wrong": int(len(wrong)),
                        "median_correct": float(np.median(correct)) if len(correct) else np.nan,
                        "median_wrong": float(np.median(wrong)) if len(wrong) else np.nan,
                        "median_gap": (float(np.median(correct)) - float(np.median(wrong)))
                        if len(correct) and len(wrong)
                        else np.nan,
                        "U": np.nan,
                        "p_one_sided_correct_gt_wrong": np.nan,
                        "note": "insufficient_samples",
                    }
                )
                continue
            stat, p = mannwhitneyu(correct, wrong, alternative="greater")
            rows.append(
                {
                    "model_label": model_label,
                    "layer": int(layer),
                    "test": "C2_mannwhitney_u_one_sided",
                    "phase": phase,
                    "n_correct": int(len(correct)),
                    "n_wrong": int(len(wrong)),
                    "median_correct": float(np.median(correct)),
                    "median_wrong": float(np.median(wrong)),
                    "median_gap": float(np.median(correct) - np.median(wrong)),
                    "U": float(stat),
                    "p_one_sided_correct_gt_wrong": float(p),
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/mechanism_visualizations/data/all_sample_metrics.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/mechanism_visualizations/data"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_exp_c(args.metrics)
    print(f"Loaded {len(df)} exp_c rows across {df['model_label'].nunique()} models")

    c1 = c1_segment_paired(df)
    c2 = c2_correctness_thinking(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    c1.to_csv(args.out_dir / "wilcoxon_exp_c_c1.csv", index=False)
    c2.to_csv(args.out_dir / "wilcoxon_exp_c_c2.csv", index=False)

    # Print summary for layer 13 (the main reported layer)
    print("\n=== C.1: paired Wilcoxon on segment differences (layer 13) ===")
    print(
        c1[c1["layer"] == 13]
        .sort_values(["model_label", "comparison"])
        .to_string(index=False)
    )
    print("\n=== C.2: Mann-Whitney U, thinking phase, correct vs wrong (layer 13) ===")
    print(
        c2[(c2["layer"] == 13) & (c2["phase"] == "thinking")]
        .sort_values("model_label")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
