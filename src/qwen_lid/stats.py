from __future__ import annotations

from typing import Iterable

import numpy as np


def clean_values(values: Iterable[float | int | None]) -> np.ndarray:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def bootstrap_ci(
    values: Iterable[float | int | None],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 1234,
) -> tuple[float | None, float | None]:
    clean = clean_values(values)
    if clean.size == 0:
        return (None, None)
    if clean.size == 1:
        value = float(clean[0])
        return (value, value)
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(n_boot, clean.size), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(low), float(high))


def summarize_values(values: Iterable[float | int | None], seed: int = 1234) -> dict[str, float | int | None]:
    clean = clean_values(values)
    ci_low, ci_high = bootstrap_ci(clean, seed=seed)
    if clean.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def wilcoxon_p_value(differences: Iterable[float | int | None]) -> float | None:
    clean = clean_values(differences)
    if clean.size < 2:
        return None
    if np.allclose(clean, 0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(clean)
    except Exception:
        return None
    return float(result.pvalue)


def paired_difference_summary(
    left: Iterable[float | int | None],
    right: Iterable[float | int | None],
    seed: int = 1234,
) -> dict[str, float | int | None]:
    left_array = np.asarray([np.nan if value is None else value for value in left], dtype=np.float64)
    right_array = np.asarray([np.nan if value is None else value for value in right], dtype=np.float64)
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    differences = left_array[mask] - right_array[mask]
    summary = summarize_values(differences, seed=seed)
    summary["wilcoxon_p"] = wilcoxon_p_value(differences)
    return summary
