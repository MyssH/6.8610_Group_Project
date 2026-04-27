from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LIDResult:
    values: np.ndarray
    valid_mask: np.ndarray
    reason: str | None = None


def _prepare_hidden_states(hidden_states: np.ndarray, normalize: bool) -> np.ndarray:
    array = np.asarray(hidden_states, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected hidden_states with shape [tokens, dim], got {array.shape}")
    if normalize and len(array):
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        array = array / norms
    return array


def compute_token_lid_with_mask(
    hidden_states: np.ndarray,
    k: int,
    normalize: bool = False,
    min_extra_tokens: int = 2,
) -> LIDResult:
    """Compute token-level Levina-Bickel LID estimates.

    Invalid tokens receive NaN and are masked out. A segment shorter than
    k + min_extra_tokens is treated as invalid by design.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    x = _prepare_hidden_states(hidden_states, normalize=normalize)
    n_tokens = x.shape[0]
    values = np.full(n_tokens, np.nan, dtype=np.float32)
    valid = np.zeros(n_tokens, dtype=bool)
    if n_tokens < k + min_extra_tokens:
        return LIDResult(values=values, valid_mask=valid, reason="too_short")
    if n_tokens - 1 < k:
        return LIDResult(values=values, valid_mask=valid, reason="not_enough_neighbors")

    diff = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32), dtype=np.float32)
    np.fill_diagonal(distances, np.inf)
    sorted_distances = np.sort(distances, axis=1)
    eps = np.finfo(np.float32).eps

    for idx in range(n_tokens):
        neighbors = sorted_distances[idx, :k]
        r_k = neighbors[k - 1]
        r_j = neighbors[: k - 1]
        if not np.isfinite(r_k) or r_k <= eps:
            continue
        if np.any(~np.isfinite(r_j)) or np.any(r_j <= eps):
            continue
        logs = np.log(r_k / r_j)
        denominator = float(np.mean(logs))
        if not np.isfinite(denominator) or denominator <= eps:
            continue
        values[idx] = float(1.0 / denominator)
        valid[idx] = True

    reason = None if bool(valid.any()) else "no_valid_tokens"
    return LIDResult(values=values, valid_mask=valid, reason=reason)


def compute_token_lid(hidden_states: np.ndarray, k: int, normalize: bool = False) -> np.ndarray:
    return compute_token_lid_with_mask(hidden_states, k=k, normalize=normalize).values


def compute_mean_lid(hidden_states: np.ndarray, k: int, normalize: bool = False) -> float | None:
    result = compute_token_lid_with_mask(hidden_states, k=k, normalize=normalize)
    if not result.valid_mask.any():
        return None
    return float(np.nanmean(result.values[result.valid_mask]))
