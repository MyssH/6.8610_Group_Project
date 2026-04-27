from __future__ import annotations

import numpy as np

from qwen_lid.lid import compute_mean_lid, compute_token_lid, compute_token_lid_with_mask


def test_lid_returns_expected_shape_for_valid_data() -> None:
    rng = np.random.default_rng(123)
    hidden = rng.normal(size=(24, 8)).astype(np.float32)
    values = compute_token_lid(hidden, k=5)
    assert values.shape == (24,)
    assert np.isfinite(values).any()


def test_mean_lid_returns_none_for_too_short_segment() -> None:
    hidden = np.ones((6, 4), dtype=np.float32)
    assert compute_mean_lid(hidden, k=5) is None
    result = compute_token_lid_with_mask(hidden, k=5)
    assert result.reason == "too_short"
    assert not result.valid_mask.any()


def test_duplicate_rows_do_not_crash() -> None:
    hidden = np.vstack(
        [
            np.zeros((4, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.ones((6, 3), dtype=np.float32),
        ]
    )
    result = compute_token_lid_with_mask(hidden, k=5)
    assert result.values.shape == (13,)
    assert result.valid_mask.dtype == bool
