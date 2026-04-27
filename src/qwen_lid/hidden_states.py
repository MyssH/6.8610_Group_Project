from __future__ import annotations

from pathlib import Path

import numpy as np


def save_hidden_states(path: str | Path, hidden_states: dict[int, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"layer_{layer}": np.asarray(values, dtype=np.float32) for layer, values in hidden_states.items()}
    payload["layers"] = np.asarray(sorted(hidden_states), dtype=np.int32)
    np.savez_compressed(path, **payload)


def load_hidden_states(path: str | Path) -> dict[int, np.ndarray]:
    loaded = np.load(Path(path), allow_pickle=False)
    layers = [int(layer) for layer in loaded["layers"].tolist()]
    return {layer: loaded[f"layer_{layer}"].astype(np.float32, copy=False) for layer in layers}


def slice_segment(hidden_states: dict[int, np.ndarray], token_start: int, token_end: int) -> dict[int, np.ndarray]:
    return {layer: values[token_start:token_end] for layer, values in hidden_states.items()}
