from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from qwen_lid.hidden_states import load_hidden_states, save_hidden_states
from qwen_lid.io_utils import append_jsonl, ensure_dir, get_git_commit, read_jsonl, save_json, utc_timestamp, write_jsonl
from qwen_lid.lid import compute_token_lid_with_mask
from qwen_lid.schemas import GenerationResult, SegmentInfo


def choose_lid_device(preferred: str | torch.device | None = None) -> torch.device:
    if preferred is not None:
        device = torch.device(preferred)
        if device.type == "cuda" and torch.cuda.is_available():
            return device
        if device.type == "mps" and torch.backends.mps.is_available():
            return device
        if device.type == "cpu":
            return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_experiment_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = ensure_dir(output_dir)
    paths = {
        "root": root,
        "hidden": ensure_dir(root / "hidden_states"),
        "figures": ensure_dir(root / "figures"),
    }
    return paths


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    size = max(1, int(batch_size))
    return [items[index : index + size] for index in range(0, len(items), size)]


def existing_records_by_condition(raw_path: str | Path) -> dict[str, dict[str, Any]]:
    records = {}
    for record in read_jsonl(raw_path):
        condition_id = record.get("condition_id")
        if condition_id:
            records[condition_id] = record
    return records


def hidden_file_exists(record: dict[str, Any]) -> bool:
    path = record.get("hidden_state_path")
    return bool(path) and Path(path).exists()


def record_generation(
    result: GenerationResult,
    *,
    experiment: str,
    condition_id: str,
    example_id: str,
    mode: str,
    prompt_variant: str,
    user_content: str,
    segments: dict[str, SegmentInfo],
    hidden_path: Path,
    raw_path: Path,
    family: str | None = None,
    correctness: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    save_hidden_states(hidden_path, result.hidden_states)
    record = result.to_record()
    record.update(
        {
            "experiment": experiment,
            "condition_id": condition_id,
            "example_id": example_id,
            "mode": mode,
            "prompt_variant": prompt_variant,
            "family": family,
            "user_content": user_content,
            "segments": {name: segment.to_dict() for name, segment in segments.items()},
            "correctness": correctness or {},
            "runtime_metadata": result.metadata,
            "decoding_config": result.metadata.get("decoding_config", {}),
            "hidden_state_path": str(hidden_path),
            "metadata": metadata or {},
        }
    )
    append_jsonl(raw_path, record)
    return record


def compute_lid_metric_rows(
    records: list[dict[str, Any]],
    *,
    k: int,
    normalize: bool,
    lid_device: str | torch.device | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = choose_lid_device(lid_device)
    for record in records:
        hidden_path = record.get("hidden_state_path")
        if not hidden_path or not Path(hidden_path).exists():
            continue
        hidden_states = load_hidden_states(hidden_path)
        hidden_tensors = {
            layer: torch.as_tensor(values, dtype=torch.float32, device=device)
            for layer, values in hidden_states.items()
        }
        correctness = record.get("correctness", {})
        for segment_name, segment in record.get("segments", {}).items():
            token_start = int(segment["token_start"])
            token_end = int(segment["token_end"])
            for layer, layer_hidden in hidden_tensors.items():
                segment_hidden = layer_hidden[token_start:token_end]
                lid_result = compute_token_lid_with_mask(segment_hidden, k=k, normalize=normalize, device=device)
                valid = bool(lid_result.valid_mask.any())
                mean_lid = float(lid_result.values[lid_result.valid_mask].mean()) if valid else None
                rows.append(
                    {
                        "experiment": record.get("experiment"),
                        "condition_id": record.get("condition_id"),
                        "example_id": record.get("example_id"),
                        "mode": record.get("mode"),
                        "prompt_variant": record.get("prompt_variant"),
                        "family": record.get("family"),
                        "segment": segment_name,
                        "layer": int(layer),
                        "k": int(k),
                        "normalize_hidden_states": bool(normalize),
                        "token_count": int(segment.get("token_count", max(0, token_end - token_start))),
                        "valid_token_count": int(lid_result.valid_mask.sum()),
                        "valid": valid,
                        "mean_lid": mean_lid,
                        "invalid_reason": None if valid else lid_result.reason,
                        "segment_parse_status": segment.get("parse_status"),
                        "correct": correctness.get("correct"),
                        "answer_parse_status": correctness.get("parse_status"),
                        "gold_answer": correctness.get("gold_answer"),
                        "model_answer": correctness.get("model_answer"),
                    }
                )
    return rows


def _write_table(path: Path, rows: list[dict[str, Any]], fallback_columns: list[str]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=fallback_columns).to_csv(path, index=False)


def write_metrics(output_dir: Path, sample_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    _write_table(output_dir / "sample_metrics.csv", sample_rows, ["experiment"])
    _write_table(output_dir / "pair_metrics.csv", pair_rows, ["experiment", "layer", "valid_pair"])
    _write_table(output_dir / "summary_metrics.csv", summary_rows, ["summary_type"])


def write_manifest(
    output_dir: Path,
    *,
    experiment: str,
    config: dict[str, Any],
    model_metadata: dict[str, Any],
    sample_counts: dict[str, Any],
) -> None:
    manifest = {
        "timestamp": utc_timestamp(),
        "git_commit": get_git_commit(),
        "experiment": experiment,
        "model_id": config.get("model_id"),
        "device": model_metadata.get("device"),
        "dtype": model_metadata.get("dtype"),
        "config": config,
        "sample_counts": sample_counts,
        "selected_layers": config.get("layers"),
        "seed": config.get("seed"),
    }
    save_json(output_dir / "run_manifest.json", manifest)


def finalize_raw_jsonl(raw_path: Path, records_by_condition: dict[str, dict[str, Any]]) -> None:
    records = [records_by_condition[key] for key in sorted(records_by_condition)]
    write_jsonl(raw_path, records)
