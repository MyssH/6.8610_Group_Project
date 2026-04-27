from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from qwen_lid.config import get_sampling_config
from qwen_lid.experiments.common import (
    compute_lid_metric_rows,
    existing_records_by_condition,
    finalize_raw_jsonl,
    hidden_file_exists,
    prepare_experiment_dirs,
    record_generation,
    write_manifest,
    write_metrics,
)
from qwen_lid.gsm8k import load_gsm8k_examples
from qwen_lid.model_loader import load_model
from qwen_lid.plotting import plot_exp_a
from qwen_lid.prompts import PromptCase, build_exp_a_degenerate_cases, build_gsm8k_canonical_prompt
from qwen_lid.schemas import SegmentInfo
from qwen_lid.seeding import set_seed
from qwen_lid.stats import summarize_values


def _build_cases(config: dict[str, Any], n_baseline: int) -> list[PromptCase]:
    cases = build_exp_a_degenerate_cases()
    if n_baseline > 0:
        examples = load_gsm8k_examples(
            split=config.get("split", "test"),
            n_samples=n_baseline,
            seed=int(config.get("seed", 1234)),
            dataset_name=config.get("dataset_name", "openai/gsm8k"),
            dataset_config=config.get("dataset_config", "main"),
        )
        for example in examples:
            cases.append(
                PromptCase(
                    case_id=f"regular_gsm8k_nothink_{example.example_id}",
                    family="regular_gsm8k_nothink",
                    user_content=build_gsm8k_canonical_prompt(example.question),
                    metadata={
                        "example_id": example.example_id,
                        "source_index": str(example.source_index),
                    },
                )
            )
    return cases


def _summary_rows(sample_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    df = pd.DataFrame(sample_rows)
    if df.empty:
        return []
    df = df[(df["segment"] == "full_output") & (df["valid"] == True)]
    rows: list[dict[str, Any]] = []
    for (family, layer), group in df.groupby(["family", "layer"], dropna=False):
        summary = summarize_values(group["mean_lid"], seed=seed)
        rows.append(
            {
                "summary_type": "family_layer",
                "family": family,
                "layer": int(layer),
                **summary,
            }
        )
    return rows


def run_exp_a(config: dict[str, Any], n_baseline: int | None = None, resume: bool = True) -> None:
    set_seed(int(config.get("seed", 1234)))
    k = int(config.get("k", 10))
    layers = [int(layer) for layer in config.get("layers", [6, 13, 20])]
    normalize = bool(config.get("normalize_hidden_states", False))
    n_baseline = int(config.get("n_baseline", 30) if n_baseline is None else n_baseline)
    output_paths = prepare_experiment_dirs(config.get("output_dir", "outputs/exp_a"))
    raw_path = output_paths["root"] / "raw_generations.jsonl"
    records_by_condition = existing_records_by_condition(raw_path)
    cases = _build_cases(config, n_baseline)

    wrapper = load_model(config.get("model_id", "Qwen/Qwen3-1.7B"))
    sampling = get_sampling_config(config, config.get("sampling_profile", "official_recommended"), "no_think")
    target_condition_ids: list[str] = []

    for case in tqdm(cases, desc="Experiment A"):
        condition_id = f"exp_a__{case.case_id}"
        target_condition_ids.append(condition_id)
        if resume and condition_id in records_by_condition and hidden_file_exists(records_by_condition[condition_id]):
            continue
        result = wrapper.generate_with_hidden_states(
            user_content=case.user_content,
            enable_thinking=False,
            selected_layers=layers,
            sampling_config=sampling,
        )
        segments = {
            "full_output": SegmentInfo(
                name="full_output",
                text=result.decoded_text,
                token_start=0,
                token_end=len(result.generated_token_ids),
                parse_status="ok",
            )
        }
        hidden_path = output_paths["hidden"] / f"{condition_id}.npz"
        record = record_generation(
            result,
            experiment="exp_a",
            condition_id=condition_id,
            example_id=case.metadata.get("example_id", case.case_id),
            mode="no_think",
            prompt_variant="degenerate" if case.family != "regular_gsm8k_nothink" else "canonical",
            family=case.family,
            user_content=case.user_content,
            segments=segments,
            hidden_path=hidden_path,
            raw_path=raw_path,
            metadata=case.metadata,
        )
        records_by_condition[condition_id] = record

    current_records_by_condition = {
        condition_id: records_by_condition[condition_id]
        for condition_id in target_condition_ids
        if condition_id in records_by_condition
    }
    finalize_raw_jsonl(raw_path, current_records_by_condition)
    records = list(current_records_by_condition.values())
    sample_rows = compute_lid_metric_rows(records, k=k, normalize=normalize)
    summary_rows = _summary_rows(sample_rows, seed=int(config.get("seed", 1234)))
    write_metrics(output_paths["root"], sample_rows, [], summary_rows)
    write_manifest(
        output_paths["root"],
        experiment="exp_a",
        config=config,
        model_metadata={"device": str(wrapper.device_info.device), "dtype": wrapper.device_info.dtype_name},
        sample_counts={"cases": len(cases), "n_baseline": n_baseline},
    )
    plot_exp_a(output_paths["root"])
