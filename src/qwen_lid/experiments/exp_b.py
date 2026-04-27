from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from qwen_lid.answer_parsing import evaluate_correctness
from qwen_lid.config import get_sampling_config
from qwen_lid.distractors import get_distractor
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
from qwen_lid.plotting import plot_exp_b
from qwen_lid.prompts import build_exp_c_prompt
from qwen_lid.seeding import set_seed
from qwen_lid.segmentation import segment_generation
from qwen_lid.stats import summarize_values


def _pair_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return []


def _summary_rows(sample_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_df = pd.DataFrame(sample_rows)
    if not sample_df.empty:
        answer_once = sample_df[sample_df["segment"] == "answer_segment"].drop_duplicates(
            ["example_id", "prompt_variant", "mode"]
        )
        for (variant, mode), group in answer_once.groupby(["prompt_variant", "mode"]):
            rows.append(
                {
                    "summary_type": "accuracy",
                    "prompt_variant": variant,
                    "mode": mode,
                    "accuracy": float(group["correct"].mean()) if group["correct"].notna().any() else None,
                    "mean_answer_tokens": float(group["token_count"].mean()) if len(group) else None,
                    "n": int(len(group)),
                }
            )
    return rows


def run_exp_b(config: dict[str, Any], n_samples: int | None = None, resume: bool = True) -> None:
    set_seed(int(config.get("seed", 1234)))
    k = int(config.get("k", 10))
    layers = [int(layer) for layer in config.get("layers", [6, 13, 20])]
    normalize = bool(config.get("normalize_hidden_states", False))
    n_samples = int(config.get("n_samples", 100) if n_samples is None else n_samples)
    output_paths = prepare_experiment_dirs(config.get("output_dir", "outputs/exp_b"))
    raw_path = output_paths["root"] / "raw_generations.jsonl"
    records_by_condition = existing_records_by_condition(raw_path)
    examples = load_gsm8k_examples(
        split=config.get("split", "test"),
        n_samples=n_samples,
        seed=int(config.get("seed", 1234)),
        dataset_name=config.get("dataset_name", "openai/gsm8k"),
        dataset_config=config.get("dataset_config", "main"),
    )

    wrapper = load_model(config.get("model_id", "Qwen/Qwen3-1.7B"))
    variants = list(config.get("prompt_variants", ["canonical", "irrelevant_context", "repetitive_filler"]))
    modes = ["no_think"]
    target_condition_ids: list[str] = []
    for example in tqdm(examples, desc="Experiment B examples"):
        for variant in variants:
            user_content = build_exp_c_prompt(example.question, variant, example.source_index)
            for mode in modes:
                condition_id = f"exp_b__{example.example_id}__{variant}__{mode}"
                target_condition_ids.append(condition_id)
                if resume and condition_id in records_by_condition and hidden_file_exists(records_by_condition[condition_id]):
                    continue
                sampling = get_sampling_config(config, config.get("sampling_profile", "official_recommended"), mode)
                result = wrapper.generate_with_hidden_states(
                    user_content=user_content,
                    enable_thinking=(mode == "think"),
                    selected_layers=layers,
                    sampling_config=sampling,
                )
                segments = segment_generation(
                    result.decoded_text,
                    result.decoded_prefixes,
                    mode,
                    generated_token_ids=result.generated_token_ids,
                    tokenizer=wrapper.tokenizer,
                )
                correctness = evaluate_correctness(example.gold_answer, segments["answer_segment"].text)
                hidden_path = output_paths["hidden"] / f"{condition_id}.npz"
                metadata = {"source_index": example.source_index}
                if variant == "irrelevant_context":
                    metadata["distractor"] = get_distractor(example.source_index)
                record = record_generation(
                    result,
                    experiment="exp_b",
                    condition_id=condition_id,
                    example_id=example.example_id,
                    mode=mode,
                    prompt_variant=variant,
                    user_content=user_content,
                    segments=segments,
                    hidden_path=hidden_path,
                    raw_path=raw_path,
                    correctness=correctness,
                    metadata=metadata,
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
    pairs = _pair_rows(sample_rows)
    summaries = _summary_rows(sample_rows, pairs, seed=int(config.get("seed", 1234)))
    write_metrics(output_paths["root"], sample_rows, pairs, summaries)
    write_manifest(
        output_paths["root"],
        experiment="exp_b",
        config=config,
        model_metadata={"device": str(wrapper.device_info.device), "dtype": wrapper.device_info.dtype_name},
        sample_counts={"n_samples": n_samples, "conditions": len(records)},
    )
    plot_exp_b(output_paths["root"])
