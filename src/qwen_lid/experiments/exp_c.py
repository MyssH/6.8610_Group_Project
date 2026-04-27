from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from qwen_lid.answer_parsing import evaluate_correctness
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
from qwen_lid.plotting import plot_exp_c
from qwen_lid.prompts import build_gsm8k_canonical_prompt
from qwen_lid.seeding import set_seed
from qwen_lid.segmentation import segment_generation
from qwen_lid.stats import paired_difference_summary, summarize_values


def _pair_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(sample_rows)
    if df.empty:
        return []
    answer = df[df["segment"] == "answer_segment"].copy()
    rows: list[dict[str, Any]] = []
    for (example_id, layer), group in answer.groupby(["example_id", "layer"], dropna=False):
        think = group[group["mode"] == "think"]
        no_think = group[group["mode"] == "no_think"]
        if think.empty or no_think.empty:
            continue
        t = think.iloc[0]
        n = no_think.iloc[0]
        valid_pair = bool(t["valid"]) and bool(n["valid"])
        rows.append(
            {
                "experiment": "exp_c",
                "example_id": example_id,
                "prompt_variant": "canonical",
                "layer": int(layer),
                "valid_pair": valid_pair,
                "think_answer_lid": t["mean_lid"],
                "no_think_answer_lid": n["mean_lid"],
                "answer_lid_diff": (t["mean_lid"] - n["mean_lid"]) if valid_pair else None,
                "think_correct": t["correct"],
                "no_think_correct": n["correct"],
                "think_answer_tokens": int(t["token_count"]),
                "no_think_answer_tokens": int(n["token_count"]),
            }
        )
    return rows


def _summary_rows(sample_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        for layer, group in pair_df.groupby("layer"):
            summary = summarize_values(group.loc[group["valid_pair"] == True, "answer_lid_diff"], seed=seed)
            summary["wilcoxon_p"] = paired_difference_summary(
                group.loc[group["valid_pair"] == True, "think_answer_lid"],
                group.loc[group["valid_pair"] == True, "no_think_answer_lid"],
                seed=seed,
            ).get("wilcoxon_p")
            rows.append({"summary_type": "paired_answer_layer", "layer": int(layer), **summary})

    sample_df = pd.DataFrame(sample_rows)
    if not sample_df.empty:
        answer_once = sample_df[sample_df["segment"] == "answer_segment"].drop_duplicates(["example_id", "mode"])
        for mode, group in answer_once.groupby("mode"):
            rows.append(
                {
                    "summary_type": "accuracy",
                    "mode": mode,
                    "accuracy": float(group["correct"].mean()) if group["correct"].notna().any() else None,
                    "n": int(len(group)),
                }
            )
    return rows


def _filter_by_valid_thinking_segment(
    sample_rows: list[dict[str, Any]],
    layers: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep paired examples only when the thinking segment has valid LID on every layer."""
    df = pd.DataFrame(sample_rows)
    if df.empty:
        return sample_rows, []

    expected_layers = {int(layer) for layer in layers}
    thinking = df[
        (df["mode"] == "think")
        & (df["segment"] == "thinking_segment")
        & (df["layer"].isin(expected_layers))
    ].copy()
    report_rows: list[dict[str, Any]] = []
    keep_ids: set[str] = set()
    for example_id in sorted(df["example_id"].dropna().unique()):
        group = thinking[thinking["example_id"] == example_id]
        observed_layers = {int(layer) for layer in group["layer"].dropna().unique()}
        valid_layers = {
            int(row.layer)
            for row in group.itertuples()
            if bool(row.valid)
        }
        token_counts = group["token_count"].dropna().unique().tolist()
        thinking_token_count = int(token_counts[0]) if token_counts else 0
        invalid_reasons = sorted(
            {
                str(value)
                for value in group["invalid_reason"].dropna().tolist()
                if str(value) and str(value) != "nan"
            }
        )
        keep = observed_layers == expected_layers and valid_layers == expected_layers
        if keep:
            keep_ids.add(str(example_id))
        report_rows.append(
            {
                "example_id": example_id,
                "thinking_token_count": thinking_token_count,
                "valid_layers": len(valid_layers),
                "observed_layers": len(observed_layers),
                "expected_layers": len(expected_layers),
                "invalid_reasons": ";".join(invalid_reasons),
                "keep": keep,
            }
        )

    filtered_rows = [row for row in sample_rows if str(row.get("example_id")) in keep_ids]
    return filtered_rows, report_rows


def run_exp_c(config: dict[str, Any], n_samples: int | None = None, resume: bool = True) -> None:
    set_seed(int(config.get("seed", 1234)))
    k = int(config.get("k", 10))
    layers = [int(layer) for layer in config.get("layers", [6, 13, 20])]
    normalize = bool(config.get("normalize_hidden_states", False))
    n_samples = int(config.get("n_samples", 200) if n_samples is None else n_samples)
    output_paths = prepare_experiment_dirs(config.get("output_dir", "outputs/exp_c"))
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
    modes = ["think", "no_think"]
    target_condition_ids: list[str] = []
    for example in tqdm(examples, desc="Experiment C examples"):
        user_content = build_gsm8k_canonical_prompt(example.question)
        for mode in modes:
            condition_id = f"exp_c__{example.example_id}__{mode}"
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
            record = record_generation(
                result,
                experiment="exp_c",
                condition_id=condition_id,
                example_id=example.example_id,
                mode=mode,
                prompt_variant="canonical",
                user_content=user_content,
                segments=segments,
                hidden_path=hidden_path,
                raw_path=raw_path,
                correctness=correctness,
                metadata={"source_index": example.source_index},
            )
            records_by_condition[condition_id] = record

    current_records_by_condition = {
        condition_id: records_by_condition[condition_id]
        for condition_id in target_condition_ids
        if condition_id in records_by_condition
    }
    finalize_raw_jsonl(raw_path, current_records_by_condition)
    records = list(current_records_by_condition.values())
    unfiltered_sample_rows = compute_lid_metric_rows(records, k=k, normalize=normalize)
    sample_rows, filter_report_rows = _filter_by_valid_thinking_segment(unfiltered_sample_rows, layers)
    pairs = _pair_rows(sample_rows)
    summaries = _summary_rows(sample_rows, pairs, seed=int(config.get("seed", 1234)))
    pd.DataFrame(unfiltered_sample_rows).to_csv(output_paths["root"] / "sample_metrics_unfiltered.csv", index=False)
    pd.DataFrame(filter_report_rows).to_csv(output_paths["root"] / "thinking_segment_filter.csv", index=False)
    pd.DataFrame([row for row in filter_report_rows if row["keep"]]).to_csv(output_paths["root"] / "kept_examples.csv", index=False)
    pd.DataFrame([row for row in filter_report_rows if not row["keep"]]).to_csv(output_paths["root"] / "excluded_examples.csv", index=False)
    write_metrics(output_paths["root"], sample_rows, pairs, summaries)
    write_manifest(
        output_paths["root"],
        experiment="exp_c",
        config=config,
        model_metadata={"device": str(wrapper.device_info.device), "dtype": wrapper.device_info.dtype_name},
        sample_counts={
            "n_samples": n_samples,
            "conditions": len(records),
            "unfiltered_examples": len({row["example_id"] for row in unfiltered_sample_rows}),
            "filtered_examples": len({row["example_id"] for row in sample_rows}),
            "excluded_examples": len([row for row in filter_report_rows if not row["keep"]]),
        },
    )
    plot_exp_c(output_paths["root"])
