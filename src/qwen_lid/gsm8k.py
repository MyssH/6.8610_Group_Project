from __future__ import annotations

from pathlib import Path

import numpy as np

from qwen_lid.answer_parsing import parse_gsm8k_gold
from qwen_lid.paths import gsm8k_disk_path
from qwen_lid.schemas import GSM8KExample


def load_or_download_gsm8k(
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    local_path: str | Path | None = None,
):
    from datasets import load_dataset, load_from_disk

    path = Path(local_path) if local_path is not None else gsm8k_disk_path()
    if path.exists():
        return load_from_disk(str(path))
    dataset = load_dataset(dataset_name, dataset_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    return dataset


def load_gsm8k_examples(
    split: str = "test",
    n_samples: int | None = None,
    seed: int = 1234,
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    local_path: str | Path | None = None,
) -> list[GSM8KExample]:
    dataset = load_or_download_gsm8k(dataset_name, dataset_config, local_path)
    if split not in dataset:
        raise ValueError(f"Split {split!r} not found in GSM8K dataset")
    split_data = dataset[split]
    count = len(split_data)
    if n_samples is None or n_samples >= count:
        selected = list(range(count))
    else:
        rng = np.random.default_rng(seed)
        selected = sorted(int(i) for i in rng.choice(count, size=n_samples, replace=False))

    examples: list[GSM8KExample] = []
    for idx in selected:
        row = split_data[idx]
        gold = parse_gsm8k_gold(row["answer"])
        examples.append(
            GSM8KExample(
                example_id=f"gsm8k_{split}_{idx:06d}",
                question=row["question"],
                answer=row["answer"],
                gold_answer=gold["answer"],
                split=split,
                source_index=idx,
            )
        )
    return examples
