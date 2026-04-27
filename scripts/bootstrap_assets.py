#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset
from huggingface_hub import snapshot_download

from qwen_lid.io_utils import save_json, utc_timestamp
from qwen_lid.paths import ensure_project_dirs, gsm8k_disk_path, model_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen3 model assets and cache GSM8K locally.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B", help="Hugging Face model id to download.")
    parser.add_argument("--dataset-name", default="openai/gsm8k", help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default="main", help="Dataset config name.")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download.")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    manifest: dict[str, object] = {
        "timestamp": utc_timestamp(),
        "model_id": args.model_id,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
    }

    if not args.skip_model:
        model_path = model_cache_path(args.model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model assets for {args.model_id} to {model_path}")
        snapshot_download(repo_id=args.model_id, local_dir=str(model_path))
        manifest["model_path"] = str(model_path)
        print("Model assets are available locally.")

    if not args.skip_dataset:
        dataset_path = gsm8k_disk_path()
        print(f"Downloading dataset {args.dataset_name}/{args.dataset_config} to {dataset_path}")
        dataset = load_dataset(args.dataset_name, args.dataset_config)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(dataset_path))
        manifest["dataset_path"] = str(dataset_path)
        manifest["dataset_splits"] = {split: len(dataset[split]) for split in dataset}
        print("GSM8K dataset is available locally.")

    save_json(ROOT / "data" / "processed" / "bootstrap_manifest.json", manifest)
    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
