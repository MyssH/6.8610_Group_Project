#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lid.config import load_config
from qwen_lid.experiments.exp_c import run_exp_c
from qwen_lid.paths import ensure_project_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment C GSM8K canonical think vs no-think LID comparison.")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of GSM8K test examples.")
    parser.add_argument("--k", type=int, choices=[5, 10, 20], default=None, help="Neighbor count for LID.")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="0-based transformer block indices.")
    parser.add_argument("--sampling-profile", default=None, choices=["matched_main", "official_recommended"], help="Sampling profile.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output-dir", default=None, help="Experiment output directory.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model id.")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of prompts to generate in parallel.")
    parser.add_argument("--normalize-hidden-states", action="store_true", help="Use L2-normalized hidden states for LID.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max generated tokens.")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate records even when saved outputs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    config = load_config("exp_c")
    if args.n_samples is not None:
        config["n_samples"] = args.n_samples
    if args.k is not None:
        config["k"] = args.k
    if args.layers is not None:
        config["layers"] = args.layers
    if args.sampling_profile is not None:
        config["sampling_profile"] = args.sampling_profile
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.model_id is not None:
        config["model_id"] = args.model_id
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.normalize_hidden_states:
        config["normalize_hidden_states"] = True
    if args.max_new_tokens is not None:
        profile = config["sampling_profiles"][config["sampling_profile"]]
        if "think" in profile or "no_think" in profile:
            for mode_config in profile.values():
                if isinstance(mode_config, dict):
                    mode_config["max_new_tokens"] = args.max_new_tokens
        else:
            profile["max_new_tokens"] = args.max_new_tokens
    run_exp_c(config, n_samples=config["n_samples"], resume=not args.no_resume)


if __name__ == "__main__":
    main()
