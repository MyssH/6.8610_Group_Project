from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lid.mechanism_visualization import MechanismConfig, run_mechanism_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory mechanism visualizations from saved Qwen LID outputs.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir-name", default="mechanism_visualizations")
    parser.add_argument("--per-model-dir-name", default="figures_mechanism")
    parser.add_argument("--exclude-model-labels", nargs="*", default=[])
    parser.add_argument("--exclude-model-names", nargs="*", default=[])
    parser.add_argument("--lid-value-normalization", choices=["none", "hidden_dim"], default="none")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lid-device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--max-lid-tokens",
        type=int,
        default=512,
        help="Subsample longer sequences for token-level LID recomputation; use 0 for all tokens.",
    )
    parser.add_argument("--max-shared-samples", type=int, default=3)
    parser.add_argument(
        "--max-exp-c-token-examples",
        type=int,
        default=80,
        help="Per-model Exp C examples used for token-level heatmaps and trajectory diagnostics; use 0 for all kept examples.",
    )
    parser.add_argument("--boundary-window", type=int, default=80)
    parser.add_argument("--normalized-bins", type=int, default=60)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-token-categories", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MechanismConfig(
        outputs_root=args.outputs_root,
        output_dir_name=args.output_dir_name,
        per_model_dir_name=args.per_model_dir_name,
        exclude_model_labels=tuple(args.exclude_model_labels),
        exclude_model_names=tuple(args.exclude_model_names),
        lid_value_normalization=args.lid_value_normalization,
        layers=tuple(args.layers),
        k=args.k,
        lid_device=args.lid_device,
        random_seed=args.seed,
        max_lid_tokens=None if args.max_lid_tokens == 0 else args.max_lid_tokens,
        max_shared_samples=args.max_shared_samples,
        max_exp_c_token_examples=args.max_exp_c_token_examples,
        boundary_window=args.boundary_window,
        normalized_bins=args.normalized_bins,
        refresh_cache=args.refresh_cache,
        token_categories=not args.no_token_categories,
    )
    result = run_mechanism_visualizations(config)
    print("Mechanism visualization pass complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
