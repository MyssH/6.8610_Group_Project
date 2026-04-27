#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lid.plotting import regenerate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate all plots from saved metrics artifacts.")
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"), help="Root output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = regenerate_all_plots(args.outputs_root)
    print(f"Regenerated {len(created)} plot files.")


if __name__ == "__main__":
    main()
