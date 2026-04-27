from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from qwen_lid.paths import CONFIG_DIR


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(experiment_name: str | None = None) -> dict[str, Any]:
    common = load_yaml(CONFIG_DIR / "common.yaml")
    if experiment_name is None:
        return common
    exp = load_yaml(CONFIG_DIR / f"{experiment_name}.yaml")
    return deep_merge(common, exp)


def get_sampling_config(config: dict[str, Any], profile_name: str, mode: str) -> dict[str, Any]:
    profiles = config["sampling_profiles"]
    profile = profiles[profile_name]
    if profile_name == "official_recommended" and mode in profile:
        sampling = dict(profile[mode])
    else:
        sampling = dict(profile)
    if mode == "think":
        if "max_thinking_tokens" in config:
            sampling.setdefault("max_thinking_tokens", config["max_thinking_tokens"])
        elif "thinking_token_budget" in config:
            sampling.setdefault("max_thinking_tokens", config["thinking_token_budget"])
    return sampling
