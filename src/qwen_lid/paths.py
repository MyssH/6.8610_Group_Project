from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "caches"
MODEL_CACHE_DIR = CACHE_DIR / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def model_cache_path(model_id: str) -> Path:
    safe_name = model_id.replace("/", "--")
    return MODEL_CACHE_DIR / safe_name


def resolve_model_source(model_id: str) -> str:
    local_path = model_cache_path(model_id)
    if local_path.exists():
        return str(local_path)
    return model_id


def gsm8k_disk_path() -> Path:
    return RAW_DATA_DIR / "gsm8k"


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CACHE_DIR,
        MODEL_CACHE_DIR,
        OUTPUT_DIR,
        OUTPUT_DIR / "exp_a",
        OUTPUT_DIR / "exp_b",
        OUTPUT_DIR / "exp_c",
        OUTPUT_DIR / "plots",
    ]:
        path.mkdir(parents=True, exist_ok=True)
