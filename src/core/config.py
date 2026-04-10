from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"


@dataclass(frozen=True)
class RuntimeConfig:
    max_features: int = 5
    cv_folds: int = 5
    random_seed: int = 42
    model_name: str = "GigaChat-2-Max"


DEFAULT_CONFIG = RuntimeConfig()

