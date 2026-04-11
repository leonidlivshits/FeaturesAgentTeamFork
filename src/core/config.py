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
    max_runtime_sec: int = 600
    internal_time_budget_sec: int = 560
    runtime_reserve_sec: int = 30
    min_seconds_per_generator: int = 8
    min_seconds_per_candidate_eval: int = 12
    max_candidate_feature_sets: int = 10
    llm_plan_max_attempts: int = 3
    default_generator_mode: str = "auto"


DEFAULT_CONFIG = RuntimeConfig()
