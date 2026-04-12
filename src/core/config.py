from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("FEATURES_AGENT_DATA_DIR", str(ROOT_DIR / "data"))).resolve()
OUTPUT_DIR = Path(os.getenv("FEATURES_AGENT_OUTPUT_DIR", str(ROOT_DIR / "output"))).resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    max_features: int = 5
    cv_folds: int = 5
    random_seed: int = 42
    model_name: str = "GigaChat-2-Max"
    max_runtime_sec: int = 600
    internal_time_budget_sec: int = 560
    runtime_reserve_sec: int = 60
    profiling_budget_sec: int = 20
    candidate_generation_budget_sec: int = 200
    selection_budget_sec: int = 300
    min_seconds_per_generator: int = 8
    min_seconds_per_candidate_eval: int = 12
    max_candidate_feature_sets: int = 10
    max_candidate_pool_size: int = 12
    llm_plan_max_attempts: int = 1
    default_generator_mode: str = "auto"
    default_llm_provider: str = "openrouter"
    openrouter_model: str = "qwen/qwen-2.5-72b-instruct"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


DEFAULT_CONFIG = RuntimeConfig()
