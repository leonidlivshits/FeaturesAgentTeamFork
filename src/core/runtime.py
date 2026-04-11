from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def apply_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class RuntimeBudget:
    total_sec: float
    reserve_sec: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def remaining(self) -> float:
        return max(0.0, float(self.total_sec) - self.elapsed())

    def has_time(self, min_required_sec: float = 0.0) -> bool:
        return self.remaining() > (float(self.reserve_sec) + float(min_required_sec))


def ensure_input_contract(data_dir: Path) -> None:
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    required_files = ("train.csv", "test.csv")
    for filename in required_files:
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required input file: {file_path}")


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()


def ensure_output_contract(output_dir: Path) -> None:
    allowed = {"train.csv", "test.csv"}
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    if existing != allowed:
        raise ValueError(
            f"Output directory must contain exactly {sorted(allowed)}, got {sorted(existing)}"
        )
