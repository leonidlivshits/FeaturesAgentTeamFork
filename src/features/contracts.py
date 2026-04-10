from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from src.data.loaders import DataBundle


@dataclass
class FeatureSet:
    name: str
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if list(self.train_features.columns) != list(self.test_features.columns):
            raise ValueError(
                f"Feature columns mismatch for set '{self.name}'. "
                f"train: {list(self.train_features.columns)}, test: {list(self.test_features.columns)}"
            )
        if len(self.train_features) != len(self.test_features):
            raise ValueError(
                f"Feature row count mismatch for set '{self.name}'. "
                f"train rows: {len(self.train_features)}, test rows: {len(self.test_features)}"
            )


class FeatureGenerator(Protocol):
    name: str

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        ...
