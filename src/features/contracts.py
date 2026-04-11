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

    def validate(
        self,
        expected_train_rows: int | None = None,
        expected_test_rows: int | None = None,
    ) -> None:
        if list(self.train_features.columns) != list(self.test_features.columns):
            raise ValueError(
                f"Feature columns mismatch for set '{self.name}'. "
                f"train: {list(self.train_features.columns)}, test: {list(self.test_features.columns)}"
            )
        if expected_train_rows is not None and len(self.train_features) != expected_train_rows:
            raise ValueError(
                f"Feature row count mismatch for set '{self.name}'. "
                f"train rows: {len(self.train_features)}, expected: {expected_train_rows}"
            )
        if expected_test_rows is not None and len(self.test_features) != expected_test_rows:
            raise ValueError(
                f"Feature row count mismatch for set '{self.name}'. "
                f"test rows: {len(self.test_features)}, expected: {expected_test_rows}"
            )


class FeatureGenerator(Protocol):
    name: str

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        ...
