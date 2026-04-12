from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

import pandas as pd

from src.data.loaders import DataBundle
from src.schema.contracts import DatasetProfile


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


@dataclass
class FeatureCandidate:
    name: str
    train_feature: pd.Series
    test_feature: pd.Series
    source_family: str
    proxy_score: float = 0.0
    missing_ratio: float = 0.0
    compute_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        expected_train_rows: int | None = None,
        expected_test_rows: int | None = None,
    ) -> None:
        if expected_train_rows is not None and len(self.train_feature) != expected_train_rows:
            raise ValueError(
                f"Feature candidate '{self.name}' train rows mismatch: "
                f"{len(self.train_feature)} != {expected_train_rows}"
            )
        if expected_test_rows is not None and len(self.test_feature) != expected_test_rows:
            raise ValueError(
                f"Feature candidate '{self.name}' test rows mismatch: "
                f"{len(self.test_feature)} != {expected_test_rows}"
            )

    def to_feature_set(self) -> FeatureSet:
        return FeatureSet(
            name=self.name,
            train_features=pd.DataFrame({self.name: self.train_feature.reset_index(drop=True)}),
            test_features=pd.DataFrame({self.name: self.test_feature.reset_index(drop=True)}),
            description=f"Single feature candidate from {self.source_family}.",
            metadata=self.metadata.copy(),
        )


class CandidateGenerator(Protocol):
    name: str

    def generate_candidates(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
        max_candidates: int,
    ) -> list[FeatureCandidate]:
        ...


def normalize_feature_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "feature"
    if not cleaned.startswith("gen_"):
        cleaned = f"gen_{cleaned}"
    return cleaned
