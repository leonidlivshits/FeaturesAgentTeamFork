from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet


class NumericHeuristicGenerator:
    name = "numeric_heuristics"

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        exclude_cols = {bundle.id_column, bundle.target_column}
        numeric_cols = [
            column
            for column in bundle.train.columns
            if column not in exclude_cols and pd.api.types.is_numeric_dtype(bundle.train[column])
        ]
        if not numeric_cols:
            return []

        train_numeric = bundle.train[numeric_cols].apply(pd.to_numeric, errors="coerce")
        test_numeric = bundle.test[numeric_cols].apply(pd.to_numeric, errors="coerce")

        candidate_a_train = pd.DataFrame(
            {
                "gen_num_non_null_ratio": train_numeric.notna().mean(axis=1),
                "gen_num_mean": train_numeric.mean(axis=1),
                "gen_num_std": train_numeric.std(axis=1).fillna(0.0),
                "gen_num_min": train_numeric.min(axis=1),
                "gen_num_max": train_numeric.max(axis=1),
            }
        )
        candidate_a_test = pd.DataFrame(
            {
                "gen_num_non_null_ratio": test_numeric.notna().mean(axis=1),
                "gen_num_mean": test_numeric.mean(axis=1),
                "gen_num_std": test_numeric.std(axis=1).fillna(0.0),
                "gen_num_min": test_numeric.min(axis=1),
                "gen_num_max": test_numeric.max(axis=1),
            }
        )

        candidate_b_train = pd.DataFrame(
            {
                "gen_num_sum": train_numeric.sum(axis=1),
                "gen_num_median": train_numeric.median(axis=1),
                "gen_num_range": train_numeric.max(axis=1) - train_numeric.min(axis=1),
                "gen_num_missing_count": train_numeric.isna().sum(axis=1),
                "gen_num_mean_abs": train_numeric.abs().mean(axis=1),
            }
        )
        candidate_b_test = pd.DataFrame(
            {
                "gen_num_sum": test_numeric.sum(axis=1),
                "gen_num_median": test_numeric.median(axis=1),
                "gen_num_range": test_numeric.max(axis=1) - test_numeric.min(axis=1),
                "gen_num_missing_count": test_numeric.isna().sum(axis=1),
                "gen_num_mean_abs": test_numeric.abs().mean(axis=1),
            }
        )

        feature_sets: list[FeatureSet] = [
            FeatureSet(
                name=f"{self.name}_stats",
                train_features=candidate_a_train.iloc[:, :max_features].copy(),
                test_features=candidate_a_test.iloc[:, :max_features].copy(),
                description="Row-wise numeric statistics.",
            ),
            FeatureSet(
                name=f"{self.name}_aggregates",
                train_features=candidate_b_train.iloc[:, :max_features].copy(),
                test_features=candidate_b_test.iloc[:, :max_features].copy(),
                description="Alternative numeric aggregates.",
            ),
        ]

        for feature_set in feature_sets:
            feature_set.train_features = feature_set.train_features.replace([np.inf, -np.inf], np.nan)
            feature_set.test_features = feature_set.test_features.replace([np.inf, -np.inf], np.nan)

        return feature_sets

