from __future__ import annotations

import pandas as pd

from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet


class CategoricalFrequencyGenerator:
    name = "categorical_frequency"

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        exclude_cols = {bundle.id_column, bundle.target_column}
        categorical_cols = [
            column
            for column in bundle.train.columns
            if column not in exclude_cols
            and (
                pd.api.types.is_object_dtype(bundle.train[column])
                or pd.api.types.is_categorical_dtype(bundle.train[column])
            )
        ]
        categorical_cols = sorted(categorical_cols)
        categorical_cols = categorical_cols[:max_features]
        if not categorical_cols:
            return []

        train_df = pd.DataFrame(index=bundle.train.index)
        test_df = pd.DataFrame(index=bundle.test.index)

        for column in categorical_cols:
            normalized_freq = (
                bundle.train[column].fillna("__nan__").astype(str).value_counts(normalize=True, dropna=False)
            )
            feature_name = f"gen_freq_{column}"
            train_df[feature_name] = (
                bundle.train[column].fillna("__nan__").astype(str).map(normalized_freq).fillna(0.0)
            )
            test_df[feature_name] = (
                bundle.test[column].fillna("__nan__").astype(str).map(normalized_freq).fillna(0.0)
            )

        return [
            FeatureSet(
                name=self.name,
                train_features=train_df,
                test_features=test_df,
                description="Category frequency encoding from train distribution.",
            )
        ]
