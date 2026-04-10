from __future__ import annotations

import pandas as pd

from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet


class MissingnessGenerator:
    name = "missingness_profile"

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        exclude_cols = {bundle.id_column, bundle.target_column}
        usable_cols = [column for column in bundle.train.columns if column not in exclude_cols]
        if not usable_cols:
            return []

        train_base = bundle.train[usable_cols]
        test_base = bundle.test[usable_cols]

        train_features = pd.DataFrame(index=bundle.train.index)
        test_features = pd.DataFrame(index=bundle.test.index)

        train_features["gen_missing_count"] = train_base.isna().sum(axis=1)
        test_features["gen_missing_count"] = test_base.isna().sum(axis=1)

        train_features["gen_missing_ratio"] = train_base.isna().mean(axis=1)
        test_features["gen_missing_ratio"] = test_base.isna().mean(axis=1)

        numeric_cols = [column for column in usable_cols if pd.api.types.is_numeric_dtype(train_base[column])]
        if numeric_cols:
            train_features["gen_zero_ratio"] = (train_base[numeric_cols] == 0).mean(axis=1)
            test_features["gen_zero_ratio"] = (test_base[numeric_cols] == 0).mean(axis=1)
        else:
            train_features["gen_zero_ratio"] = 0.0
            test_features["gen_zero_ratio"] = 0.0

        object_cols = [
            column
            for column in usable_cols
            if pd.api.types.is_object_dtype(train_base[column])
            or pd.api.types.is_categorical_dtype(train_base[column])
        ]
        if object_cols:
            train_lengths = train_base[object_cols].fillna("").astype(str).apply(lambda col: col.str.len())
            test_lengths = test_base[object_cols].fillna("").astype(str).apply(lambda col: col.str.len())
            train_features["gen_text_len_mean"] = train_lengths.mean(axis=1)
            test_features["gen_text_len_mean"] = test_lengths.mean(axis=1)
        else:
            train_features["gen_text_len_mean"] = 0.0
            test_features["gen_text_len_mean"] = 0.0

        train_features["gen_row_nunique"] = train_base.nunique(axis=1, dropna=True)
        test_features["gen_row_nunique"] = test_base.nunique(axis=1, dropna=True)

        return [
            FeatureSet(
                name=self.name,
                train_features=train_features.iloc[:, :max_features].copy(),
                test_features=test_features.iloc[:, :max_features].copy(),
                description="Missingness and row-level profile features.",
            )
        ]

