from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_output_contract(
    *,
    input_train: pd.DataFrame,
    input_test: pd.DataFrame,
    output_train_path: Path,
    output_test_path: Path,
    max_features: int,
) -> None:
    if not output_train_path.exists():
        raise FileNotFoundError(f"Missing output file: {output_train_path}")
    if not output_test_path.exists():
        raise FileNotFoundError(f"Missing output file: {output_test_path}")

    output_train = pd.read_csv(output_train_path)
    output_test = pd.read_csv(output_test_path)

    if len(output_train) != len(input_train):
        raise ValueError(
            f"Train row count mismatch. expected={len(input_train)}, got={len(output_train)}"
        )
    if len(output_test) != len(input_test):
        raise ValueError(
            f"Test row count mismatch. expected={len(input_test)}, got={len(output_test)}"
        )

    for column in input_train.columns:
        if column not in output_train.columns:
            raise ValueError(f"Missing required column in output/train.csv: {column}")
    for column in input_test.columns:
        if column not in output_test.columns:
            raise ValueError(f"Missing required column in output/test.csv: {column}")

    if not output_train.columns.is_unique:
        raise ValueError("Duplicate columns in output/train.csv")
    if not output_test.columns.is_unique:
        raise ValueError("Duplicate columns in output/test.csv")

    reserved_train = set(input_train.columns)
    reserved_test = set(input_test.columns)
    feature_cols_train = [c for c in output_train.columns if c not in reserved_train]
    feature_cols_test = [c for c in output_test.columns if c not in reserved_test]

    if feature_cols_train != feature_cols_test:
        raise ValueError(
            "Feature columns mismatch between output/train.csv and output/test.csv."
        )
    if not (1 <= len(feature_cols_train) <= max_features):
        raise ValueError(
            f"Generated feature count must be in [1, {max_features}], got {len(feature_cols_train)}."
        )

    for column in feature_cols_train:
        if output_train[column].isna().all():
            raise ValueError(f"Feature {column} in output/train.csv is all NaN")
        if output_test[column].isna().all():
            raise ValueError(f"Feature {column} in output/test.csv is all NaN")

