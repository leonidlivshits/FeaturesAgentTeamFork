from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import infer_key_columns, infer_key_columns_fallback


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

    try:
        id_column, target_column = infer_key_columns(train=input_train, test=input_test)
    except Exception:
        _, _, id_column, target_column = infer_key_columns_fallback(
            train=input_train,
            test=input_test,
        )

    required_train = [id_column, target_column]
    required_test = [id_column]
    for column in required_train:
        if column not in output_train.columns:
            raise ValueError(f"Missing required column in output/train.csv: {column}")
    for column in required_test:
        if column not in output_test.columns:
            raise ValueError(f"Missing required column in output/test.csv: {column}")

    if target_column in output_test.columns:
        raise ValueError(f"output/test.csv must not contain target column '{target_column}'.")

    if not output_train.columns.is_unique:
        raise ValueError("Duplicate columns in output/train.csv")
    if not output_test.columns.is_unique:
        raise ValueError("Duplicate columns in output/test.csv")

    reserved_train = set(required_train)
    reserved_test = set(required_test)
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
