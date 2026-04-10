from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.contracts import FeatureSet


def write_submission(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    selected_feature_set: FeatureSet,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = train_base.copy()
    test_out = test_base.copy()

    for column in selected_feature_set.train_features.columns:
        if column in train_out.columns or column in test_out.columns:
            raise ValueError(f"Generated feature '{column}' conflicts with source columns.")

    train_out = pd.concat(
        [train_out.reset_index(drop=True), selected_feature_set.train_features.reset_index(drop=True)],
        axis=1,
    )
    test_out = pd.concat(
        [test_out.reset_index(drop=True), selected_feature_set.test_features.reset_index(drop=True)],
        axis=1,
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    return train_path, test_path

