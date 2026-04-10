from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PREFERRED_ID_NAMES = (
    "id",
    "client_id",
    "application_id",
    "request_id",
    "record_id",
)
PREFERRED_TARGET_NAMES = ("target", "label", "y", "default")


@dataclass
class DataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    aux_tables: dict[str, pd.DataFrame]
    data_readme: str
    id_column: str
    target_column: str


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep=None, engine="python")


def load_data_bundle(data_dir: Path) -> DataBundle:
    train = read_csv_auto(data_dir / "train.csv")
    test = read_csv_auto(data_dir / "test.csv")

    aux_tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.name in {"train.csv", "test.csv"}:
            continue
        aux_tables[csv_path.stem] = read_csv_auto(csv_path)

    readme_path = data_dir / "readme.txt"
    data_readme = readme_path.read_text(encoding="utf-8", errors="ignore") if readme_path.exists() else ""

    id_column, target_column = infer_key_columns(train=train, test=test)

    return DataBundle(
        train=train,
        test=test,
        aux_tables=aux_tables,
        data_readme=data_readme,
        id_column=id_column,
        target_column=target_column,
    )


def infer_key_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, str]:
    common_columns = [column for column in train.columns if column in test.columns]
    if not common_columns:
        raise ValueError("Unable to infer id column: no shared columns between train and test.")

    target_candidates = [column for column in train.columns if column not in test.columns]
    if not target_candidates:
        raise ValueError("Unable to infer target column: train and test columns are identical.")

    target_column = pick_target_column(target_candidates)
    id_column = pick_id_column(train=train, test=test, common_columns=common_columns)
    return id_column, target_column


def pick_target_column(target_candidates: list[str]) -> str:
    lowered_map = {column.lower(): column for column in target_candidates}
    for preferred in PREFERRED_TARGET_NAMES:
        if preferred in lowered_map:
            return lowered_map[preferred]
    if len(target_candidates) == 1:
        return target_candidates[0]
    return target_candidates[0]


def pick_id_column(train: pd.DataFrame, test: pd.DataFrame, common_columns: list[str]) -> str:
    lowered_map = {column.lower(): column for column in common_columns}
    for preferred in PREFERRED_ID_NAMES:
        if preferred in lowered_map:
            return lowered_map[preferred]

    ranked = sorted(
        common_columns,
        key=lambda column: (
            train[column].isna().mean() + test[column].isna().mean(),
            -(train[column].nunique(dropna=True) / max(len(train), 1)),
        ),
    )
    return ranked[0]

