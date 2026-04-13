from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

import pandas as pd

from src.data.schema import SchemaContext, build_schema_context

logger = logging.getLogger(__name__)


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
    schema_context: SchemaContext


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    candidates: list[pd.DataFrame] = []
    separators = (None, ",", ";", "\t", "|")
    encodings = ("utf-8-sig", "utf-8", "cp1251")
    for encoding in encodings:
        for sep in separators:
            try:
                if sep is None:
                    frame = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
                else:
                    frame = pd.read_csv(path, sep=sep, engine="python", encoding=encoding)
            except Exception:
                continue
            candidates.append(_sanitize_column_names(frame))

    if not candidates:
        raise ValueError(f"Unable to read CSV with auto-detection: {path}")

    best = max(candidates, key=_parse_quality_score)
    if len(best.columns) <= 1 and any(token in str(best.columns[0]) for token in (";", "\t", "|")):
        logger.warning("CSV '%s' parsed into a single suspicious column: %s", path, list(best.columns))
    return best


def load_data_bundle(data_dir: Path) -> DataBundle:
    train = read_csv_auto(data_dir / "train.csv")
    test = read_csv_auto(data_dir / "test.csv")
    train, test = _align_train_test_columns(train=train, test=test)

    aux_tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.name in {"train.csv", "test.csv"}:
            continue
        aux_tables[csv_path.stem] = read_csv_auto(csv_path)

    readme_path = data_dir / "readme.txt"
    data_readme = readme_path.read_text(encoding="utf-8", errors="ignore") if readme_path.exists() else ""

    id_column, target_column = infer_key_columns(train=train, test=test)
    schema_context = build_schema_context(
        train=train,
        test=test,
        aux_tables=aux_tables,
        readme_text=data_readme,
        id_column=id_column,
        target_column=target_column,
    )

    return DataBundle(
        train=train,
        test=test,
        aux_tables=aux_tables,
        data_readme=data_readme,
        id_column=id_column,
        target_column=target_column,
        schema_context=schema_context,
    )


def infer_key_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, str]:
    common_columns = [column for column in train.columns if column in test.columns]
    if not common_columns:
        train_norm = {_normalize_column_key(column): str(column) for column in train.columns}
        test_norm = {_normalize_column_key(column): str(column) for column in test.columns}
        shared_norm = sorted(set(train_norm) & set(test_norm))
        if shared_norm:
            debug_pairs = [(train_norm[token], test_norm[token]) for token in shared_norm[:5]]
            raise ValueError(
                "Unable to infer id column: shared columns exist only after normalization. "
                f"Examples: {debug_pairs}"
            )
        raise ValueError("Unable to infer id column: no shared columns between train and test.")

    # Standard case: target exists only in train.
    target_candidates = [column for column in train.columns if column not in test.columns]
    if target_candidates:
        target_column = pick_target_column(target_candidates)
    else:
        # Some local debug datasets may include target in test too.
        # In this case we still infer a target by preferred names / binary heuristic.
        target_column = pick_target_column_from_shared(train=train, common_columns=common_columns)

    id_candidates = [column for column in common_columns if column != target_column]
    if not id_candidates:
        raise ValueError("Unable to infer id column: no columns left after excluding target.")

    id_column = pick_id_column(train=train, test=test, common_columns=id_candidates)
    return id_column, target_column


def pick_target_column(target_candidates: list[str]) -> str:
    lowered_map = {column.lower(): column for column in target_candidates}
    for preferred in PREFERRED_TARGET_NAMES:
        if preferred in lowered_map:
            return lowered_map[preferred]
    if len(target_candidates) == 1:
        return target_candidates[0]
    return target_candidates[0]


def pick_target_column_from_shared(train: pd.DataFrame, common_columns: list[str]) -> str:
    lowered_map = {column.lower(): column for column in common_columns}
    for preferred in PREFERRED_TARGET_NAMES:
        if preferred in lowered_map:
            return lowered_map[preferred]

    # Fallback: choose a low-cardinality binary-like column that is unlikely an id.
    binary_candidates: list[str] = []
    for column in common_columns:
        series = train[column]
        nunique = series.nunique(dropna=True)
        if nunique <= 2 and series.dtype != "object":
            if "id" not in column.lower():
                binary_candidates.append(column)
    if binary_candidates:
        return binary_candidates[0]

    raise ValueError(
        "Unable to infer target column: train/test columns are identical and no known target field found."
    )


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


def _sanitize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    used: set[str] = set()
    for original in frame.columns:
        name = str(original).replace("\ufeff", "").strip()
        name = re.sub(r"\s+", " ", name)
        if not name:
            name = "column"
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used.add(candidate)
        renamed[original] = candidate
    return frame.rename(columns=renamed)


def _normalize_column_key(column: str) -> str:
    normalized = str(column).replace("\ufeff", "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _align_train_test_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_map: dict[str, str] = {}
    for column in train.columns:
        key = _normalize_column_key(str(column))
        if key and key not in train_map:
            train_map[key] = str(column)

    rename_test: dict[str, str] = {}
    taken = set(test.columns)
    for column in test.columns:
        current = str(column)
        key = _normalize_column_key(current)
        matched = train_map.get(key)
        if not matched or matched == current:
            continue
        if matched in taken and matched not in rename_test.values():
            continue
        rename_test[current] = matched
        taken.add(matched)

    if rename_test:
        test = test.rename(columns=rename_test)
    return train, test


def _parse_quality_score(frame: pd.DataFrame) -> tuple[int, int, int]:
    n_columns = int(frame.shape[1])
    unnamed = sum(str(column).lower().startswith("unnamed") for column in frame.columns)
    n_rows = int(frame.shape[0])
    return (n_columns, -unnamed, n_rows)
