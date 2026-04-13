from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import re

import pandas as pd

from src.data.schema import SchemaContext, build_schema_context

logger = logging.getLogger(__name__)


EXCLUDED_AUX_FILENAMES = {
    "train.csv",
    "test.csv",
    "train_labels.csv",
    "test_labels.csv",
    "labels.csv",
    "sample_submission.csv",
}


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
    encodings = ("utf-8-sig", "utf-8", "cp1251")
    delimiter_priority = _detect_delimiters(path=path)
    candidates: list[pd.DataFrame] = []

    for encoding in encodings:
        for sep in delimiter_priority:
            frame = _read_csv_with_sep(path=path, sep=sep, encoding=encoding)
            if frame is None:
                continue
            cleaned = _drop_artifact_columns(_sanitize_column_names(frame))
            candidates.append(cleaned)
            # Fast path: plausible parse, avoid extra heavy full-file reads.
            if cleaned.shape[1] > 1:
                return cleaned

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

    try:
        id_column, target_column = infer_key_columns(train=train, test=test)
    except Exception as error:
        logger.warning("Key inference failed (%s). Applying fallback key inference.", error)
        train, test, id_column, target_column = infer_key_columns_fallback(train=train, test=test)

    aux_tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        if _should_skip_aux_file(csv_path.name):
            logger.info("Skipping auxiliary file '%s' due reserved naming rule.", csv_path.name)
            continue
        frame = read_csv_auto(csv_path)
        if _looks_like_target_table(
            frame=frame,
            table_name=csv_path.stem,
            id_column=id_column,
            target_column=target_column,
        ):
            logger.warning(
                "Skipping auxiliary table '%s' because it looks like a label/target table.",
                csv_path.name,
            )
            continue
        aux_tables[csv_path.stem] = frame

    readme_path = data_dir / "readme.txt"
    data_readme = readme_path.read_text(encoding="utf-8", errors="ignore") if readme_path.exists() else ""
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


def infer_key_columns_fallback(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    train_fixed = train.copy()
    test_fixed = test.copy()

    # Target: prefer explicit names; otherwise choose likely label-like column from train-only fields.
    target_candidates = [column for column in train_fixed.columns if column not in test_fixed.columns]
    target_column = ""
    if target_candidates:
        target_column = pick_target_column(target_candidates)
    else:
        lowered_map = {str(column).lower(): str(column) for column in train_fixed.columns}
        for preferred in PREFERRED_TARGET_NAMES:
            if preferred in lowered_map:
                target_column = lowered_map[preferred]
                break
        if not target_column:
            binary_candidates = [
                str(column)
                for column in train_fixed.columns
                if train_fixed[column].nunique(dropna=True) <= 2 and "id" not in str(column).lower()
            ]
            if binary_candidates:
                target_column = binary_candidates[0]
            else:
                target_column = str(train_fixed.columns[-1])

    # ID: try common columns first.
    common_columns = [column for column in train_fixed.columns if column in test_fixed.columns and column != target_column]
    if common_columns:
        id_column = pick_id_column(train=train_fixed, test=test_fixed, common_columns=common_columns)
        return train_fixed, test_fixed, id_column, target_column

    # Positional rescue: align first column of test to first train non-target column.
    train_non_target = [str(column) for column in train_fixed.columns if str(column) != target_column]
    if not train_non_target or test_fixed.shape[1] == 0:
        raise ValueError("Fallback key inference failed: cannot identify id column.")

    id_column = train_non_target[0]
    test_first = str(test_fixed.columns[0])
    if test_first != id_column:
        # Rename only if target id name is not already present to avoid duplicate columns.
        if id_column not in test_fixed.columns:
            test_fixed = test_fixed.rename(columns={test_first: id_column})
        else:
            # Last-resort: force a shared synthetic id on row order.
            synthetic_id = "__row_id__"
            train_fixed[synthetic_id] = range(len(train_fixed))
            test_fixed[synthetic_id] = range(len(test_fixed))
            id_column = synthetic_id

    return train_fixed, test_fixed, id_column, target_column


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


def _drop_artifact_columns(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = list(frame.columns)
    drop_columns = [
        column
        for column in candidates
        if str(column).strip().lower().startswith("unnamed")
        or not str(column).strip()
    ]
    if not drop_columns:
        return frame
    kept = [column for column in candidates if column not in drop_columns]
    if not kept:
        return frame
    return frame[kept].copy()


def _detect_delimiters(path: Path) -> tuple[str, ...]:
    default = (",", ";", "\t", "|")
    try:
        sample_bytes = path.read_bytes()[:65536]
    except Exception:
        return default
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            sample = sample_bytes.decode(encoding, errors="ignore")
        except Exception:
            continue
        if not sample.strip():
            continue
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
            detected = str(dialect.delimiter)
            ordered = [detected] + [item for item in default if item != detected]
            return tuple(ordered)
        except Exception:
            continue
    return default


def _read_csv_with_sep(path: Path, sep: str, encoding: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, sep=sep, engine="c", encoding=encoding)
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep=sep, engine="python", encoding=encoding)
    except Exception:
        return None


def _should_skip_aux_file(file_name: str) -> bool:
    lowered = str(file_name).strip().lower()
    if lowered in EXCLUDED_AUX_FILENAMES:
        return True
    if lowered.endswith("_labels.csv"):
        return True
    return False


def _looks_like_target_table(
    *,
    frame: pd.DataFrame,
    table_name: str,
    id_column: str,
    target_column: str,
) -> bool:
    if frame.empty:
        return False
    lowered_name = table_name.lower()
    if any(token in lowered_name for token in ("label", "target", "submission", "predict")):
        return True

    columns = [str(column) for column in frame.columns]
    lowered_columns = {column.lower() for column in columns}
    lowered_id = id_column.lower()
    has_id_column = lowered_id in lowered_columns
    if not has_id_column:
        has_id_column = any(
            token in lowered_columns
            for token in ("id", "row_id", "client_id", "request_id", "application_id")
        )
    if not has_id_column:
        return False

    target_like_tokens = {"target", target_column.lower(), "label", "y", "y_true", "prediction", "score"}
    target_like_columns = [column for column in columns if column.lower() in target_like_tokens]
    if not target_like_columns:
        return False

    non_id_target_columns = [
        column
        for column in columns
        if column.lower() not in {lowered_id, *target_like_tokens}
    ]
    if len(non_id_target_columns) > 1:
        return False

    for column in target_like_columns:
        series = frame[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 4:
            return True
        if pd.api.types.is_numeric_dtype(series):
            finite = pd.to_numeric(series, errors="coerce").dropna()
            if not finite.empty:
                lower = float(finite.quantile(0.001))
                upper = float(finite.quantile(0.999))
                if 0.0 <= lower and upper <= 1.0:
                    return True
    return False
