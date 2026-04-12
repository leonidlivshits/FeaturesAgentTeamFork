from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd


KEY_TOKENS = ("id", "key", "uuid", "hash")
ENTITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "user": ("user", "client", "customer", "uid"),
    "product": ("product", "item", "sku"),
    "order": ("order", "basket", "cart"),
    "row": ("row", "record", "request", "application"),
}


@dataclass(frozen=True)
class TableSchema:
    name: str
    n_rows: int
    columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    key_candidates: tuple[str, ...]


@dataclass(frozen=True)
class JoinEdge:
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    left_coverage: float
    right_coverage: float
    score: float
    evidence: str


@dataclass(frozen=True)
class DatasetProfile:
    dataset_type: str
    detected_entities: dict[str, str]
    confidence: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SchemaContext:
    table_schemas: dict[str, TableSchema]
    join_edges: tuple[JoinEdge, ...]
    recommended_joins: dict[str, str]
    column_descriptions: dict[str, str]
    dataset_profile: DatasetProfile

    def to_prompt_summary(self, max_tables: int = 10, max_edges: int = 10) -> str:
        table_lines: list[str] = []
        for table_name in sorted(self.table_schemas.keys())[:max_tables]:
            table = self.table_schemas[table_name]
            table_lines.append(
                f"- {table.name}: rows={table.n_rows}, cols={len(table.columns)}, keys={list(table.key_candidates)[:3]}"
            )

        edge_lines: list[str] = []
        for edge in list(self.join_edges)[:max_edges]:
            edge_lines.append(
                f"- {edge.left_table}.{edge.left_key} <-> {edge.right_table}.{edge.right_key} score={edge.score:.3f}"
            )

        recommended_lines: list[str] = []
        for table_name in sorted(self.recommended_joins.keys()):
            recommended_lines.append(f"- train/test -> {table_name} by {self.recommended_joins[table_name]}")

        descriptions = [
            f"- {name}: {description[:120]}"
            for name, description in sorted(self.column_descriptions.items())[:12]
        ]

        return (
            f"Dataset type: {self.dataset_profile.dataset_type} (confidence={self.dataset_profile.confidence:.2f})\n"
            f"Entities: {self.dataset_profile.detected_entities}\n"
            "Tables:\n"
            + ("\n".join(table_lines) if table_lines else "- n/a")
            + "\nJoin edges:\n"
            + ("\n".join(edge_lines) if edge_lines else "- n/a")
            + "\nRecommended joins:\n"
            + ("\n".join(recommended_lines) if recommended_lines else "- n/a")
            + "\nDictionary hints:\n"
            + ("\n".join(descriptions) if descriptions else "- n/a")
        )


def build_schema_context(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    aux_tables: dict[str, pd.DataFrame],
    readme_text: str,
    id_column: str,
    target_column: str,
) -> SchemaContext:
    all_tables: dict[str, pd.DataFrame] = {"train": train, "test": test}
    for table_name in sorted(aux_tables.keys()):
        all_tables[table_name] = aux_tables[table_name]

    column_descriptions = extract_column_descriptions(aux_tables)
    table_schemas = {
        table_name: infer_table_schema(
            table_name=table_name,
            dataframe=table_df,
            column_descriptions=column_descriptions,
        )
        for table_name, table_df in all_tables.items()
    }

    join_edges = infer_join_edges(
        table_frames=all_tables,
        target_column=target_column,
        readme_text=readme_text,
    )
    recommended_joins = infer_recommended_joins(
        train=train,
        test=test,
        aux_tables=aux_tables,
        join_edges=join_edges,
        id_column=id_column,
        target_column=target_column,
        readme_text=readme_text,
    )
    profile = infer_dataset_profile(
        train=train,
        test=test,
        aux_tables=aux_tables,
        readme_text=readme_text,
        id_column=id_column,
    )

    return SchemaContext(
        table_schemas=table_schemas,
        join_edges=tuple(join_edges),
        recommended_joins=recommended_joins,
        column_descriptions=column_descriptions,
        dataset_profile=profile,
    )


def infer_table_schema(
    *,
    table_name: str,
    dataframe: pd.DataFrame,
    column_descriptions: dict[str, str],
) -> TableSchema:
    columns = sorted(str(column) for column in dataframe.columns)
    numeric_columns = sorted(
        column for column in columns if pd.api.types.is_numeric_dtype(dataframe[column])
    )
    categorical_columns = sorted(column for column in columns if column not in numeric_columns)
    key_candidates = rank_key_candidates(
        dataframe=dataframe,
        columns=columns,
        column_descriptions=column_descriptions,
    )
    return TableSchema(
        name=table_name,
        n_rows=int(len(dataframe)),
        columns=tuple(columns),
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        key_candidates=tuple(key_candidates),
    )


def rank_key_candidates(
    *,
    dataframe: pd.DataFrame,
    columns: list[str],
    column_descriptions: dict[str, str],
) -> list[str]:
    scored: list[tuple[str, float]] = []
    n_rows = max(int(len(dataframe)), 1)
    for column in columns:
        series = dataframe[column]
        nunique_ratio = float(series.nunique(dropna=True)) / n_rows
        missing_ratio = float(series.isna().mean())
        name_score = key_name_score(column)
        dict_score = dictionary_key_score(column, column_descriptions)
        total = 0.45 * name_score + 0.45 * nunique_ratio + 0.10 * (1.0 - missing_ratio) + 0.05 * dict_score
        scored.append((column, total))

    ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
    result = [column for column, score in ranked if score >= 0.55 or key_name_score(column) >= 0.9]
    return result[:6]


def infer_join_edges(
    *,
    table_frames: dict[str, pd.DataFrame],
    target_column: str,
    readme_text: str,
) -> list[JoinEdge]:
    table_names = sorted(table_frames.keys())
    edges: list[JoinEdge] = []
    lower_readme = (readme_text or "").lower()

    for idx, left_name in enumerate(table_names):
        left_df = table_frames[left_name]
        left_cols = {str(column): left_df[column] for column in left_df.columns}
        for right_name in table_names[idx + 1 :]:
            right_df = table_frames[right_name]
            right_cols = {str(column): right_df[column] for column in right_df.columns}
            common_columns = sorted(set(left_cols.keys()) & set(right_cols.keys()))
            for column in common_columns:
                if column.lower() == target_column.lower():
                    continue
                if not is_possible_join_key(column):
                    continue
                left_cov, right_cov = coverage_pair(left_cols[column], right_cols[column])
                if left_cov <= 0.01 or right_cov <= 0.01:
                    continue
                readme_score = readme_relation_score(lower_readme, left_name, right_name, column)
                score = 0.60 * min(left_cov, right_cov) + 0.25 * key_name_score(column) + 0.15 * readme_score
                if score < 0.05:
                    continue
                evidence = "readme+coverage" if readme_score > 0 else "coverage"
                edges.append(
                    JoinEdge(
                        left_table=left_name,
                        right_table=right_name,
                        left_key=column,
                        right_key=column,
                        left_coverage=left_cov,
                        right_coverage=right_cov,
                        score=float(score),
                        evidence=evidence,
                    )
                )

    return sorted(
        edges,
        key=lambda edge: (-edge.score, edge.left_table, edge.right_table, edge.left_key, edge.right_key),
    )


def infer_recommended_joins(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    aux_tables: dict[str, pd.DataFrame],
    join_edges: Iterable[JoinEdge],
    id_column: str,
    target_column: str,
    readme_text: str,
) -> dict[str, str]:
    base_columns = sorted(set(train.columns) & set(test.columns))
    lower_readme = (readme_text or "").lower()
    edge_lookup: dict[tuple[str, str, str], float] = {}
    for edge in join_edges:
        edge_lookup[(edge.left_table, edge.right_table, edge.left_key)] = edge.score
        edge_lookup[(edge.right_table, edge.left_table, edge.right_key)] = edge.score

    result: dict[str, str] = {}
    for table_name in sorted(aux_tables.keys()):
        aux_df = aux_tables[table_name]
        candidates = [column for column in base_columns if column in aux_df.columns and column.lower() != target_column.lower()]
        if not candidates:
            continue

        scored: list[tuple[str, float]] = []
        for column in candidates:
            train_cov, aux_cov = coverage_pair(train[column], aux_df[column])
            readme_score = readme_relation_score(lower_readme, "train", table_name, column)
            edge_score = edge_lookup.get(("train", table_name, column), 0.0)
            id_bonus = 0.12 if column.lower() == id_column.lower() else 0.0
            score = 0.55 * min(train_cov, aux_cov) + 0.20 * key_name_score(column) + 0.15 * readme_score + 0.10 * edge_score + id_bonus
            scored.append((column, score))

        best_column, best_score = max(scored, key=lambda item: (item[1], item[0]))
        if best_score >= 0.05:
            result[table_name] = best_column

    return result


def infer_dataset_profile(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    aux_tables: dict[str, pd.DataFrame],
    readme_text: str,
    id_column: str,
) -> DatasetProfile:
    common_columns = sorted(set(train.columns) & set(test.columns))
    lower_readme = (readme_text or "").lower()
    table_names = {name.lower() for name in aux_tables.keys()}
    detected_entities = detect_entities(common_columns)
    notes: list[str] = []

    repeat_markers = {
        "tables": {"users", "orders", "order_items", "products"},
        "columns": {"user_id", "product_id", "order_id"},
        "readme": ("repeat", "reorder", "повторн", "товар"),
    }
    deposit_markers = {
        "tables": {"client_data"},
        "columns": {"client_id", "campaign", "pdays", "previous"},
        "readme": ("deposit", "bank", "вклад"),
    }

    repeat_score = 0.0
    deposit_score = 0.0

    if len(repeat_markers["tables"] & table_names) >= 2:
        repeat_score += 0.5
        notes.append("Found repeat-purchase table set.")
    repeat_cols = len(repeat_markers["columns"] & {column.lower() for column in common_columns})
    repeat_score += min(0.4, repeat_cols * 0.12)
    if any(marker in lower_readme for marker in repeat_markers["readme"]):
        repeat_score += 0.2

    if len(deposit_markers["tables"] & table_names) >= 1:
        deposit_score += 0.4
        notes.append("Found deposit-related table names.")
    deposit_cols = len(deposit_markers["columns"] & {column.lower() for column in train.columns})
    deposit_score += min(0.4, deposit_cols * 0.1)
    if "client" in id_column.lower():
        deposit_score += 0.15
    if any(marker in lower_readme for marker in deposit_markers["readme"]):
        deposit_score += 0.2

    if repeat_score >= max(0.5, deposit_score + 0.1):
        dataset_type = "repeat_purchase"
        confidence = min(1.0, repeat_score)
    elif deposit_score >= max(0.5, repeat_score + 0.1):
        dataset_type = "deposit"
        confidence = min(1.0, deposit_score)
    else:
        dataset_type = "generic"
        confidence = max(0.35, min(0.65, max(repeat_score, deposit_score)))
        notes.append("Dataset type is ambiguous, using generic strategy.")

    return DatasetProfile(
        dataset_type=dataset_type,
        detected_entities=detected_entities,
        confidence=float(confidence),
        notes=tuple(sorted(set(notes))),
    )


def detect_entities(columns: Iterable[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for entity, keywords in ENTITY_KEYWORDS.items():
        for column in sorted(columns):
            normalized = column.lower()
            if any(keyword in normalized for keyword in keywords):
                found[entity] = column
                break
    return found


def extract_column_descriptions(aux_tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    candidates = [name for name in aux_tables.keys() if "dictionary" in name.lower()]
    if not candidates:
        return {}

    dictionary_table_name = sorted(candidates)[0]
    dataframe = aux_tables[dictionary_table_name].copy()
    if dataframe.empty:
        return {}

    column_name_field = pick_column(dataframe.columns, ("field", "column", "feature", "name"))
    description_field = pick_column(dataframe.columns, ("description", "desc", "meaning", "comment"))
    if not column_name_field or not description_field:
        return {}

    result: dict[str, str] = {}
    for _, row in dataframe.iterrows():
        key = str(row.get(column_name_field, "")).strip()
        description = str(row.get(description_field, "")).strip()
        if key and description:
            result[key.lower()] = description
    return result


def pick_column(columns: Iterable[str], options: tuple[str, ...]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in columns}
    for option in options:
        if option in lowered:
            return lowered[option]
    for column in sorted(columns):
        normalized = str(column).lower()
        if any(option in normalized for option in options):
            return str(column)
    return None


def key_name_score(column: str) -> float:
    normalized = str(column).lower()
    score = 0.0
    if normalized.endswith("_id"):
        score += 1.0
    if normalized == "id":
        score += 1.0
    if any(token in normalized for token in KEY_TOKENS):
        score += 0.7
    if any(normalized.startswith(prefix) for prefix in ("user_", "product_", "order_", "client_", "row_")):
        score += 0.3
    return float(min(1.0, score))


def dictionary_key_score(column: str, column_descriptions: dict[str, str]) -> float:
    description = column_descriptions.get(column.lower(), "").lower()
    if not description:
        return 0.0
    patterns = ("identifier", "id", "ключ", "идентификатор", "foreign key")
    return 1.0 if any(pattern in description for pattern in patterns) else 0.0


def is_possible_join_key(column: str) -> bool:
    normalized = column.lower()
    if normalized in {"target", "label"}:
        return False
    if key_name_score(column) >= 0.7:
        return True
    return bool(re.search(r"(id|key)$", normalized))


def coverage_pair(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    left_unique = set(left.dropna().astype(str).unique())
    right_unique = set(right.dropna().astype(str).unique())
    if not left_unique or not right_unique:
        return 0.0, 0.0
    intersection = left_unique & right_unique
    left_cov = len(intersection) / len(left_unique)
    right_cov = len(intersection) / len(right_unique)
    return float(left_cov), float(right_cov)


def readme_relation_score(readme_text: str, left_table: str, right_table: str, column: str) -> float:
    if not readme_text:
        return 0.0
    lt = left_table.lower().replace("_", " ")
    rt = right_table.lower().replace("_", " ")
    ck = column.lower().replace("_", " ")
    keywords = ("join", "key", "foreign", "связ", "ключ", "идентификатор")
    has_tables = lt in readme_text and rt in readme_text
    has_column = ck in readme_text
    has_keyword = any(keyword in readme_text for keyword in keywords)
    if has_tables and has_column and has_keyword:
        return 1.0
    if (has_tables and has_column) or (has_column and has_keyword):
        return 0.5
    return 0.0
