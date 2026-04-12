from __future__ import annotations

from dataclasses import asdict
import re

import pandas as pd

from src.data.loaders import DataBundle
from src.schema.contracts import DatasetProfile, JoinEdge, JoinPlan, TableProfile


BASE_TABLE_NAME = "base"
GROUP_HINTS = ("user", "customer", "client", "account", "member", "borrower")
TIME_HINTS = ("date", "time", "timestamp", "ts", "dt", "month", "day", "hour")
LEAKAGE_HINTS = (
    "target",
    "label",
    "default",
    "outcome",
    "approved",
    "prediction",
    "score",
    "fraud",
    "reorder",
    "response",
)
ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "user": ("user", "customer", "client", "account", "member", "borrower"),
    "product": ("product", "item", "sku", "catalog"),
    "order": ("order", "purchase", "basket", "cart", "transaction"),
    "application": ("application", "deposit", "loan", "credit", "request"),
    "event": ("event", "history", "log", "session", "activity"),
}


def build_dataset_profile(bundle: DataBundle) -> DatasetProfile:
    base_columns = [column for column in bundle.train.columns if column != bundle.target_column]
    base_train = bundle.train[base_columns].copy()
    base_test = bundle.test[[column for column in base_columns if column in bundle.test.columns]].copy()
    base_profile = _profile_table(
        name=BASE_TABLE_NAME,
        table=base_train,
        id_column=bundle.id_column,
        target_column=bundle.target_column,
        readme_text=bundle.data_readme,
    )

    table_profiles: dict[str, TableProfile] = {BASE_TABLE_NAME: base_profile}
    for table_name, table in sorted(bundle.aux_tables.items()):
        table_profiles[table_name] = _profile_table(
            name=table_name,
            table=table,
            id_column=bundle.id_column,
            target_column=bundle.target_column,
            readme_text=bundle.data_readme,
        )

    dataset_type = _infer_dataset_type(bundle=bundle, table_profiles=table_profiles)
    table_profiles[BASE_TABLE_NAME].role = "base"
    for table_name, profile in table_profiles.items():
        if table_name == BASE_TABLE_NAME:
            continue
        profile.role = _infer_table_role(table_name, profile.columns, dataset_type)

    join_edges = _build_join_edges(
        frames={BASE_TABLE_NAME: pd.concat([base_train, base_test], axis=0, ignore_index=True), **bundle.aux_tables},
        table_profiles=table_profiles,
        readme_text=bundle.data_readme,
    )
    join_plans = _build_join_plans(join_edges=join_edges, dataset_type=dataset_type)
    group_column = _pick_group_column(bundle=bundle, base_profile=base_profile, dataset_type=dataset_type)
    time_column = _pick_time_column(base_profile=base_profile)
    leakage_risk_columns = {
        column
        for profile in table_profiles.values()
        for column in profile.leakage_risk_columns
    }

    metadata = {
        "table_roles": {name: profile.role for name, profile in table_profiles.items()},
        "join_edges": [asdict(edge) for edge in join_edges[:12]],
        "join_plan_count": len(join_plans),
    }
    return DatasetProfile(
        dataset_type=dataset_type,
        id_column=bundle.id_column,
        target_column=bundle.target_column,
        group_column=group_column,
        time_column=time_column,
        base_table=BASE_TABLE_NAME,
        table_profiles=table_profiles,
        join_edges=join_edges,
        join_plans=join_plans,
        leakage_risk_columns=leakage_risk_columns,
        metadata=metadata,
    )


def _profile_table(
    name: str,
    table: pd.DataFrame,
    id_column: str,
    target_column: str,
    readme_text: str,
) -> TableProfile:
    columns = tuple(table.columns)
    numeric_columns = tuple(
        column for column in table.columns if pd.api.types.is_numeric_dtype(table[column])
    )
    categorical_columns = tuple(column for column in table.columns if column not in numeric_columns)
    time_columns = tuple(column for column in table.columns if _looks_like_time_column(column, table[column]))
    key_candidates = tuple(
        column
        for column in table.columns
        if _is_key_candidate(column, table[column], id_column=id_column)
    )
    repeated_key_candidates = tuple(
        column
        for column in key_candidates
        if table[column].nunique(dropna=True) < max(len(table), 1)
    )
    leakage_risk_columns = tuple(
        column
        for column in table.columns
        if _is_leakage_risk(column, target_column=target_column, readme_text=readme_text)
    )

    return TableProfile(
        name=name,
        role=_infer_table_role(name, columns, "generic"),
        row_count=len(table),
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        time_columns=time_columns,
        key_candidates=key_candidates,
        repeated_key_candidates=repeated_key_candidates,
        leakage_risk_columns=leakage_risk_columns,
    )


def _infer_dataset_type(bundle: DataBundle, table_profiles: dict[str, TableProfile]) -> str:
    aux_columns_text = " ".join(
        " ".join(profile.columns)
        for name, profile in sorted(table_profiles.items())
        if name != BASE_TABLE_NAME
    )
    joined_text = " ".join(
        [
            bundle.data_readme.lower(),
            " ".join(bundle.train.columns.astype(str).tolist()).lower(),
            " ".join(bundle.test.columns.astype(str).tolist()).lower(),
            " ".join(table_profiles.keys()).lower(),
            aux_columns_text.lower(),
        ]
    )
    repeat_score = _keyword_score(
        joined_text,
        ("user", "product", "order", "purchase", "cart", "basket", "reorder", "item"),
    )
    deposit_score = _keyword_score(
        joined_text,
        (
            "client",
            "deposit",
            "loan",
            "credit",
            "income",
            "balance",
            "application",
            "borrower",
            "bank",
            "campaign",
            "euribor",
            "housing",
            "education",
            "marital",
            "job",
        ),
    )
    if "client_data" in table_profiles and any(token in joined_text for token in ("deposit", "campaign", "euribor", "bank")):
        deposit_score += 1.0
    if repeat_score >= max(1.25, deposit_score + 0.35):
        return "repeat_purchase_like"
    if deposit_score >= max(1.25, repeat_score + 0.25):
        return "deposit_like"
    return "generic"


def _infer_table_role(name: str, columns: tuple[str, ...], dataset_type: str) -> str:
    text = f"{name} {' '.join(columns)}".lower()
    role_scores = {
        role: _keyword_score(text, keywords)
        for role, keywords in ROLE_HINTS.items()
    }
    best_role, best_score = max(role_scores.items(), key=lambda item: item[1], default=("generic", 0.0))
    if best_score <= 0.0:
        if dataset_type == "deposit_like" and any(token in text for token in ("client", "application")):
            return "application"
        return "generic"
    return best_role


def _build_join_edges(
    frames: dict[str, pd.DataFrame],
    table_profiles: dict[str, TableProfile],
    readme_text: str,
) -> list[JoinEdge]:
    readme_lower = readme_text.lower()
    edges: list[JoinEdge] = []
    table_names = sorted(frames)
    for idx, left_name in enumerate(table_names):
        left_frame = frames[left_name]
        left_profile = table_profiles[left_name]
        for right_name in table_names[idx + 1 :]:
            right_frame = frames[right_name]
            right_profile = table_profiles[right_name]
            common_columns = sorted(set(left_frame.columns) & set(right_frame.columns))
            for column in common_columns:
                if column in left_profile.leakage_risk_columns or column in right_profile.leakage_risk_columns:
                    continue
                if column == "":
                    continue
                coverage_left, coverage_right = _coverage_pair(left_frame[column], right_frame[column])
                name_score = _join_name_score(column=column, left_name=left_name, right_name=right_name)
                readme_score = _readme_join_score(readme_lower, left_name=left_name, right_name=right_name, column=column)
                total_score = 0.55 * min(coverage_left, coverage_right) + 0.20 * max(coverage_left, coverage_right)
                total_score += 0.15 * name_score + 0.10 * readme_score
                if total_score < 0.14:
                    continue
                edges.append(
                    JoinEdge(
                        left_table=left_name,
                        right_table=right_name,
                        left_key=column,
                        right_key=column,
                        coverage_left=coverage_left,
                        coverage_right=coverage_right,
                        name_score=name_score,
                        readme_score=readme_score,
                        total_score=round(float(total_score), 6),
                    )
                )
    return sorted(edges, key=lambda edge: (-edge.total_score, edge.left_table, edge.right_table, edge.left_key))


def _build_join_plans(join_edges: list[JoinEdge], dataset_type: str) -> list[JoinPlan]:
    direct_plans: list[JoinPlan] = []
    for edge in join_edges:
        if edge.left_table == BASE_TABLE_NAME:
            direct_plans.append(
                JoinPlan(
                    path=(edge,),
                    output_table=edge.right_table,
                    output_key=edge.right_key,
                    hop_count=1,
                    total_score=edge.total_score,
                )
            )
        elif edge.right_table == BASE_TABLE_NAME:
            direct_plans.append(
                JoinPlan(
                    path=(edge,),
                    output_table=edge.left_table,
                    output_key=edge.left_key,
                    hop_count=1,
                    total_score=edge.total_score,
                )
            )

    if dataset_type != "repeat_purchase_like":
        return sorted(direct_plans, key=lambda plan: (-plan.total_score, plan.output_table, plan.output_key))

    chained_plans: list[JoinPlan] = list(direct_plans)
    for first in direct_plans:
        intermediate = first.output_table
        for edge in join_edges:
            if BASE_TABLE_NAME in {edge.left_table, edge.right_table}:
                continue
            if intermediate not in {edge.left_table, edge.right_table}:
                continue
            output_table = edge.right_table if edge.left_table == intermediate else edge.left_table
            output_key = edge.right_key if edge.left_table == intermediate else edge.left_key
            if output_table == BASE_TABLE_NAME:
                continue
            score = round(0.6 * first.total_score + 0.4 * edge.total_score, 6)
            if score < 0.18:
                continue
            chained_plans.append(
                JoinPlan(
                    path=(*first.path, edge),
                    output_table=output_table,
                    output_key=output_key,
                    hop_count=2,
                    total_score=score,
                )
            )

    dedup: dict[tuple[str, str, int], JoinPlan] = {}
    for plan in chained_plans:
        key = (plan.output_table, plan.output_key, plan.hop_count)
        if key not in dedup or plan.total_score > dedup[key].total_score:
            dedup[key] = plan
    return sorted(dedup.values(), key=lambda plan: (-plan.total_score, plan.hop_count, plan.output_table))[:8]


def _pick_group_column(bundle: DataBundle, base_profile: TableProfile, dataset_type: str) -> str | None:
    preferred = list(base_profile.repeated_key_candidates) + list(base_profile.key_candidates)
    for column in preferred:
        if column == bundle.id_column:
            continue
        lowered = column.lower()
        if dataset_type == "repeat_purchase_like" and any(token in lowered for token in GROUP_HINTS + ("product",)):
            return column
        if dataset_type == "deposit_like" and any(token in lowered for token in GROUP_HINTS):
            return column

    for column in preferred:
        if column != bundle.id_column and any(token in column.lower() for token in GROUP_HINTS):
            return column
    return None


def _pick_time_column(base_profile: TableProfile) -> str | None:
    return base_profile.time_columns[0] if base_profile.time_columns else None


def _coverage_pair(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    left_unique = set(left.dropna().astype(str).unique())
    right_unique = set(right.dropna().astype(str).unique())
    if not left_unique or not right_unique:
        return 0.0, 0.0
    overlap = left_unique & right_unique
    return len(overlap) / len(left_unique), len(overlap) / len(right_unique)


def _join_name_score(column: str, left_name: str, right_name: str) -> float:
    lowered = column.lower()
    score = 0.0
    if lowered.endswith("_id"):
        score += 0.6
    if "id" in lowered:
        score += 0.2
    if any(token in lowered for token in ("user", "customer", "client", "account", "order", "product")):
        score += 0.2
    if lowered.startswith(left_name.lower()) or lowered.startswith(right_name.lower()):
        score += 0.1
    return min(score, 1.0)


def _readme_join_score(readme_text: str, left_name: str, right_name: str, column: str) -> float:
    score = 0.0
    for token in (left_name.lower(), right_name.lower(), column.lower()):
        if token in readme_text:
            score += 0.15
    if all(token in readme_text for token in (left_name.lower(), right_name.lower(), column.lower())):
        score += 0.25
    return min(score, 1.0)


def _looks_like_time_column(column: str, series: pd.Series) -> bool:
    lowered = column.lower()
    if any(token in lowered for token in TIME_HINTS):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except Exception:
        return False
    return parsed.notna().mean() >= 0.7


def _is_key_candidate(column: str, series: pd.Series, id_column: str) -> bool:
    lowered = column.lower()
    if lowered == id_column.lower():
        return True
    if lowered.endswith("_id") or "id" in lowered or "key" in lowered:
        return True
    if series.nunique(dropna=True) >= max(2, int(len(series) * 0.3)):
        return lowered in {"user", "customer", "client", "account", "product", "order"}
    return False


def _is_leakage_risk(column: str, target_column: str, readme_text: str) -> bool:
    lowered = column.lower()
    if lowered == target_column.lower():
        return True
    if any(token in lowered for token in LEAKAGE_HINTS):
        return True
    if lowered in {"y", "y_true"}:
        return True
    if lowered in readme_text.lower() and "target" in lowered:
        return True
    return False


def _keyword_score(text: str, keywords: tuple[str, ...]) -> float:
    score = 0.0
    for keyword in keywords:
        occurrences = len(re.findall(rf"\b{re.escape(keyword)}\b", text))
        score += min(occurrences, 2) * 0.5
        if keyword in text:
            score += 0.25
    return score
