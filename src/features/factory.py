from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import DEFAULT_CONFIG
from src.core.json_utils import safe_json_load
from src.core.llm import get_effective_llm_provider, get_llm_client
from src.core.runtime import RuntimeBudget
from src.data.loaders import DataBundle
from src.evaluation.feature_selector import score_single_feature_proxy
from src.features.contracts import FeatureCandidate, normalize_feature_name
from src.features.generators.categorical_frequency import CategoricalFrequencyGenerator
from src.features.generators.heuristic_numeric import NumericHeuristicGenerator
from src.features.generators.missingness import MissingnessGenerator
from src.features.generators.relational_aggregates import RelationalAggregatesGenerator
from src.features.registry import get_effective_generator_mode
from src.schema.contracts import DatasetProfile, JoinEdge, JoinPlan

logger = logging.getLogger(__name__)

HUGE_TABLE_ROWS = 1_000_000
VERY_HUGE_TABLE_ROWS = 2_500_000


LLM_ALLOWED_OPERATIONS = {
    "row_mean",
    "row_std",
    "ratio",
    "difference",
    "log1p_abs",
    "missing_ratio",
    "cat_freq",
    "text_len",
    "is_missing",
}


@dataclass(frozen=True)
class LlmFeatureSpec:
    name: str
    operation: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class LlmOperationRule:
    min_columns: int
    max_columns: int
    column_type: str


LLM_OPERATION_RULES = {
    "row_mean": LlmOperationRule(1, 8, "numeric"),
    "row_std": LlmOperationRule(1, 8, "numeric"),
    "ratio": LlmOperationRule(2, 2, "numeric"),
    "difference": LlmOperationRule(2, 2, "numeric"),
    "log1p_abs": LlmOperationRule(1, 1, "numeric"),
    "missing_ratio": LlmOperationRule(1, 8, "any"),
    "cat_freq": LlmOperationRule(1, 1, "categorical"),
    "text_len": LlmOperationRule(1, 1, "categorical"),
    "is_missing": LlmOperationRule(1, 1, "any"),
}


class SchemaAwareFeatureFactory:
    def __init__(self) -> None:
        self.mode = get_effective_generator_mode()

    def generate_candidates(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
        runtime_budget: RuntimeBudget,
    ) -> list[FeatureCandidate]:
        started_at = time.perf_counter()
        target = bundle.train[bundle.target_column]
        train_base, test_base = self._build_base_context(bundle=bundle, profile=profile)
        used_names: set[str] = set()

        candidates: list[FeatureCandidate] = []
        candidates.extend(
            self._generate_generic_candidates(
                train_df=train_base,
                test_df=test_base,
                target=target,
                used_names=used_names,
            )
        )

        if profile.dataset_type == "deposit_like":
            candidates.extend(
                self._generate_deposit_candidates(
                    train_df=train_base,
                    test_df=test_base,
                    profile=profile,
                    target=target,
                    used_names=used_names,
                )
            )
        elif profile.dataset_type == "repeat_purchase_like":
            candidates.extend(
                self._generate_repeat_purchase_candidates(
                    train_df=train_base,
                    test_df=test_base,
                    profile=profile,
                    target=target,
                    used_names=used_names,
                )
            )

        candidates.extend(
            self._generate_join_candidates(
                bundle=bundle,
                profile=profile,
                target=target,
                used_names=used_names,
            )
        )

        if self.mode != "heuristic" and runtime_budget.has_time(25):
            llm_train_context, llm_test_context = self._build_llm_context(
                train_df=train_base,
                test_df=test_base,
                join_candidates=candidates,
            )
            candidates.extend(
                self._generate_llm_candidates(
                    bundle=bundle,
                    profile=profile,
                    train_context=llm_train_context,
                    test_context=llm_test_context,
                    target=target,
                    heuristics=candidates,
                    used_names=used_names,
                )
            )

        candidates.extend(
            self._convert_fallback_generators(
                bundle=bundle,
                target=target,
                used_names=used_names,
            )
        )

        logger.info(
            "Feature factory produced %s raw candidates in %.2fs for dataset_type=%s mode=%s",
            len(candidates),
            time.perf_counter() - started_at,
            profile.dataset_type,
            self.mode,
        )
        return candidates

    def _build_base_context(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        shared_columns = [
            column
            for column in bundle.train.columns
            if column in bundle.test.columns and column not in {bundle.id_column, bundle.target_column}
        ]
        safe_columns = [column for column in shared_columns if column not in profile.leakage_risk_columns]
        return bundle.train[safe_columns].copy(), bundle.test[safe_columns].copy()

    def _generate_generic_candidates(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target: pd.Series,
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        candidates: list[FeatureCandidate] = []
        numeric_cols = [column for column in train_df.columns if pd.api.types.is_numeric_dtype(train_df[column])]
        categorical_cols = [column for column in train_df.columns if column not in numeric_cols]

        top_numeric = self._rank_numeric_columns(train_df=train_df, target=target, columns=numeric_cols, limit=6)
        top_categorical = self._rank_categorical_columns(
            train_df=train_df,
            test_df=test_df,
            target=target,
            columns=categorical_cols,
            limit=4,
        )

        if numeric_cols:
            numeric_frame_train = train_df[top_numeric or numeric_cols[:8]].apply(pd.to_numeric, errors="coerce")
            numeric_frame_test = test_df[top_numeric or numeric_cols[:8]].apply(pd.to_numeric, errors="coerce")
            candidates.extend(
                self._build_candidates(
                    [
                        ("row_num_mean", numeric_frame_train.mean(axis=1), numeric_frame_test.mean(axis=1)),
                        ("row_num_std", numeric_frame_train.std(axis=1).fillna(0.0), numeric_frame_test.std(axis=1).fillna(0.0)),
                        ("row_num_min", numeric_frame_train.min(axis=1), numeric_frame_test.min(axis=1)),
                        ("row_num_max", numeric_frame_train.max(axis=1), numeric_frame_test.max(axis=1)),
                        ("row_num_range", numeric_frame_train.max(axis=1) - numeric_frame_train.min(axis=1), numeric_frame_test.max(axis=1) - numeric_frame_test.min(axis=1)),
                        ("row_num_sum", numeric_frame_train.sum(axis=1), numeric_frame_test.sum(axis=1)),
                        ("row_num_mean_abs", numeric_frame_train.abs().mean(axis=1), numeric_frame_test.abs().mean(axis=1)),
                        ("row_num_non_null_ratio", numeric_frame_train.notna().mean(axis=1), numeric_frame_test.notna().mean(axis=1)),
                    ],
                    source_family="generic_row_profile",
                    target=target,
                    used_names=used_names,
                    compute_cost=0.12,
                )
            )

            for column in top_numeric[:4]:
                train_col = pd.to_numeric(train_df[column], errors="coerce")
                test_col = pd.to_numeric(test_df[column], errors="coerce")
                candidates.extend(
                    self._build_candidates(
                        [
                            (f"{column}_log1p_abs", np.sign(train_col) * np.log1p(np.abs(train_col)), np.sign(test_col) * np.log1p(np.abs(test_col))),
                            (f"{column}_is_missing", train_col.isna().astype("int8"), test_col.isna().astype("int8")),
                        ],
                        source_family="generic_numeric_unary",
                        target=target,
                        used_names=used_names,
                        compute_cost=0.1,
                    )
                )

            for left, right in combinations(top_numeric[:3], 2):
                left_train = pd.to_numeric(train_df[left], errors="coerce")
                right_train = pd.to_numeric(train_df[right], errors="coerce").replace(0, np.nan)
                left_test = pd.to_numeric(test_df[left], errors="coerce")
                right_test = pd.to_numeric(test_df[right], errors="coerce").replace(0, np.nan)
                candidates.extend(
                    self._build_candidates(
                        [
                            (f"{left}_to_{right}_ratio", left_train / right_train, left_test / right_test),
                            (f"{left}_minus_{right}", left_train - pd.to_numeric(train_df[right], errors="coerce"), left_test - pd.to_numeric(test_df[right], errors="coerce")),
                        ],
                        source_family="generic_numeric_pairs",
                        target=target,
                        used_names=used_names,
                        compute_cost=0.14,
                    )
                )

        candidates.extend(
            self._build_candidates(
                [
                    ("missing_count", train_df.isna().sum(axis=1), test_df.isna().sum(axis=1)),
                    ("missing_ratio", train_df.isna().mean(axis=1), test_df.isna().mean(axis=1)),
                    ("row_nunique", train_df.nunique(axis=1, dropna=True), test_df.nunique(axis=1, dropna=True)),
                ],
                source_family="generic_missingness",
                target=target,
                used_names=used_names,
                compute_cost=0.08,
            )
        )

        if numeric_cols:
            zero_cols = top_numeric or numeric_cols[:6]
            zero_train = train_df[zero_cols].apply(pd.to_numeric, errors="coerce")
            zero_test = test_df[zero_cols].apply(pd.to_numeric, errors="coerce")
            candidates.extend(
                self._build_candidates(
                    [("zero_ratio", (zero_train == 0).mean(axis=1), (zero_test == 0).mean(axis=1))],
                    source_family="generic_missingness",
                    target=target,
                    used_names=used_names,
                    compute_cost=0.08,
                )
            )

        for column in top_categorical:
            frequencies = train_df[column].fillna("__nan__").astype(str).value_counts(normalize=True, dropna=False)
            candidates.extend(
                self._build_candidates(
                    [
                        (f"{column}_freq", train_df[column].fillna("__nan__").astype(str).map(frequencies).fillna(0.0), test_df[column].fillna("__nan__").astype(str).map(frequencies).fillna(0.0)),
                        (f"{column}_rarity", 1.0 - train_df[column].fillna("__nan__").astype(str).map(frequencies).fillna(0.0), 1.0 - test_df[column].fillna("__nan__").astype(str).map(frequencies).fillna(0.0)),
                        (f"{column}_text_len", train_df[column].fillna("").astype(str).str.len(), test_df[column].fillna("").astype(str).str.len()),
                    ],
                    source_family="generic_categorical",
                    target=target,
                    used_names=used_names,
                    compute_cost=0.1,
                )
            )

        return candidates

    def _generate_deposit_candidates(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        profile: DatasetProfile,
        target: pd.Series,
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        candidates: list[FeatureCandidate] = []
        group_column = profile.group_column if profile.group_column in train_df.columns else None
        numeric_cols = [column for column in train_df.columns if pd.api.types.is_numeric_dtype(train_df[column])]
        financial_cols = [
            column
            for column in numeric_cols
            if any(token in column.lower() for token in ("amount", "balance", "income", "salary", "credit", "loan", "deposit", "payment"))
        ]
        if group_column:
            candidates.extend(
                self._group_count_candidates(
                    train_df=train_df,
                    test_df=test_df,
                    group_column=group_column,
                    target=target,
                    used_names=used_names,
                    family="deposit_group",
                )
            )
            for column in (financial_cols or numeric_cols)[:2]:
                group_train, group_test = self._group_mean_feature(train_df, test_df, group_column, column)
                candidates.extend(
                    self._build_candidates(
                        [(f"{group_column}_{column}_mean", group_train, group_test)],
                        source_family="deposit_group",
                        target=target,
                        used_names=used_names,
                        compute_cost=0.16,
                    )
                )

        for left, right in combinations(financial_cols[:3], 2):
            left_train = pd.to_numeric(train_df[left], errors="coerce")
            right_train = pd.to_numeric(train_df[right], errors="coerce").replace(0, np.nan)
            left_test = pd.to_numeric(test_df[left], errors="coerce")
            right_test = pd.to_numeric(test_df[right], errors="coerce").replace(0, np.nan)
            candidates.extend(
                self._build_candidates(
                    [
                        (f"{left}_to_{right}_financial_ratio", left_train / right_train, left_test / right_test),
                        (f"{left}_minus_{right}_financial_diff", left_train - pd.to_numeric(train_df[right], errors="coerce"), left_test - pd.to_numeric(test_df[right], errors="coerce")),
                    ],
                    source_family="deposit_financial",
                    target=target,
                    used_names=used_names,
                    compute_cost=0.18,
                )
            )
        return candidates

    def _generate_repeat_purchase_candidates(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        profile: DatasetProfile,
        target: pd.Series,
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        candidates: list[FeatureCandidate] = []
        user_column = self._find_column_by_tokens(train_df.columns, ("user", "customer", "client", "account"))
        product_column = self._find_column_by_tokens(train_df.columns, ("product", "item", "sku"))
        group_column = profile.group_column if profile.group_column in train_df.columns else user_column
        time_column = profile.time_column if profile.time_column in train_df.columns else None

        if user_column:
            candidates.extend(
                self._group_count_candidates(
                    train_df=train_df,
                    test_df=test_df,
                    group_column=user_column,
                    target=target,
                    used_names=used_names,
                    family="repeat_user",
                )
            )
        if product_column:
            candidates.extend(
                self._group_count_candidates(
                    train_df=train_df,
                    test_df=test_df,
                    group_column=product_column,
                    target=target,
                    used_names=used_names,
                    family="repeat_product",
                )
            )
        if user_column and product_column:
            train_pair, test_pair = self._pair_count_feature(train_df, test_df, user_column, product_column)
            candidates.extend(
                self._build_candidates(
                    [(f"{user_column}_{product_column}_pair_count", train_pair, test_pair)],
                    source_family="repeat_user_product",
                    target=target,
                    used_names=used_names,
                    compute_cost=0.18,
                )
            )

        if time_column:
            parsed_train = pd.to_datetime(train_df[time_column], errors="coerce")
            parsed_test = pd.to_datetime(test_df[time_column], errors="coerce")
            candidates.extend(
                self._build_candidates(
                    [
                        (f"{time_column}_month", parsed_train.dt.month.fillna(0), parsed_test.dt.month.fillna(0)),
                        (f"{time_column}_dayofweek", parsed_train.dt.dayofweek.fillna(0), parsed_test.dt.dayofweek.fillna(0)),
                        (f"{time_column}_is_weekend", parsed_train.dt.dayofweek.isin([5, 6]).astype("int8"), parsed_test.dt.dayofweek.isin([5, 6]).astype("int8")),
                    ],
                    source_family="repeat_time",
                    target=target,
                    used_names=used_names,
                    compute_cost=0.12,
                )
            )
            if group_column:
                recency_train, recency_test = self._group_recency_feature(
                    train_df=train_df,
                    test_df=test_df,
                    group_column=group_column,
                    time_column=time_column,
                )
                candidates.extend(
                    self._build_candidates(
                        [(f"{group_column}_recency_days", recency_train, recency_test)],
                        source_family="repeat_time",
                        target=target,
                        used_names=used_names,
                        compute_cost=0.18,
                    )
                )
        return candidates

    def _generate_join_candidates(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
        target: pd.Series,
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        candidates: list[FeatureCandidate] = []
        processed_direct = 0
        for plan in sorted(profile.join_plans, key=lambda item: (-item.total_score, item.hop_count, item.output_table)):
            output_df = bundle.aux_tables.get(plan.output_table)
            output_rows = len(output_df) if output_df is not None else 0
            if plan.hop_count == 2 and output_rows >= HUGE_TABLE_ROWS:
                continue
            if plan.hop_count == 1 and output_rows >= VERY_HUGE_TABLE_ROWS and processed_direct >= 1:
                continue
            if plan.hop_count == 1:
                processed_direct += 1
                candidates.extend(
                    self._generate_one_hop_candidates(
                        bundle=bundle,
                        profile=profile,
                        plan=plan,
                        target=target,
                        used_names=used_names,
                    )
                )
            elif plan.hop_count == 2:
                candidates.extend(
                    self._generate_two_hop_candidates(
                        bundle=bundle,
                        profile=profile,
                        plan=plan,
                        target=target,
                        used_names=used_names,
                    )
                )
        return candidates

    def _generate_one_hop_candidates(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
        plan: JoinPlan,
        target: pd.Series,
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        edge = plan.path[0]
        aux_table = plan.output_table
        aux_df = bundle.aux_tables.get(aux_table)
        if aux_df is None:
            return []
        base_key = edge.left_key if edge.left_table == profile.base_table else edge.right_key
        aux_key = edge.right_key if edge.left_table == profile.base_table else edge.left_key
        aggregated = self._aggregate_table(
            table=aux_df,
            table_name=aux_table,
            profile=profile,
            key_column=aux_key,
            limit_numeric=3,
            limit_categorical=2,
        )
        if aggregated.empty or base_key not in bundle.train.columns or base_key not in bundle.test.columns:
            return []

        train_joined = bundle.train[[base_key]].merge(aggregated, left_on=base_key, right_on=aux_key, how="left")
        test_joined = bundle.test[[base_key]].merge(aggregated, left_on=base_key, right_on=aux_key, how="left")
        train_joined = train_joined.drop(columns=[base_key, aux_key], errors="ignore")
        test_joined = test_joined.drop(columns=[base_key, aux_key], errors="ignore")
        return self._frame_to_candidates(
            train_df=train_joined,
            test_df=test_joined,
            source_family=f"join_{aux_table}",
            target=target,
            used_names=used_names,
            compute_cost=0.32,
            metadata={"join_complexity": 1.0, "plan_score": plan.total_score, "join_table": aux_table},
        )

    def _generate_two_hop_candidates(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
        plan: JoinPlan,
        target: pd.Series,
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        if len(plan.path) != 2:
            return []
        first_edge, second_edge = plan.path
        intermediate_table = first_edge.right_table if first_edge.left_table == profile.base_table else first_edge.left_table
        target_table = plan.output_table
        intermediate_df = bundle.aux_tables.get(intermediate_table)
        target_df = bundle.aux_tables.get(target_table)
        if intermediate_df is None or target_df is None:
            return []
        if len(intermediate_df) >= HUGE_TABLE_ROWS or len(target_df) >= HUGE_TABLE_ROWS:
            return []

        base_key = first_edge.left_key if first_edge.left_table == profile.base_table else first_edge.right_key
        intermediate_base_key = first_edge.right_key if first_edge.left_table == profile.base_table else first_edge.left_key
        bridge_key = second_edge.left_key if second_edge.left_table == intermediate_table else second_edge.right_key
        target_key = second_edge.right_key if second_edge.left_table == intermediate_table else second_edge.left_key
        if base_key not in bundle.train.columns or base_key not in bundle.test.columns:
            return []
        if intermediate_base_key not in intermediate_df.columns or bridge_key not in intermediate_df.columns or target_key not in target_df.columns:
            return []

        target_aggregated = self._aggregate_table(
            table=target_df,
            table_name=target_table,
            profile=profile,
            key_column=target_key,
            limit_numeric=2,
            limit_categorical=1,
        )
        if target_aggregated.empty:
            return []

        intermediate_columns = list(dict.fromkeys([intermediate_base_key, bridge_key]))
        merged_intermediate = intermediate_df[intermediate_columns].merge(
            target_aggregated,
            left_on=bridge_key,
            right_on=target_key,
            how="left",
        )
        drop_columns = [
            column
            for column in dict.fromkeys([bridge_key, target_key])
            if column != intermediate_base_key
        ]
        merged_intermediate = merged_intermediate.drop(columns=drop_columns, errors="ignore")
        if merged_intermediate.empty:
            return []

        prefix = f"gen_path_{intermediate_table}_{target_table}"
        grouped = merged_intermediate.groupby(intermediate_base_key, dropna=False)
        result = grouped.size().rename(f"{prefix}_row_count").to_frame()
        numeric_columns = [
            column
            for column in merged_intermediate.columns
            if column != intermediate_base_key and pd.api.types.is_numeric_dtype(merged_intermediate[column])
        ][:3]
        for column in numeric_columns:
            result[f"{prefix}_{column}_mean"] = grouped[column].mean()
            result[f"{prefix}_{column}_max"] = grouped[column].max()
        result = result.reset_index()

        train_joined = bundle.train[[base_key]].merge(result, left_on=base_key, right_on=intermediate_base_key, how="left")
        test_joined = bundle.test[[base_key]].merge(result, left_on=base_key, right_on=intermediate_base_key, how="left")
        train_joined = train_joined.drop(columns=[base_key, intermediate_base_key], errors="ignore")
        test_joined = test_joined.drop(columns=[base_key, intermediate_base_key], errors="ignore")
        return self._frame_to_candidates(
            train_df=train_joined,
            test_df=test_joined,
            source_family=f"path_{intermediate_table}_{target_table}",
            target=target,
            used_names=used_names,
            compute_cost=0.48,
            metadata={"join_complexity": 2.0, "plan_score": plan.total_score, "join_table": target_table},
        )

    def _aggregate_table(
        self,
        table: pd.DataFrame,
        table_name: str,
        profile: DatasetProfile,
        key_column: str,
        limit_numeric: int,
        limit_categorical: int,
    ) -> pd.DataFrame:
        if key_column not in table.columns:
            return pd.DataFrame()
        table_profile = profile.table_profiles.get(table_name)
        leakage_cols = set(table_profile.leakage_risk_columns) if table_profile is not None else set()
        safe_prefix = table_name.replace(" ", "_").lower()
        numeric_cols = [
            column
            for column in table.columns
            if column != key_column and column not in leakage_cols and pd.api.types.is_numeric_dtype(table[column])
        ][:limit_numeric]
        categorical_cols = [
            column
            for column in table.columns
            if column != key_column and column not in leakage_cols and column not in numeric_cols
        ][:limit_categorical]
        if len(table) >= HUGE_TABLE_ROWS:
            numeric_cols = numeric_cols[:1]
            categorical_cols = categorical_cols[:1]
        if not numeric_cols and not categorical_cols:
            return pd.DataFrame()

        grouped = table.groupby(key_column, dropna=False)
        result = grouped.size().rename(f"gen_{safe_prefix}_row_count").to_frame()
        for column in numeric_cols:
            result[f"gen_{safe_prefix}_{column}_mean"] = grouped[column].mean()
            result[f"gen_{safe_prefix}_{column}_std"] = grouped[column].std().fillna(0.0)
            result[f"gen_{safe_prefix}_{column}_sum"] = grouped[column].sum()
        for column in categorical_cols:
            result[f"gen_{safe_prefix}_{column}_nunique"] = grouped[column].nunique(dropna=True)

        time_columns = [
            column
            for column in categorical_cols
            if self._looks_like_time_series(table[column], column)
        ][:1]
        for column in time_columns:
            parsed = pd.to_datetime(table[column], errors="coerce", format="mixed")
            span = grouped.apply(
                lambda frame: (pd.to_datetime(frame[column], errors="coerce", format="mixed").max() - pd.to_datetime(frame[column], errors="coerce", format="mixed").min()).days
                if pd.to_datetime(frame[column], errors="coerce", format="mixed").notna().any()
                else np.nan
            )
            result[f"gen_{safe_prefix}_{column}_span_days"] = span

        return result.reset_index()

    def _frame_to_candidates(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        source_family: str,
        target: pd.Series,
        used_names: set[str],
        compute_cost: float,
        metadata: dict[str, Any],
    ) -> list[FeatureCandidate]:
        candidates: list[FeatureCandidate] = []
        for column in train_df.columns:
            candidates.extend(
                self._build_candidates(
                    [(column, train_df[column], test_df[column])],
                    source_family=source_family,
                    target=target,
                    used_names=used_names,
                    compute_cost=compute_cost,
                    metadata=metadata,
                )
            )
        return candidates

    def _build_llm_context(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        join_candidates: list[FeatureCandidate],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ranked = sorted(join_candidates, key=lambda item: (-float(item.proxy_score), item.name))
        context_train = train_df.copy()
        context_test = test_df.copy()
        for candidate in ranked[:8]:
            context_train[candidate.name] = candidate.train_feature.reset_index(drop=True)
            context_test[candidate.name] = candidate.test_feature.reset_index(drop=True)
        return context_train, context_test

    def _generate_llm_candidates(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
        train_context: pd.DataFrame,
        test_context: pd.DataFrame,
        target: pd.Series,
        heuristics: list[FeatureCandidate],
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        planner = LlmCandidatePlanner()
        if not planner.is_available:
            return []
        top_heuristics = [
            {"name": candidate.name, "family": candidate.source_family, "proxy": round(float(candidate.proxy_score), 6)}
            for candidate in sorted(heuristics, key=lambda item: (-float(item.proxy_score), item.name))[:8]
        ]
        return planner.generate_candidates(
            bundle=bundle,
            profile=profile,
            train_context=train_context,
            test_context=test_context,
            target=target,
            top_heuristics=top_heuristics,
            used_names=used_names,
        )

    def _convert_fallback_generators(
        self,
        bundle: DataBundle,
        target: pd.Series,
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        fallback_generators = [
            NumericHeuristicGenerator(),
            RelationalAggregatesGenerator(),
            CategoricalFrequencyGenerator(),
            MissingnessGenerator(),
        ]
        candidates: list[FeatureCandidate] = []
        for generator in fallback_generators:
            try:
                feature_sets = generator.generate(bundle=bundle, max_features=DEFAULT_CONFIG.max_features)
            except Exception as error:
                logger.debug("Fallback generator failed: %s -> %s", generator.name, error)
                continue
            for feature_set in feature_sets:
                for column in feature_set.train_features.columns:
                    candidates.extend(
                        self._build_candidates(
                            [(column, feature_set.train_features[column], feature_set.test_features[column])],
                            source_family=f"fallback_{generator.name}",
                            target=target,
                            used_names=used_names,
                            compute_cost=0.5,
                            metadata={"fallback_generator": generator.name, "join_complexity": 1.0 if "relational" in generator.name else 0.0},
                        )
                    )
        return candidates

    def _group_count_candidates(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_column: str,
        target: pd.Series,
        used_names: set[str],
        family: str,
    ) -> list[FeatureCandidate]:
        counts = train_df[group_column].fillna("__nan__").astype(str).value_counts()
        train_series = train_df[group_column].fillna("__nan__").astype(str).map(counts).fillna(0.0)
        test_series = test_df[group_column].fillna("__nan__").astype(str).map(counts).fillna(0.0)
        return self._build_candidates(
            [(f"{group_column}_count", train_series, test_series)],
            source_family=family,
            target=target,
            used_names=used_names,
            compute_cost=0.14,
        )

    def _group_mean_feature(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_column: str,
        value_column: str,
    ) -> tuple[pd.Series, pd.Series]:
        fit_frame = pd.DataFrame(
            {
                "_group_key": train_df[group_column].fillna("__nan__").astype(str),
                "_value": pd.to_numeric(train_df[value_column], errors="coerce"),
            }
        )
        means = fit_frame.groupby("_group_key", dropna=False)["_value"].mean()
        train_series = train_df[group_column].fillna("__nan__").astype(str).map(means)
        test_series = test_df[group_column].fillna("__nan__").astype(str).map(means)
        return train_series, test_series

    def _pair_count_feature(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        left_column: str,
        right_column: str,
    ) -> tuple[pd.Series, pd.Series]:
        train_key = train_df[left_column].fillna("__nan__").astype(str) + "||" + train_df[right_column].fillna("__nan__").astype(str)
        counts = train_key.value_counts()
        train_key = train_df[left_column].fillna("__nan__").astype(str) + "||" + train_df[right_column].fillna("__nan__").astype(str)
        test_key = test_df[left_column].fillna("__nan__").astype(str) + "||" + test_df[right_column].fillna("__nan__").astype(str)
        return train_key.map(counts).fillna(0.0), test_key.map(counts).fillna(0.0)

    def _group_recency_feature(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_column: str,
        time_column: str,
    ) -> tuple[pd.Series, pd.Series]:
        train_time = pd.to_datetime(train_df[time_column], errors="coerce")
        test_time = pd.to_datetime(test_df[time_column], errors="coerce")
        fit_frame = pd.DataFrame(
            {
                "_group_key": train_df[group_column].fillna("__nan__").astype(str),
                "_time": train_time,
            }
        )
        max_time = fit_frame.groupby("_group_key", dropna=False)["_time"].max()
        train_series = (train_df[group_column].fillna("__nan__").astype(str).map(max_time) - train_time).dt.days
        test_series = (test_df[group_column].fillna("__nan__").astype(str).map(max_time) - test_time).dt.days
        return train_series, test_series

    def _rank_numeric_columns(
        self,
        train_df: pd.DataFrame,
        target: pd.Series,
        columns: list[str],
        limit: int,
    ) -> list[str]:
        scored = []
        for column in columns:
            series = pd.to_numeric(train_df[column], errors="coerce")
            score = score_single_feature_proxy(series, target)
            scored.append((column, score))
        return [column for column, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]]

    def _rank_categorical_columns(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target: pd.Series,
        columns: list[str],
        limit: int,
    ) -> list[str]:
        scored = []
        for column in columns:
            frequencies = train_df[column].fillna("__nan__").astype(str).value_counts(normalize=True, dropna=False)
            proxy_series = train_df[column].fillna("__nan__").astype(str).map(frequencies).fillna(0.0)
            score = score_single_feature_proxy(proxy_series, target)
            scored.append((column, score))
        return [column for column, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]]

    def _build_candidates(
        self,
        definitions: list[tuple[str, pd.Series, pd.Series]],
        source_family: str,
        target: pd.Series,
        used_names: set[str],
        compute_cost: float,
        metadata: dict[str, Any] | None = None,
    ) -> list[FeatureCandidate]:
        candidates: list[FeatureCandidate] = []
        for name, train_series, test_series in definitions:
            candidate = self._make_candidate(
                name=name,
                train_series=train_series,
                test_series=test_series,
                source_family=source_family,
                target=target,
                used_names=used_names,
                compute_cost=compute_cost,
                metadata=metadata or {},
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _make_candidate(
        self,
        name: str,
        train_series: pd.Series,
        test_series: pd.Series,
        source_family: str,
        target: pd.Series,
        used_names: set[str],
        compute_cost: float,
        metadata: dict[str, Any],
    ) -> FeatureCandidate | None:
        safe_name = self._reserve_name(name, used_names)
        train = train_series.reset_index(drop=True).copy()
        test = test_series.reset_index(drop=True).copy()
        if pd.api.types.is_numeric_dtype(train) and pd.api.types.is_numeric_dtype(test):
            train = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
            test = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite_train = train[np.isfinite(train)]
            if not finite_train.empty and finite_train.nunique(dropna=True) > 4:
                lower = float(finite_train.quantile(0.001))
                upper = float(finite_train.quantile(0.999))
                if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                    train = train.clip(lower, upper)
                    test = test.clip(lower, upper)
        else:
            train = train.fillna("__nan__").astype(str)
            test = test.fillna("__nan__").astype(str)

        if train.isna().all() or test.isna().all():
            return None
        if train.nunique(dropna=True) < 2:
            return None

        missing_ratio = float(np.mean([train.isna().mean(), test.isna().mean()])) if hasattr(train, "isna") else 0.0
        proxy = score_single_feature_proxy(train, target)
        return FeatureCandidate(
            name=safe_name,
            train_feature=train,
            test_feature=test,
            source_family=source_family,
            proxy_score=float(proxy),
            missing_ratio=missing_ratio,
            compute_cost=float(compute_cost),
            metadata=metadata.copy(),
        )

    def _reserve_name(self, name: str, used_names: set[str]) -> str:
        base = normalize_feature_name(name)
        candidate = base
        suffix = 2
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    @staticmethod
    def _find_column_by_tokens(columns: pd.Index | list[str], tokens: tuple[str, ...]) -> str | None:
        for column in columns:
            lowered = str(column).lower()
            if any(token in lowered for token in tokens):
                return str(column)
        return None

    @staticmethod
    def _looks_like_time_series(series: pd.Series, column_name: str) -> bool:
        lowered = column_name.lower()
        if any(token in lowered for token in ("date", "time", "timestamp", "ts", "dt")):
            return True
        sample = series.dropna().astype(str).head(20)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return bool(parsed.notna().mean() >= 0.7)


class LlmCandidatePlanner:
    def __init__(self) -> None:
        self.provider = get_effective_llm_provider()
        self.client = get_llm_client(timeout=25)
        self.is_available = self.provider != "none" and self.client is not None and os.getenv("FEATURES_AGENT_DISABLE_LLM", "0").strip() not in {"1", "true", "yes"}

    def generate_candidates(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
        train_context: pd.DataFrame,
        test_context: pd.DataFrame,
        target: pd.Series,
        top_heuristics: list[dict[str, Any]],
        used_names: set[str],
    ) -> list[FeatureCandidate]:
        if not self.is_available:
            return []
        common_columns = [column for column in train_context.columns if column in test_context.columns]
        numeric_columns = [column for column in common_columns if pd.api.types.is_numeric_dtype(train_context[column])]
        categorical_columns = [column for column in common_columns if column not in numeric_columns]
        if not common_columns:
            return []

        prompt = self._build_prompt(
            bundle=bundle,
            profile=profile,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            top_heuristics=top_heuristics,
        )

        all_candidates: list[FeatureCandidate] = []
        seen_signatures: set[tuple[str, tuple[str, ...]]] = set()
        for attempt in range(1, DEFAULT_CONFIG.llm_plan_max_attempts + 1):
            try:
                response = self.client.invoke(prompt)
            except Exception as error:
                logger.warning("LLM attempt %s failed: %s", attempt, error)
                continue
            response_text = self._response_to_text(response)
            specs = self._parse_specs(
                response_text=response_text,
                known_columns=set(common_columns),
                numeric_columns=set(numeric_columns),
                categorical_columns=set(categorical_columns),
            )
            for spec in specs:
                signature = (spec.operation, spec.columns)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                candidate = self._apply_spec(
                    spec=spec,
                    train_context=train_context,
                    test_context=test_context,
                    target=target,
                    used_names=used_names,
                )
                if candidate is not None:
                    candidate.metadata["llm_provider"] = self.provider
                    candidate.metadata["plan_source"] = "llm"
                    all_candidates.append(candidate)

        ranked = sorted(all_candidates, key=lambda item: (-float(item.proxy_score), float(item.compute_cost), item.name))
        return ranked[: DEFAULT_CONFIG.max_features * 2]

    def _build_prompt(
        self,
        bundle: DataBundle,
        profile: DatasetProfile,
        numeric_columns: list[str],
        categorical_columns: list[str],
        top_heuristics: list[dict[str, Any]],
    ) -> str:
        readme_fragment = bundle.data_readme[:2500] if bundle.data_readme else "No readme provided."
        numeric_preview = ", ".join(numeric_columns[:24]) if numeric_columns else "none"
        categorical_preview = ", ".join(categorical_columns[:24]) if categorical_columns else "none"
        heuristics_preview = ", ".join(
            f"{item['name']}[{item['family']}]={item['proxy']:.4f}"
            for item in top_heuristics
        ) or "none"
        table_roles = ", ".join(
            f"{name}:{table.role}"
            for name, table in sorted(profile.table_profiles.items())
        )
        return f"""
You are proposing feature ideas for binary classification.
Return JSON only, no markdown.

Dataset profile:
- dataset_type: {profile.dataset_type}
- group_column: {profile.group_column or "none"}
- time_column: {profile.time_column or "none"}
- table_roles: {table_roles}
- join_plan_count: {len(profile.join_plans)}

Top heuristic signals:
{heuristics_preview}

Allowed operations:
- row_mean
- row_std
- ratio
- difference
- log1p_abs
- missing_ratio
- cat_freq
- text_len
- is_missing

Constraints:
- return at most {DEFAULT_CONFIG.max_features * 2} candidate features
- use only listed columns
- prefer features that are different from top heuristic signals
- feature names must be short snake_case

Numeric columns:
{numeric_preview}

Categorical or text columns:
{categorical_preview}

Readme excerpt:
{readme_fragment}

Output schema:
{{
  "features": [
    {{"name": "feature_name", "operation": "ratio", "columns": ["col_a", "col_b"]}}
  ]
}}
"""

    def _parse_specs(
        self,
        response_text: str,
        known_columns: set[str],
        numeric_columns: set[str],
        categorical_columns: set[str],
    ) -> list[LlmFeatureSpec]:
        parsed = safe_json_load(response_text)
        if not isinstance(parsed, dict):
            return []
        items = parsed.get("features", [])
        if not isinstance(items, list):
            return []
        specs: list[LlmFeatureSpec] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            operation = str(item.get("operation", "")).strip()
            columns = tuple(str(column) for column in item.get("columns", []) if str(column))
            spec = self._validate_spec(
                name=name,
                operation=operation,
                columns=columns,
                known_columns=known_columns,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
            )
            if spec is not None:
                specs.append(spec)
        return specs

    def _validate_spec(
        self,
        name: str,
        operation: str,
        columns: tuple[str, ...],
        known_columns: set[str],
        numeric_columns: set[str],
        categorical_columns: set[str],
    ) -> LlmFeatureSpec | None:
        if not name or operation not in LLM_ALLOWED_OPERATIONS or not columns:
            return None
        if any(column not in known_columns for column in columns):
            return None

        rule = LLM_OPERATION_RULES[operation]
        dedup_columns = tuple(dict.fromkeys(columns))
        if not (rule.min_columns <= len(dedup_columns) <= rule.max_columns):
            return None
        if rule.column_type == "numeric" and any(column not in numeric_columns for column in dedup_columns):
            return None
        if rule.column_type == "categorical" and any(column not in categorical_columns for column in dedup_columns):
            return None
        return LlmFeatureSpec(name=name, operation=operation, columns=dedup_columns)

    def _apply_spec(
        self,
        spec: LlmFeatureSpec,
        train_context: pd.DataFrame,
        test_context: pd.DataFrame,
        target: pd.Series,
        used_names: set[str],
    ) -> FeatureCandidate | None:
        operation = spec.operation
        columns = list(spec.columns)
        if operation == "row_mean":
            train = train_context[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            test = test_context[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        elif operation == "row_std":
            train = train_context[columns].apply(pd.to_numeric, errors="coerce").std(axis=1).fillna(0.0)
            test = test_context[columns].apply(pd.to_numeric, errors="coerce").std(axis=1).fillna(0.0)
        elif operation == "ratio":
            train = pd.to_numeric(train_context[columns[0]], errors="coerce") / pd.to_numeric(train_context[columns[1]], errors="coerce").replace(0, np.nan)
            test = pd.to_numeric(test_context[columns[0]], errors="coerce") / pd.to_numeric(test_context[columns[1]], errors="coerce").replace(0, np.nan)
        elif operation == "difference":
            train = pd.to_numeric(train_context[columns[0]], errors="coerce") - pd.to_numeric(train_context[columns[1]], errors="coerce")
            test = pd.to_numeric(test_context[columns[0]], errors="coerce") - pd.to_numeric(test_context[columns[1]], errors="coerce")
        elif operation == "log1p_abs":
            train_raw = pd.to_numeric(train_context[columns[0]], errors="coerce")
            test_raw = pd.to_numeric(test_context[columns[0]], errors="coerce")
            train = np.sign(train_raw) * np.log1p(np.abs(train_raw))
            test = np.sign(test_raw) * np.log1p(np.abs(test_raw))
        elif operation == "missing_ratio":
            train = train_context[columns].isna().mean(axis=1)
            test = test_context[columns].isna().mean(axis=1)
        elif operation == "cat_freq":
            freq = train_context[columns[0]].fillna("__nan__").astype(str).value_counts(normalize=True, dropna=False)
            train = train_context[columns[0]].fillna("__nan__").astype(str).map(freq).fillna(0.0)
            test = test_context[columns[0]].fillna("__nan__").astype(str).map(freq).fillna(0.0)
        elif operation == "text_len":
            train = train_context[columns[0]].fillna("").astype(str).str.len()
            test = test_context[columns[0]].fillna("").astype(str).str.len()
        elif operation == "is_missing":
            train = train_context[columns[0]].isna().astype("int8")
            test = test_context[columns[0]].isna().astype("int8")
        else:
            return None

        candidate_name = normalize_feature_name(spec.name)
        suffix = 2
        while candidate_name in used_names:
            candidate_name = f"{normalize_feature_name(spec.name)}_{suffix}"
            suffix += 1
        used_names.add(candidate_name)

        if pd.api.types.is_numeric_dtype(train) and pd.api.types.is_numeric_dtype(test):
            train = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
            test = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            train = train.fillna("__nan__").astype(str)
            test = test.fillna("__nan__").astype(str)

        if train.isna().all() or test.isna().all() or train.nunique(dropna=True) < 2:
            return None
        return FeatureCandidate(
            name=candidate_name,
            train_feature=train.reset_index(drop=True),
            test_feature=test.reset_index(drop=True),
            source_family="llm_planner",
            proxy_score=score_single_feature_proxy(train, target),
            missing_ratio=float(np.mean([train.isna().mean(), test.isna().mean()])) if hasattr(train, "isna") else 0.0,
            compute_cost=0.42,
            metadata={"join_complexity": 0.5, "llm_operation": operation},
        )

    @staticmethod
    def _response_to_text(response: object) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "\n".join(parts)
        return str(content)
