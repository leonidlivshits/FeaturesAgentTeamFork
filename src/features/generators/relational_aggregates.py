from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet


@dataclass
class JoinPlan:
    table_name: str
    key_column: str


class RelationalAggregatesGenerator:
    name = "relational_aggregates"

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        if not bundle.aux_tables:
            return []

        train_joined = pd.DataFrame(index=bundle.train.index)
        test_joined = pd.DataFrame(index=bundle.test.index)
        join_plans = self._build_join_plans(bundle)
        if not join_plans:
            return []

        for plan in join_plans:
            aux_df = bundle.aux_tables[plan.table_name]
            aggregated = self._aggregate_aux_table(aux_df=aux_df, key_column=plan.key_column, table_prefix=plan.table_name)
            if aggregated.empty:
                continue

            train_merged = bundle.train[[plan.key_column]].merge(aggregated, on=plan.key_column, how="left")
            test_merged = bundle.test[[plan.key_column]].merge(aggregated, on=plan.key_column, how="left")

            train_merged = train_merged.drop(columns=[plan.key_column], errors="ignore")
            test_merged = test_merged.drop(columns=[plan.key_column], errors="ignore")

            train_joined = pd.concat([train_joined, train_merged], axis=1)
            test_joined = pd.concat([test_joined, test_merged], axis=1)

        if train_joined.empty:
            return []

        selected_columns = self._select_top_columns(
            features_df=train_joined,
            target=bundle.train[bundle.target_column],
            top_k=max_features,
        )
        if not selected_columns:
            return []

        return [
            FeatureSet(
                name=self.name,
                train_features=train_joined[selected_columns].copy(),
                test_features=test_joined[selected_columns].copy(),
                description="Auto-joined aggregate features from auxiliary tables.",
                metadata={
                    "tables_used": [plan.table_name for plan in join_plans],
                    "join_keys": {plan.table_name: plan.key_column for plan in join_plans},
                },
            )
        ]

    def _build_join_plans(self, bundle: DataBundle) -> list[JoinPlan]:
        base_columns = set(bundle.train.columns) & set(bundle.test.columns)
        plans: list[JoinPlan] = []

        for table_name, aux_df in bundle.aux_tables.items():
            common_cols = [column for column in aux_df.columns if column in base_columns]
            if not common_cols:
                continue

            best_key = max(
                common_cols,
                key=lambda column: self._join_coverage_score(
                    base_values=bundle.train[column],
                    aux_values=aux_df[column],
                ),
            )

            if self._join_coverage_score(bundle.train[best_key], aux_df[best_key]) <= 0.05:
                continue

            plans.append(JoinPlan(table_name=table_name, key_column=best_key))

        return plans

    @staticmethod
    def _join_coverage_score(base_values: pd.Series, aux_values: pd.Series) -> float:
        base_unique = set(base_values.dropna().astype(str).unique())
        if not base_unique:
            return 0.0
        aux_unique = set(aux_values.dropna().astype(str).unique())
        return len(base_unique & aux_unique) / len(base_unique)

    def _aggregate_aux_table(self, aux_df: pd.DataFrame, key_column: str, table_prefix: str) -> pd.DataFrame:
        if key_column not in aux_df.columns:
            return pd.DataFrame()

        safe_prefix = table_prefix.replace(" ", "_").lower()
        working = aux_df.copy()

        numeric_cols = [
            column
            for column in working.columns
            if column != key_column and pd.api.types.is_numeric_dtype(working[column])
        ][:4]
        categorical_cols = [
            column
            for column in working.columns
            if column != key_column and not pd.api.types.is_numeric_dtype(working[column])
        ][:2]

        aggs: dict[str, list[str]] = {}
        for column in numeric_cols:
            aggs[column] = ["mean", "std"]
        for column in categorical_cols:
            aggs[column] = ["nunique"]

        grouped = working.groupby(key_column, dropna=False)
        result = grouped.size().rename(f"gen_{safe_prefix}_row_count").to_frame()

        if aggs:
            agg_df = grouped.agg(aggs)
            agg_df.columns = [f"gen_{safe_prefix}_{column}_{agg}" for column, agg in agg_df.columns]
            result = result.join(agg_df, how="left")

        return result.reset_index()

    def _select_top_columns(self, features_df: pd.DataFrame, target: pd.Series, top_k: int) -> list[str]:
        if features_df.empty:
            return []

        scores: list[tuple[str, float]] = []
        y = target.reset_index(drop=True)

        for column in features_df.columns:
            feature = features_df[column].reset_index(drop=True)
            score = self._single_feature_signal(feature=feature, target=y)
            scores.append((column, score))

        ranked = sorted(scores, key=lambda pair: pair[1], reverse=True)
        selected = [column for column, score in ranked if np.isfinite(score) and score > 0][:top_k]
        if selected:
            return selected

        # Fallback if all single-feature scores are weak or invalid.
        return list(features_df.columns[:top_k])

    @staticmethod
    def _single_feature_signal(feature: pd.Series, target: pd.Series) -> float:
        if feature.nunique(dropna=True) < 2:
            return 0.0

        if pd.api.types.is_numeric_dtype(feature):
            x = pd.to_numeric(feature, errors="coerce").fillna(-999.0)
        else:
            x = pd.Series(pd.factorize(feature.fillna("__nan__").astype(str))[0], index=feature.index)

        try:
            auc = roc_auc_score(target, x)
        except Exception:
            return 0.0
        return abs(float(auc) - 0.5)

