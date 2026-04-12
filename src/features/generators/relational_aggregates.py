from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet


@dataclass
class JoinPlan:
    table_name: str
    key_column: str
    coverage_score: float
    hint_score: float
    total_score: float


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
            aggregated = self._aggregate_aux_table(
                aux_df=aux_df,
                key_column=plan.key_column,
                table_prefix=plan.table_name,
                forbidden_columns={bundle.target_column},
            )
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
                    "dataset_type": bundle.schema_context.dataset_profile.dataset_type,
                    "schema_recommended_joins": dict(bundle.schema_context.recommended_joins),
                    "join_scores": {
                        plan.table_name: {
                            "coverage": round(plan.coverage_score, 6),
                            "hint": round(plan.hint_score, 6),
                            "total": round(plan.total_score, 6),
                        }
                        for plan in join_plans
                    },
                },
            )
        ]

    def _build_join_plans(self, bundle: DataBundle) -> list[JoinPlan]:
        base_columns = set(bundle.train.columns) & set(bundle.test.columns)
        plans: list[JoinPlan] = []
        readme_text = (bundle.data_readme or "").lower()
        recommended_joins = bundle.schema_context.recommended_joins
        dictionary_hints = bundle.schema_context.column_descriptions

        for table_name in sorted(bundle.aux_tables.keys()):
            aux_df = bundle.aux_tables[table_name]
            common_cols = sorted([column for column in aux_df.columns if column in base_columns])
            if not common_cols:
                continue

            scored_candidates: list[tuple[str, float, float, float]] = []
            for column in common_cols:
                coverage_train = self._join_coverage_score(
                    base_values=bundle.train[column],
                    aux_values=aux_df[column],
                )
                coverage_test = self._join_coverage_score(
                    base_values=bundle.test[column],
                    aux_values=aux_df[column],
                )
                coverage = min(coverage_train, coverage_test)
                hint = self._readme_hint_score(
                    readme_text=readme_text,
                    table_name=table_name,
                    key_column=column,
                    id_column=bundle.id_column,
                )
                prior = self._column_name_prior(column=column, id_column=bundle.id_column)
                schema_bonus = 0.0
                if recommended_joins.get(table_name) == column:
                    schema_bonus += 0.35
                if self._is_schema_supported_join(bundle=bundle, table_name=table_name, key_column=column):
                    schema_bonus += 0.10
                if self._dictionary_mentions_key(dictionary_hints, column):
                    schema_bonus += 0.08

                total = 0.62 * coverage + 0.18 * hint + 0.08 * prior + 0.12 * min(1.0, schema_bonus)
                scored_candidates.append((column, coverage, hint, total))

            if not scored_candidates:
                continue

            best_key, coverage_score, hint_score, total_score = max(
                scored_candidates,
                key=lambda item: item[3],
            )

            # Conservative gate: keep only meaningful join plans.
            is_schema_recommended = recommended_joins.get(table_name) == best_key
            if coverage_score <= 0.03 and hint_score < 0.6 and not is_schema_recommended:
                continue
            if total_score < 0.08:
                continue

            plans.append(
                JoinPlan(
                    table_name=table_name,
                    key_column=best_key,
                    coverage_score=coverage_score,
                    hint_score=hint_score,
                    total_score=total_score,
                )
            )

        return sorted(
            plans,
            key=lambda plan: (-plan.total_score, plan.table_name, plan.key_column),
        )

    @staticmethod
    def _dictionary_mentions_key(column_descriptions: dict[str, str], key_column: str) -> bool:
        description = column_descriptions.get(key_column.lower(), "").lower()
        if not description:
            return False
        return any(marker in description for marker in ("идентификатор", "identifier", "foreign key", "ключ"))

    @staticmethod
    def _is_schema_supported_join(bundle: DataBundle, table_name: str, key_column: str) -> bool:
        for edge in bundle.schema_context.join_edges:
            if edge.left_table == "train" and edge.right_table == table_name and edge.left_key == key_column:
                return True
            if edge.right_table == "train" and edge.left_table == table_name and edge.right_key == key_column:
                return True
        return False

    @staticmethod
    def _join_coverage_score(base_values: pd.Series, aux_values: pd.Series) -> float:
        base_unique = set(base_values.dropna().astype(str).unique())
        if not base_unique:
            return 0.0
        aux_unique = set(aux_values.dropna().astype(str).unique())
        return len(base_unique & aux_unique) / len(base_unique)

    def _readme_hint_score(
        self,
        *,
        readme_text: str,
        table_name: str,
        key_column: str,
        id_column: str,
    ) -> float:
        if not readme_text:
            return 0.0

        table_variants = self._text_variants(table_name)
        key_variants = self._text_variants(key_column)
        if not table_variants or not key_variants:
            return 0.0

        score = 0.0
        keywords = (
            "join",
            "key",
            "foreign key",
            "связ",
            "ключ",
            "внешн",
            "идентифик",
            "id",
        )

        if any(variant in readme_text for variant in table_variants) and any(
            variant in readme_text for variant in key_variants
        ):
            score += 0.25

        table_pattern = "|".join(re.escape(variant) for variant in table_variants if variant)
        key_pattern = "|".join(re.escape(variant) for variant in key_variants if variant)
        if table_pattern and key_pattern:
            proximity_patterns = [
                rf"(?:{table_pattern}).{{0,140}}(?:{key_pattern})",
                rf"(?:{key_pattern}).{{0,140}}(?:{table_pattern})",
            ]
            for pattern in proximity_patterns:
                if re.search(pattern, readme_text, flags=re.IGNORECASE | re.DOTALL):
                    score += 0.25
                    break

        for keyword in keywords:
            if keyword in readme_text and any(variant in readme_text for variant in key_variants):
                score += 0.05
                break

        if key_column.lower() == id_column.lower():
            score += 0.1

        return float(min(1.0, score))

    @staticmethod
    def _column_name_prior(column: str, id_column: str) -> float:
        name = column.lower()
        prior = 0.0
        if name == id_column.lower():
            prior += 1.0
        if name.endswith("_id"):
            prior += 0.8
        if "id" in name:
            prior += 0.4
        if "key" in name:
            prior += 0.3
        return float(min(1.0, prior))

    @staticmethod
    def _text_variants(value: str) -> list[str]:
        raw = value.strip().lower()
        if not raw:
            return []
        variants = {
            raw,
            raw.replace("_", " "),
            re.sub(r"[^a-zа-я0-9_ ]+", " ", raw).strip(),
            re.sub(r"[^a-zа-я0-9]+", "", raw),
        }
        return [variant for variant in variants if variant]

    def _aggregate_aux_table(
        self,
        aux_df: pd.DataFrame,
        key_column: str,
        table_prefix: str,
        forbidden_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        if key_column not in aux_df.columns:
            return pd.DataFrame()

        safe_prefix = table_prefix.replace(" ", "_").lower()
        working = aux_df.copy()
        forbidden_normalized = {column.lower() for column in (forbidden_columns or set()) if column}

        numeric_cols = sorted(
            [
            column
            for column in working.columns
            if column != key_column
            and column.lower() not in forbidden_normalized
            and pd.api.types.is_numeric_dtype(working[column])
            ]
        )[:4]
        categorical_cols = sorted(
            [
            column
            for column in working.columns
            if column != key_column
            and column.lower() not in forbidden_normalized
            and not pd.api.types.is_numeric_dtype(working[column])
            ]
        )[:2]

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
