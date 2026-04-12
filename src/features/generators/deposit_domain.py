from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet


class DepositDomainGenerator:
    name = "deposit_domain"

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        if bundle.schema_context.dataset_profile.dataset_type != "deposit":
            return []

        joined_train, joined_test, used_tables = self._build_joined_context(bundle)
        if joined_train.empty or joined_test.empty:
            return []

        target = bundle.train[bundle.target_column]
        result: list[FeatureSet] = []

        robust_train, robust_test = self._build_robust_features(joined_train, joined_test)
        robust_cols = self._select_diverse_columns(robust_train, target, top_k=max_features)
        if robust_cols:
            result.append(
                FeatureSet(
                    name=f"{self.name}_robust",
                    train_features=robust_train[robust_cols].copy(),
                    test_features=robust_test[robust_cols].copy(),
                    description="Robust row-wise statistics from deposit-related joins.",
                    metadata={
                        "dataset_type": bundle.schema_context.dataset_profile.dataset_type,
                        "tables_used": used_tables,
                        "family": "robust",
                    },
                )
            )

        interaction_train, interaction_test = self._build_interaction_features(joined_train, joined_test)
        interaction_cols = self._select_diverse_columns(interaction_train, target, top_k=max_features)
        if interaction_cols:
            result.append(
                FeatureSet(
                    name=f"{self.name}_interaction",
                    train_features=interaction_train[interaction_cols].copy(),
                    test_features=interaction_test[interaction_cols].copy(),
                    description="Numeric interaction features for deposit profile.",
                    metadata={
                        "dataset_type": bundle.schema_context.dataset_profile.dataset_type,
                        "tables_used": used_tables,
                        "family": "interaction",
                    },
                )
            )

        category_train, category_test = self._build_categorical_profile(joined_train, joined_test)
        category_cols = self._select_diverse_columns(category_train, target, top_k=max_features)
        if category_cols:
            result.append(
                FeatureSet(
                    name=f"{self.name}_categorical",
                    train_features=category_train[category_cols].copy(),
                    test_features=category_test[category_cols].copy(),
                    description="Frequency/rarity profile from categorical columns in deposit data.",
                    metadata={
                        "dataset_type": bundle.schema_context.dataset_profile.dataset_type,
                        "tables_used": used_tables,
                        "family": "categorical",
                    },
                )
            )

        return result

    def _build_joined_context(self, bundle: DataBundle) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        base_columns = sorted(set(bundle.train.columns) & set(bundle.test.columns))
        base_columns = [column for column in base_columns if column != bundle.target_column]
        train_joined = bundle.train[base_columns].copy()
        test_joined = bundle.test[base_columns].copy()

        prioritized_tables = self._prioritize_tables(bundle)
        used_tables: list[str] = []
        for table_name in prioritized_tables:
            aux_df = bundle.aux_tables[table_name]
            join_key = self._pick_join_key(bundle=bundle, table_name=table_name, aux_df=aux_df, base_columns=base_columns)
            if not join_key:
                continue
            aggregated = self._aggregate_aux_table(
                aux_df=aux_df,
                join_key=join_key,
                table_prefix=table_name,
                forbidden_columns={bundle.target_column},
            )
            if aggregated.empty:
                continue

            train_joined = train_joined.merge(aggregated, on=join_key, how="left")
            test_joined = test_joined.merge(aggregated, on=join_key, how="left")
            used_tables.append(f"{table_name}:{join_key}")

        return train_joined, test_joined, used_tables

    @staticmethod
    def _prioritize_tables(bundle: DataBundle) -> list[str]:
        tables = sorted(bundle.aux_tables.keys())
        return sorted(
            tables,
            key=lambda name: (
                0 if "client" in name.lower() else 1,
                0 if "data" in name.lower() else 1,
                name,
            ),
        )

    @staticmethod
    def _pick_join_key(
        *,
        bundle: DataBundle,
        table_name: str,
        aux_df: pd.DataFrame,
        base_columns: list[str],
    ) -> str | None:
        preferred = bundle.schema_context.recommended_joins.get(table_name)
        if preferred and preferred in aux_df.columns:
            return preferred
        if bundle.id_column in aux_df.columns:
            return bundle.id_column

        candidates = [column for column in base_columns if column in aux_df.columns and column != bundle.target_column]
        if not candidates:
            return None
        return candidates[0]

    @staticmethod
    def _aggregate_aux_table(
        *,
        aux_df: pd.DataFrame,
        join_key: str,
        table_prefix: str,
        forbidden_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        if join_key not in aux_df.columns:
            return pd.DataFrame()

        forbidden = {column.lower() for column in (forbidden_columns or set()) if column}
        safe_prefix = table_prefix.lower().replace(" ", "_")
        working = aux_df.copy()

        numeric_cols = sorted(
            [
                column
                for column in working.columns
                if column != join_key
                and column.lower() not in forbidden
                and pd.api.types.is_numeric_dtype(working[column])
            ]
        )[:8]
        categorical_cols = sorted(
            [
                column
                for column in working.columns
                if column != join_key
                and column.lower() not in forbidden
                and not pd.api.types.is_numeric_dtype(working[column])
            ]
        )[:4]

        grouped = working.groupby(join_key, dropna=False)
        result = grouped.size().rename(f"gen_dep_{safe_prefix}_row_count").to_frame()

        if numeric_cols:
            agg = grouped[numeric_cols].agg(["mean", "median", "std", "min", "max"])
            agg.columns = [f"gen_dep_{safe_prefix}_{column}_{agg_name}" for column, agg_name in agg.columns]
            result = result.join(agg, how="left")

        for column in categorical_cols:
            nunique = grouped[column].nunique(dropna=True).rename(f"gen_dep_{safe_prefix}_{column}_nunique")
            mode_share = (
                grouped[column]
                .apply(
                    lambda x: float(
                        x.fillna("__nan__").astype(str).value_counts(normalize=True, dropna=False).iloc[0]
                    )
                    if len(x) > 0
                    else 0.0
                )
                .rename(f"gen_dep_{safe_prefix}_{column}_mode_share")
            )
            result = result.join(nunique, how="left")
            result = result.join(mode_share, how="left")

        return result.reset_index()

    def _build_robust_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        numeric = self._select_source_numeric(train_df)
        if not numeric:
            return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)

        train_num = train_df[numeric].apply(pd.to_numeric, errors="coerce")
        test_num = test_df[numeric].apply(pd.to_numeric, errors="coerce")

        train_result = pd.DataFrame(index=train_df.index)
        test_result = pd.DataFrame(index=test_df.index)

        train_result["gen_dep_row_mean"] = train_num.mean(axis=1)
        test_result["gen_dep_row_mean"] = test_num.mean(axis=1)

        train_result["gen_dep_row_median"] = train_num.median(axis=1)
        test_result["gen_dep_row_median"] = test_num.median(axis=1)

        train_result["gen_dep_row_std"] = train_num.std(axis=1).fillna(0.0)
        test_result["gen_dep_row_std"] = test_num.std(axis=1).fillna(0.0)

        q75_train = train_num.quantile(0.75, axis=1)
        q25_train = train_num.quantile(0.25, axis=1)
        q75_test = test_num.quantile(0.75, axis=1)
        q25_test = test_num.quantile(0.25, axis=1)

        train_result["gen_dep_row_iqr"] = q75_train - q25_train
        test_result["gen_dep_row_iqr"] = q75_test - q25_test

        train_result["gen_dep_row_range"] = train_num.max(axis=1) - train_num.min(axis=1)
        test_result["gen_dep_row_range"] = test_num.max(axis=1) - test_num.min(axis=1)

        train_result["gen_dep_row_missing_ratio"] = train_num.isna().mean(axis=1)
        test_result["gen_dep_row_missing_ratio"] = test_num.isna().mean(axis=1)

        return self._sanitize_pair(train_result, test_result)

    def _build_interaction_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        numeric = self._select_source_numeric(train_df)
        if len(numeric) < 2:
            return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)

        source_cols = numeric[:6]
        train_result = pd.DataFrame(index=train_df.index)
        test_result = pd.DataFrame(index=test_df.index)

        for left, right in list(combinations(source_cols, 2))[:6]:
            left_train = pd.to_numeric(train_df[left], errors="coerce")
            right_train = pd.to_numeric(train_df[right], errors="coerce")
            left_test = pd.to_numeric(test_df[left], errors="coerce")
            right_test = pd.to_numeric(test_df[right], errors="coerce")

            safe_right_train = right_train.replace(0, np.nan)
            safe_right_test = right_test.replace(0, np.nan)

            short_left = self._short_name(left)
            short_right = self._short_name(right)
            ratio_name = f"gen_dep_ratio_{short_left}_{short_right}"
            diff_name = f"gen_dep_diff_{short_left}_{short_right}"

            train_result[ratio_name] = left_train / safe_right_train
            test_result[ratio_name] = left_test / safe_right_test
            train_result[diff_name] = left_train - right_train
            test_result[diff_name] = left_test - right_test

        lead_col = source_cols[0]
        train_lead = pd.to_numeric(train_df[lead_col], errors="coerce")
        test_lead = pd.to_numeric(test_df[lead_col], errors="coerce")
        train_result[f"gen_dep_log1p_abs_{self._short_name(lead_col)}"] = np.sign(train_lead) * np.log1p(np.abs(train_lead))
        test_result[f"gen_dep_log1p_abs_{self._short_name(lead_col)}"] = np.sign(test_lead) * np.log1p(np.abs(test_lead))

        return self._sanitize_pair(train_result, test_result)

    def _build_categorical_profile(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        categorical = [
            column
            for column in sorted(train_df.columns)
            if column.lower() != "target"
            and not column.lower().endswith("_id")
            and (
                pd.api.types.is_object_dtype(train_df[column])
                or pd.api.types.is_categorical_dtype(train_df[column])
            )
        ][:4]
        if not categorical:
            return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)

        train_result = pd.DataFrame(index=train_df.index)
        test_result = pd.DataFrame(index=test_df.index)
        for column in categorical:
            freq = train_df[column].fillna("__nan__").astype(str).value_counts(normalize=True, dropna=False)
            encoded_train = train_df[column].fillna("__nan__").astype(str).map(freq).fillna(0.0)
            encoded_test = test_df[column].fillna("__nan__").astype(str).map(freq).fillna(0.0)
            short = self._short_name(column)
            train_result[f"gen_dep_freq_{short}"] = encoded_train
            test_result[f"gen_dep_freq_{short}"] = encoded_test
            train_result[f"gen_dep_rare_{short}"] = (encoded_train <= 0.01).astype("int8")
            test_result[f"gen_dep_rare_{short}"] = (encoded_test <= 0.01).astype("int8")

        return self._sanitize_pair(train_result, test_result)

    def _select_diverse_columns(self, features: pd.DataFrame, target: pd.Series, top_k: int) -> list[str]:
        if features.empty:
            return []

        y = target.reset_index(drop=True)
        signals: list[tuple[str, float]] = []
        for column in features.columns:
            signal = self._single_feature_signal(features[column], y)
            signals.append((column, signal))

        ranked = sorted(signals, key=lambda item: (item[1], item[0]), reverse=True)
        selected: list[str] = []
        for column, score in ranked:
            if len(selected) >= top_k:
                break
            if score <= 0:
                continue
            if self._is_redundant(features, column, selected):
                continue
            selected.append(column)

        if selected:
            return selected
        return list(features.columns[:top_k])

    @staticmethod
    def _is_redundant(features: pd.DataFrame, candidate: str, selected: list[str], threshold: float = 0.985) -> bool:
        candidate_series = features[candidate]
        for existing in selected:
            existing_series = features[existing]
            if pd.api.types.is_numeric_dtype(candidate_series) and pd.api.types.is_numeric_dtype(existing_series):
                a = pd.to_numeric(candidate_series, errors="coerce").fillna(-999.0)
                b = pd.to_numeric(existing_series, errors="coerce").fillna(-999.0)
                if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
                    continue
                corr = float(a.corr(b))
                if np.isfinite(corr) and abs(corr) >= threshold:
                    return True
            else:
                a = candidate_series.fillna("__nan__").astype(str)
                b = existing_series.fillna("__nan__").astype(str)
                if bool((a == b).all()):
                    return True
        return False

    @staticmethod
    def _single_feature_signal(feature: pd.Series, target: pd.Series) -> float:
        if feature.nunique(dropna=True) < 2:
            return 0.0

        if pd.api.types.is_numeric_dtype(feature):
            x = pd.to_numeric(feature, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-999.0)
        else:
            x = pd.Series(pd.factorize(feature.fillna("__nan__").astype(str))[0], index=feature.index)

        try:
            auc = roc_auc_score(target, x)
            return abs(float(auc) - 0.5)
        except Exception:
            return 0.0

    @staticmethod
    def _select_source_numeric(dataframe: pd.DataFrame) -> list[str]:
        numeric = [
            column
            for column in sorted(dataframe.columns)
            if not column.lower().endswith("_id")
            and pd.api.types.is_numeric_dtype(dataframe[column])
            and dataframe[column].nunique(dropna=True) > 2
        ]
        return numeric[:14]

    @staticmethod
    def _sanitize_pair(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = train_df.replace([np.inf, -np.inf], np.nan)
        test = test_df.replace([np.inf, -np.inf], np.nan)
        return train, test

    @staticmethod
    def _short_name(value: str) -> str:
        normalized = (
            value.lower()
            .replace("gen_dep_", "")
            .replace("gen_", "")
            .replace("__", "_")
            .replace("-", "_")
        )
        return "_".join([chunk for chunk in normalized.split("_") if chunk][:3])[:30] or "feature"
