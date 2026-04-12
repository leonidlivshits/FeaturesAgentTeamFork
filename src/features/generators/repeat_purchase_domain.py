from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet


class RepeatPurchaseDomainGenerator:
    name = "repeat_purchase_domain"

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        if bundle.schema_context.dataset_profile.dataset_type != "repeat_purchase":
            return []

        user_col = self._resolve_user_column(bundle)
        product_col = self._resolve_product_column(bundle)
        if not user_col or not product_col:
            return []

        train_context = bundle.train.copy()
        test_context = bundle.test.copy()

        users_features = self._build_users_features(bundle, user_col=user_col)
        if not users_features.empty:
            train_context = train_context.merge(users_features, on=user_col, how="left")
            test_context = test_context.merge(users_features, on=user_col, how="left")

        product_features = self._build_product_features(bundle, product_col=product_col)
        if not product_features.empty:
            train_context = train_context.merge(product_features, on=product_col, how="left")
            test_context = test_context.merge(product_features, on=product_col, how="left")

        pair_features = self._build_user_product_features(
            bundle,
            user_col=user_col,
            product_col=product_col,
        )
        if not pair_features.empty:
            train_context = train_context.merge(pair_features, on=[user_col, product_col], how="left")
            test_context = test_context.merge(pair_features, on=[user_col, product_col], how="left")

        temporal_features = self._build_temporal_user_features(bundle, user_col=user_col)
        if not temporal_features.empty:
            train_context = train_context.merge(temporal_features, on=user_col, how="left")
            test_context = test_context.merge(temporal_features, on=user_col, how="left")

        core_train, core_test = self._build_core_features(
            train_df=train_context,
            test_df=test_context,
            user_col=user_col,
            product_col=product_col,
        )
        temporal_train, temporal_test = self._build_temporal_features(
            train_df=train_context,
            test_df=test_context,
            user_col=user_col,
            product_col=product_col,
        )

        target = bundle.train[bundle.target_column]
        feature_sets: list[FeatureSet] = []

        core_cols = self._select_diverse_columns(core_train, target, top_k=max_features)
        if core_cols:
            feature_sets.append(
                FeatureSet(
                    name=f"{self.name}_core",
                    train_features=core_train[core_cols].copy(),
                    test_features=core_test[core_cols].copy(),
                    description="User/product/pair features for repeat-purchase dataset.",
                    metadata={
                        "dataset_type": bundle.schema_context.dataset_profile.dataset_type,
                        "family": "core",
                        "user_column": user_col,
                        "product_column": product_col,
                    },
                )
            )

        temporal_cols = self._select_diverse_columns(temporal_train, target, top_k=max_features)
        if temporal_cols:
            feature_sets.append(
                FeatureSet(
                    name=f"{self.name}_temporal",
                    train_features=temporal_train[temporal_cols].copy(),
                    test_features=temporal_test[temporal_cols].copy(),
                    description="Order-time and behavioral features for repeat-purchase dataset.",
                    metadata={
                        "dataset_type": bundle.schema_context.dataset_profile.dataset_type,
                        "family": "temporal",
                        "user_column": user_col,
                        "product_column": product_col,
                    },
                )
            )

        return feature_sets

    def _resolve_user_column(self, bundle: DataBundle) -> str | None:
        detected = bundle.schema_context.dataset_profile.detected_entities.get("user")
        if detected and detected in bundle.train.columns and detected in bundle.test.columns:
            return detected
        for candidate in ("user_id", "client_id", "customer_id"):
            if candidate in bundle.train.columns and candidate in bundle.test.columns:
                return candidate
        return None

    def _resolve_product_column(self, bundle: DataBundle) -> str | None:
        detected = bundle.schema_context.dataset_profile.detected_entities.get("product")
        if detected and detected in bundle.train.columns and detected in bundle.test.columns:
            return detected
        for candidate in ("product_id", "item_id", "sku_id"):
            if candidate in bundle.train.columns and candidate in bundle.test.columns:
                return candidate
        return None

    def _build_users_features(self, bundle: DataBundle, user_col: str) -> pd.DataFrame:
        users_table = bundle.aux_tables.get("users")
        if users_table is None or users_table.empty or user_col not in users_table.columns:
            return pd.DataFrame()

        keep_cols = [column for column in users_table.columns if column == user_col or column.lower() != bundle.target_column.lower()]
        users = users_table[keep_cols].copy()
        rename_map = {column: f"gen_rp_user_{column}" for column in users.columns if column != user_col}
        return users.rename(columns=rename_map)

    def _build_product_features(self, bundle: DataBundle, product_col: str) -> pd.DataFrame:
        order_items = bundle.aux_tables.get("order_items")
        if order_items is None or order_items.empty or product_col not in order_items.columns:
            return pd.DataFrame()

        working = order_items.copy()
        grouped = working.groupby(product_col, dropna=False)
        result = pd.DataFrame(index=grouped.size().index)
        result["gen_rp_product_order_count"] = grouped.size()

        if "reordered" in working.columns:
            result["gen_rp_product_reordered_rate"] = grouped["reordered"].mean()
        if "add_to_cart_order" in working.columns:
            result["gen_rp_product_avg_cart_pos"] = grouped["add_to_cart_order"].mean()
            result["gen_rp_product_cart_pos_std"] = grouped["add_to_cart_order"].std()

        products = bundle.aux_tables.get("products")
        if products is not None and not products.empty and product_col in products.columns:
            meta = products[[column for column in products.columns if column in {product_col, "aisle_id", "department_id"}]].copy()
            result = result.reset_index().merge(meta, on=product_col, how="left")
            for column in ("aisle_id", "department_id"):
                if column in result.columns:
                    result[column] = pd.to_numeric(result[column], errors="coerce")
                    result = result.rename(columns={column: f"gen_rp_product_{column}"})
            return result

        return result.reset_index()

    def _build_user_product_features(self, bundle: DataBundle, user_col: str, product_col: str) -> pd.DataFrame:
        orders = bundle.aux_tables.get("orders")
        order_items = bundle.aux_tables.get("order_items")
        if orders is None or order_items is None:
            return pd.DataFrame()
        if orders.empty or order_items.empty:
            return pd.DataFrame()
        if "order_id" not in orders.columns or "order_id" not in order_items.columns:
            return pd.DataFrame()
        if user_col not in orders.columns or product_col not in order_items.columns:
            return pd.DataFrame()

        merge_cols = [column for column in ("order_id", user_col, "order_number", "days_since_prior_order", "order_dow", "order_hour_of_day", "eval_set") if column in orders.columns]
        orders_reduced = orders[merge_cols].copy()
        if "eval_set" in orders_reduced.columns:
            filtered = orders_reduced[orders_reduced["eval_set"].astype(str).str.lower().isin({"prior", "train"})]
            if not filtered.empty:
                orders_reduced = filtered

        item_cols = [column for column in ("order_id", product_col, "add_to_cart_order", "reordered") if column in order_items.columns]
        item_reduced = order_items[item_cols].copy()
        merged = item_reduced.merge(orders_reduced, on="order_id", how="inner")
        if merged.empty:
            return pd.DataFrame()

        pair_group = merged.groupby([user_col, product_col], dropna=False)
        pair = pair_group.size().rename("gen_rp_pair_order_count").to_frame().reset_index()

        if "reordered" in merged.columns:
            pair["gen_rp_pair_reordered_rate"] = pair_group["reordered"].mean().values
        if "add_to_cart_order" in merged.columns:
            pair["gen_rp_pair_avg_cart_pos"] = pair_group["add_to_cart_order"].mean().values
        if "order_number" in merged.columns:
            pair["gen_rp_pair_last_order_num"] = pair_group["order_number"].max().values
            pair["gen_rp_pair_first_order_num"] = pair_group["order_number"].min().values
            pair["gen_rp_pair_order_num_span"] = (
                pair["gen_rp_pair_last_order_num"] - pair["gen_rp_pair_first_order_num"]
            )
        if "days_since_prior_order" in merged.columns:
            pair["gen_rp_pair_days_since_prior_mean"] = pair_group["days_since_prior_order"].mean().values

        user_orders = (
            merged.groupby(user_col, dropna=False)
            .size()
            .rename("gen_rp_user_total_items")
            .reset_index()
        )
        pair = pair.merge(user_orders, on=user_col, how="left")
        pair["gen_rp_pair_item_share"] = (
            pair["gen_rp_pair_order_count"] / pair["gen_rp_user_total_items"].replace(0, np.nan)
        )

        return pair

    def _build_temporal_user_features(self, bundle: DataBundle, user_col: str) -> pd.DataFrame:
        orders = bundle.aux_tables.get("orders")
        if orders is None or orders.empty or user_col not in orders.columns:
            return pd.DataFrame()

        cols = [column for column in (user_col, "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order") if column in orders.columns]
        working = orders[cols].copy()
        grouped = working.groupby(user_col, dropna=False)
        result = grouped.size().rename("gen_rp_user_order_rows").to_frame()

        if "order_number" in working.columns:
            result["gen_rp_user_max_order_num"] = grouped["order_number"].max()
            result["gen_rp_user_mean_order_num"] = grouped["order_number"].mean()
        if "days_since_prior_order" in working.columns:
            result["gen_rp_user_days_since_prior_mean"] = grouped["days_since_prior_order"].mean()
            result["gen_rp_user_days_since_prior_std"] = grouped["days_since_prior_order"].std()
        if "order_hour_of_day" in working.columns:
            result["gen_rp_user_order_hour_mean"] = grouped["order_hour_of_day"].mean()
        if "order_dow" in working.columns:
            result["gen_rp_user_order_dow_mean"] = grouped["order_dow"].mean()

        return result.reset_index()

    def _build_core_features(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        user_col: str,
        product_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        candidate_prefixes = ("gen_rp_user_", "gen_rp_product_", "gen_rp_pair_")
        numeric_columns = [
            column
            for column in sorted(train_df.columns)
            if column not in {user_col, product_col}
            and any(column.startswith(prefix) for prefix in candidate_prefixes)
            and pd.api.types.is_numeric_dtype(train_df[column])
        ]
        if not numeric_columns:
            return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)

        train = train_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        test = test_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        out_train = pd.DataFrame(index=train_df.index)
        out_test = pd.DataFrame(index=test_df.index)

        out_train["gen_rp_core_mean"] = train.mean(axis=1)
        out_test["gen_rp_core_mean"] = test.mean(axis=1)
        out_train["gen_rp_core_std"] = train.std(axis=1).fillna(0.0)
        out_test["gen_rp_core_std"] = test.std(axis=1).fillna(0.0)
        out_train["gen_rp_core_missing_ratio"] = train.isna().mean(axis=1)
        out_test["gen_rp_core_missing_ratio"] = test.isna().mean(axis=1)

        for col in numeric_columns[:3]:
            short = self._short_name(col)
            out_train[f"gen_rp_log1p_abs_{short}"] = np.sign(train[col]) * np.log1p(np.abs(train[col]))
            out_test[f"gen_rp_log1p_abs_{short}"] = np.sign(test[col]) * np.log1p(np.abs(test[col]))

        if len(numeric_columns) >= 2:
            a, b = numeric_columns[0], numeric_columns[1]
            out_train["gen_rp_core_ratio_main"] = train[a] / train[b].replace(0, np.nan)
            out_test["gen_rp_core_ratio_main"] = test[a] / test[b].replace(0, np.nan)
            out_train["gen_rp_core_diff_main"] = train[a] - train[b]
            out_test["gen_rp_core_diff_main"] = test[a] - test[b]

        return self._sanitize_pair(out_train, out_test)

    def _build_temporal_features(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        user_col: str,
        product_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        out_train = pd.DataFrame(index=train_df.index)
        out_test = pd.DataFrame(index=test_df.index)

        temporal_cols = [
            column
            for column in sorted(train_df.columns)
            if column not in {user_col, product_col}
            and (
                "hour" in column.lower()
                or "dow" in column.lower()
                or "days_since" in column.lower()
                or "order_num" in column.lower()
            )
            and pd.api.types.is_numeric_dtype(train_df[column])
        ]
        if temporal_cols:
            train_temp = train_df[temporal_cols].apply(pd.to_numeric, errors="coerce")
            test_temp = test_df[temporal_cols].apply(pd.to_numeric, errors="coerce")
            out_train["gen_rp_temp_mean"] = train_temp.mean(axis=1)
            out_test["gen_rp_temp_mean"] = test_temp.mean(axis=1)
            out_train["gen_rp_temp_std"] = train_temp.std(axis=1).fillna(0.0)
            out_test["gen_rp_temp_std"] = test_temp.std(axis=1).fillna(0.0)

            if len(temporal_cols) >= 2:
                for left, right in list(combinations(temporal_cols[:4], 2))[:3]:
                    short_left = self._short_name(left)
                    short_right = self._short_name(right)
                    out_train[f"gen_rp_temp_diff_{short_left}_{short_right}"] = train_temp[left] - train_temp[right]
                    out_test[f"gen_rp_temp_diff_{short_left}_{short_right}"] = test_temp[left] - test_temp[right]

        behavioral_cols = [
            column
            for column in sorted(train_df.columns)
            if column.startswith("gen_rp_pair_") and pd.api.types.is_numeric_dtype(train_df[column])
        ]
        if behavioral_cols:
            train_beh = train_df[behavioral_cols].apply(pd.to_numeric, errors="coerce")
            test_beh = test_df[behavioral_cols].apply(pd.to_numeric, errors="coerce")
            out_train["gen_rp_behavior_mean"] = train_beh.mean(axis=1)
            out_test["gen_rp_behavior_mean"] = test_beh.mean(axis=1)

        return self._sanitize_pair(out_train, out_test)

    def _select_diverse_columns(self, features: pd.DataFrame, target: pd.Series, top_k: int) -> list[str]:
        if features.empty:
            return []

        y = target.reset_index(drop=True)
        ranked: list[tuple[str, float]] = []
        for column in features.columns:
            signal = self._single_feature_signal(features[column], y)
            ranked.append((column, signal))
        ranked = sorted(ranked, key=lambda item: (item[1], item[0]), reverse=True)

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
                if bool((candidate_series.fillna("__nan__").astype(str) == existing_series.fillna("__nan__").astype(str)).all()):
                    return True
        return False

    @staticmethod
    def _sanitize_pair(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            train_df.replace([np.inf, -np.inf], np.nan),
            test_df.replace([np.inf, -np.inf], np.nan),
        )

    @staticmethod
    def _short_name(value: str) -> str:
        normalized = value.lower().replace("gen_rp_", "").replace("__", "_").replace("-", "_")
        return "_".join([chunk for chunk in normalized.split("_") if chunk][:3])[:30] or "feature"
