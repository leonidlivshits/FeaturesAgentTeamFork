from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.core.llm import get_gigachat_client
from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet

logger = logging.getLogger(__name__)


ALLOWED_OPERATIONS = {
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
class OperationRule:
    min_columns: int
    max_columns: int
    column_type: str  # "any" | "numeric" | "categorical"


OPERATION_RULES: dict[str, OperationRule] = {
    "row_mean": OperationRule(min_columns=1, max_columns=8, column_type="numeric"),
    "row_std": OperationRule(min_columns=1, max_columns=8, column_type="numeric"),
    "ratio": OperationRule(min_columns=2, max_columns=2, column_type="numeric"),
    "difference": OperationRule(min_columns=2, max_columns=2, column_type="numeric"),
    "log1p_abs": OperationRule(min_columns=1, max_columns=1, column_type="numeric"),
    "missing_ratio": OperationRule(min_columns=1, max_columns=8, column_type="any"),
    "cat_freq": OperationRule(min_columns=1, max_columns=1, column_type="categorical"),
    "text_len": OperationRule(min_columns=1, max_columns=1, column_type="categorical"),
    "is_missing": OperationRule(min_columns=1, max_columns=1, column_type="any"),
}


@dataclass
class FeatureSpec:
    name: str
    operation: str
    columns: list[str]


class LlmGuidedGenerator:
    name = "llm_guided"

    def __init__(self) -> None:
        self.client = get_gigachat_client()

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        common_columns = [
            column
            for column in bundle.train.columns
            if column in bundle.test.columns and column != bundle.id_column
        ]
        if not common_columns:
            return []

        numeric_columns = [
            column
            for column in common_columns
            if pd.api.types.is_numeric_dtype(bundle.train[column])
        ]
        categorical_columns = [column for column in common_columns if column not in numeric_columns]
        if not numeric_columns and not categorical_columns:
            return []

        plan = self._build_plan(
            bundle=bundle,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )
        if not plan:
            return []

        train_features, test_features, feature_operations = self._apply_plan(bundle=bundle, specs=plan)
        if train_features.empty:
            return []

        selected_columns, selection_scores = self._select_top_columns(
            features=train_features,
            target=bundle.train[bundle.target_column],
            top_k=max_features,
        )
        if not selected_columns:
            return []

        selected_operations = [feature_operations.get(column, "unknown") for column in selected_columns]
        logger.info(
            "LLM-guided selected features: %s",
            ", ".join(f"{column}({selection_scores.get(column, 0.0):.4f})" for column in selected_columns),
        )

        return [
            FeatureSet(
                name=self.name,
                train_features=train_features[selected_columns].copy(),
                test_features=test_features[selected_columns].copy(),
                description="Feature set proposed by LLM and computed by safe local operators.",
                metadata={
                    "selected_operations": selected_operations,
                    "selected_feature_scores": {column: selection_scores.get(column, 0.0) for column in selected_columns},
                    "candidate_plan_size": len(plan),
                },
            )
        ]

    def _build_plan(
        self,
        bundle: DataBundle,
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_features: int,
    ) -> list[FeatureSpec]:
        llm_plan = self._request_llm_plan(
            bundle=bundle,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )
        if llm_plan:
            return self._normalize_plan(llm_plan, max_features=max_features)
        fallback = self._fallback_plan(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )
        return self._normalize_plan(fallback, max_features=max_features)

    def _request_llm_plan(
        self,
        bundle: DataBundle,
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_features: int,
    ) -> list[FeatureSpec]:
        if self.client is None:
            return []

        prompt = self._build_prompt(
            data_readme=bundle.data_readme,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )

        try:
            response = self.client.invoke(prompt)
            response_text = self._response_to_text(response)
            return self._parse_plan(
                response_text=response_text,
                known_columns=set(numeric_columns + categorical_columns),
                numeric_columns=set(numeric_columns),
                categorical_columns=set(categorical_columns),
            )
        except Exception as error:
            logger.warning("LLM plan generation failed, fallback will be used: %s", error)
            return []

    @staticmethod
    def _build_prompt(
        data_readme: str,
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_features: int,
    ) -> str:
        readme_fragment = data_readme[:3000] if data_readme else "No data readme provided."
        numeric_preview = ", ".join(numeric_columns[:30]) if numeric_columns else "none"
        categorical_preview = ", ".join(categorical_columns[:30]) if categorical_columns else "none"
        return f"""
You are designing features for binary classification on tabular data.
You MUST return JSON only without markdown.

Allowed operations:
- row_mean: columns>=1 numeric
- row_std: columns>=1 numeric
- ratio: columns=2 numeric
- difference: columns=2 numeric
- log1p_abs: columns=1 numeric
- missing_ratio: columns>=1 any
- cat_freq: columns=1 categorical
- text_len: columns=1 categorical
- is_missing: columns=1 any

Constraints:
- Return at most {max_features} features.
- Use only listed columns.
- Feature names must be short and in snake_case.

Numeric columns: {numeric_preview}
Categorical columns: {categorical_preview}

Data readme:
{readme_fragment}

Output schema:
{{
  "features": [
    {{"name": "feature_name", "operation": "ratio", "columns": ["col_a", "col_b"]}}
  ]
}}
"""

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

    def _parse_plan(
        self,
        response_text: str,
        known_columns: set[str],
        numeric_columns: set[str],
        categorical_columns: set[str],
    ) -> list[FeatureSpec]:
        parsed = self._safe_json_load(response_text)
        if not isinstance(parsed, dict):
            return []
        items = parsed.get("features", [])
        if not isinstance(items, list):
            return []

        result: list[FeatureSpec] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            operation = str(item.get("operation", "")).strip()
            columns = item.get("columns", [])
            if not isinstance(columns, list):
                continue
            columns = [str(column) for column in columns]
            spec = self._validate_spec(
                name=name,
                operation=operation,
                columns=columns,
                known_columns=known_columns,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
            )
            if spec is not None:
                result.append(spec)
        return result

    @staticmethod
    def _safe_json_load(text: str) -> object:
        payload = text.strip()
        payload = payload.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(payload)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", payload)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

    def _validate_spec(
        self,
        name: str,
        operation: str,
        columns: list[str],
        known_columns: set[str],
        numeric_columns: set[str],
        categorical_columns: set[str],
    ) -> FeatureSpec | None:
        if not name or operation not in ALLOWED_OPERATIONS:
            return None
        if not columns:
            return None
        if any(column not in known_columns for column in columns):
            return None

        rule = OPERATION_RULES.get(operation)
        if rule is None:
            return None

        dedup_columns = list(dict.fromkeys(columns))
        if not (rule.min_columns <= len(dedup_columns) <= rule.max_columns):
            return None

        if rule.column_type == "numeric" and any(column not in numeric_columns for column in dedup_columns):
            return None
        if rule.column_type == "categorical" and any(column not in categorical_columns for column in dedup_columns):
            return None

        return FeatureSpec(name=name, operation=operation, columns=dedup_columns[: rule.max_columns])

    @staticmethod
    def _fallback_plan(
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_features: int,
    ) -> list[FeatureSpec]:
        specs: list[FeatureSpec] = []

        if numeric_columns:
            specs.append(FeatureSpec(name="num_row_mean", operation="row_mean", columns=numeric_columns[: min(5, len(numeric_columns))]))
            specs.append(FeatureSpec(name="num_row_std", operation="row_std", columns=numeric_columns[: min(5, len(numeric_columns))]))
            specs.append(FeatureSpec(name="num_log1p_main", operation="log1p_abs", columns=[numeric_columns[0]]))

        if len(numeric_columns) >= 2:
            specs.append(FeatureSpec(name="num_ratio_main", operation="ratio", columns=[numeric_columns[0], numeric_columns[1]]))
            specs.append(
                FeatureSpec(name="num_diff_main", operation="difference", columns=[numeric_columns[0], numeric_columns[1]])
            )

        if categorical_columns:
            specs.append(FeatureSpec(name="cat_freq_main", operation="cat_freq", columns=[categorical_columns[0]]))
            specs.append(FeatureSpec(name="cat_len_main", operation="text_len", columns=[categorical_columns[0]]))

        mixed_columns = (numeric_columns[:2] + categorical_columns[:2])[:4]
        if mixed_columns:
            specs.append(FeatureSpec(name="missing_ratio_main", operation="missing_ratio", columns=mixed_columns))
            specs.append(FeatureSpec(name="is_missing_main", operation="is_missing", columns=[mixed_columns[0]]))

        return specs[: max(max_features * 2, max_features)]

    @staticmethod
    def _normalize_plan(specs: list[FeatureSpec], max_features: int) -> list[FeatureSpec]:
        deduplicated: list[FeatureSpec] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()

        for spec in specs:
            signature = (spec.operation, tuple(spec.columns))
            if signature in seen:
                continue
            seen.add(signature)
            deduplicated.append(spec)

        hard_limit = max(max_features * 2, max_features)
        return deduplicated[:hard_limit]

    def _apply_plan(
        self,
        bundle: DataBundle,
        specs: list[FeatureSpec],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
        train_result = pd.DataFrame(index=bundle.train.index)
        test_result = pd.DataFrame(index=bundle.test.index)
        feature_operations: dict[str, str] = {}

        used_names: set[str] = set()
        for spec in specs:
            feature_name = self._safe_feature_name(spec.name, used_names=used_names)
            if not feature_name:
                continue

            try:
                train_series, test_series = self._compute_feature(bundle=bundle, spec=spec)
            except Exception as error:
                logger.debug("Failed to compute feature '%s': %s", feature_name, error)
                continue
            if train_series is None or test_series is None:
                continue

            train_series, test_series = self._sanitize_feature_pair(train_series, test_series)
            if train_series is None or test_series is None:
                continue

            if train_series.isna().all() or test_series.isna().all():
                continue
            if train_series.nunique(dropna=True) < 2:
                continue

            train_result[feature_name] = train_series
            test_result[feature_name] = test_series
            feature_operations[feature_name] = spec.operation

        return train_result, test_result, feature_operations

    @staticmethod
    def _safe_feature_name(name: str, used_names: set[str]) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
        if not cleaned:
            return ""
        if not cleaned.startswith("gen_"):
            cleaned = f"gen_{cleaned}"

        final_name = cleaned
        suffix = 2
        while final_name in used_names:
            final_name = f"{cleaned}_{suffix}"
            suffix += 1
        used_names.add(final_name)
        return final_name

    def _compute_feature(self, bundle: DataBundle, spec: FeatureSpec) -> tuple[pd.Series | None, pd.Series | None]:
        train_df = bundle.train
        test_df = bundle.test
        operation = spec.operation
        columns = spec.columns

        if operation == "row_mean":
            train = train_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            test = test_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            return train, test

        if operation == "row_std":
            train = train_df[columns].apply(pd.to_numeric, errors="coerce").std(axis=1).fillna(0.0)
            test = test_df[columns].apply(pd.to_numeric, errors="coerce").std(axis=1).fillna(0.0)
            return train, test

        if operation == "ratio" and len(columns) >= 2:
            train_a = pd.to_numeric(train_df[columns[0]], errors="coerce")
            train_b = pd.to_numeric(train_df[columns[1]], errors="coerce").replace(0, np.nan)
            test_a = pd.to_numeric(test_df[columns[0]], errors="coerce")
            test_b = pd.to_numeric(test_df[columns[1]], errors="coerce").replace(0, np.nan)
            return train_a / train_b, test_a / test_b

        if operation == "difference" and len(columns) >= 2:
            train = pd.to_numeric(train_df[columns[0]], errors="coerce") - pd.to_numeric(train_df[columns[1]], errors="coerce")
            test = pd.to_numeric(test_df[columns[0]], errors="coerce") - pd.to_numeric(test_df[columns[1]], errors="coerce")
            return train, test

        if operation == "log1p_abs" and len(columns) >= 1:
            train = pd.to_numeric(train_df[columns[0]], errors="coerce")
            test = pd.to_numeric(test_df[columns[0]], errors="coerce")
            return np.sign(train) * np.log1p(np.abs(train)), np.sign(test) * np.log1p(np.abs(test))

        if operation == "missing_ratio":
            train = train_df[columns].isna().mean(axis=1)
            test = test_df[columns].isna().mean(axis=1)
            return train, test

        if operation == "cat_freq" and len(columns) >= 1:
            column = columns[0]
            freq = train_df[column].fillna("__nan__").astype(str).value_counts(normalize=True, dropna=False)
            train = train_df[column].fillna("__nan__").astype(str).map(freq).fillna(0.0)
            test = test_df[column].fillna("__nan__").astype(str).map(freq).fillna(0.0)
            return train, test

        if operation == "text_len" and len(columns) >= 1:
            column = columns[0]
            return train_df[column].fillna("").astype(str).str.len(), test_df[column].fillna("").astype(str).str.len()

        if operation == "is_missing" and len(columns) >= 1:
            column = columns[0]
            return train_df[column].isna().astype("int8"), test_df[column].isna().astype("int8")

        return None, None

    @staticmethod
    def _sanitize_feature_pair(
        train_series: pd.Series,
        test_series: pd.Series,
    ) -> tuple[pd.Series | None, pd.Series | None]:
        train = train_series.reset_index(drop=True).copy()
        test = test_series.reset_index(drop=True).copy()

        if pd.api.types.is_numeric_dtype(train) and pd.api.types.is_numeric_dtype(test):
            train = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
            test = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan)

            finite_train = train[np.isfinite(train)]
            if not finite_train.empty and finite_train.nunique(dropna=True) > 2:
                q_low = float(finite_train.quantile(0.001))
                q_high = float(finite_train.quantile(0.999))
                if np.isfinite(q_low) and np.isfinite(q_high) and q_low < q_high:
                    train = train.clip(lower=q_low, upper=q_high)
                    test = test.clip(lower=q_low, upper=q_high)
            return train, test

        train = train.fillna("__nan__").astype(str)
        test = test.fillna("__nan__").astype(str)
        return train, test

    def _select_top_columns(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        top_k: int,
    ) -> tuple[list[str], dict[str, float]]:
        if features.empty:
            return [], {}

        y = target.reset_index(drop=True)
        if y.nunique(dropna=True) < 2:
            selected = list(features.columns[:top_k])
            return selected, {column: 0.0 for column in selected}

        scores: list[tuple[str, float]] = []
        for column in features.columns:
            signal = self._score_column_auc(features[column], y)
            scores.append((column, signal))
            logger.debug("Feature signal: %s -> %.6f", column, signal)

        ranked = sorted(scores, key=lambda item: item[1], reverse=True)
        picked = [column for column, score in ranked if np.isfinite(score) and score > 0][:top_k]
        score_map = {column: float(score) for column, score in ranked}
        if picked:
            return picked, score_map
        fallback = list(features.columns[:top_k])
        return fallback, score_map

    @staticmethod
    def _score_column_auc(column: pd.Series, target: pd.Series) -> float:
        if column.nunique(dropna=True) < 2:
            return 0.0

        if pd.api.types.is_numeric_dtype(column):
            x = pd.to_numeric(column, errors="coerce").fillna(-999.0)
        else:
            x = pd.Series(pd.factorize(column.fillna("__nan__").astype(str))[0], index=column.index)

        try:
            auc = roc_auc_score(target, x)
            return abs(float(auc) - 0.5)
        except Exception:
            return 0.0
