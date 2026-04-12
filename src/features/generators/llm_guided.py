from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.core.config import DEFAULT_CONFIG
from src.core.json_utils import safe_json_load
from src.core.llm import get_effective_llm_provider, get_llm_client
from src.data.loaders import DataBundle
from src.features.contracts import FeatureSet

logger = logging.getLogger(__name__)
_LLM_PLAN_CACHE: dict[str, list[list["FeatureSpec"]]] = {}


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
        self.disable_llm = os.getenv("FEATURES_AGENT_DISABLE_LLM", "0").strip() in {"1", "true", "yes"}
        self.llm_provider = "none" if self.disable_llm else get_effective_llm_provider()
        self.client = None if self.disable_llm else get_llm_client()
        budget_override = os.getenv("FEATURES_AGENT_LLM_CHAR_BUDGET", "").strip()
        try:
            self.char_budget = int(budget_override) if budget_override else DEFAULT_CONFIG.llm_char_budget_per_run
        except Exception:
            self.char_budget = DEFAULT_CONFIG.llm_char_budget_per_run
        self.char_budget = max(0, int(self.char_budget))
        self.used_chars = 0

    def generate(self, bundle: DataBundle, max_features: int) -> list[FeatureSet]:
        train_context, test_context = self._build_feature_context(bundle)
        common_columns = sorted(
            [
            column
            for column in train_context.columns
            if column in test_context.columns and column not in {bundle.id_column, bundle.target_column}
            ]
        )
        if not common_columns:
            return []

        numeric_columns = sorted(
            [
            column
            for column in common_columns
            if pd.api.types.is_numeric_dtype(train_context[column])
            ]
        )
        categorical_columns = sorted([column for column in common_columns if column not in numeric_columns])
        if not numeric_columns and not categorical_columns:
            return []

        plan, plan_source, plan_metadata = self._build_plan(
            bundle=bundle,
            train_df=train_context,
            test_df=test_context,
            target=bundle.train[bundle.target_column],
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )
        if not plan:
            return []

        train_features, test_features, feature_operations = self._apply_plan(
            train_df=train_context,
            test_df=test_context,
            specs=plan,
        )
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
                    "plan_source": plan_source,
                    "llm_provider": self.llm_provider,
                    "dataset_type": bundle.schema_context.dataset_profile.dataset_type,
                    "selected_operations": selected_operations,
                    "selected_feature_scores": {column: selection_scores.get(column, 0.0) for column in selected_columns},
                    "candidate_plan_size": len(plan),
                    "llm_char_budget_total": int(self.char_budget),
                    "llm_char_budget_used": int(self.used_chars),
                    **plan_metadata,
                },
            )
        ]

    def _build_plan(
        self,
        bundle: DataBundle,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target: pd.Series,
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_features: int,
    ) -> tuple[list[FeatureSpec], str, dict[str, object]]:
        llm_candidates, cache_hit = self._request_llm_candidate_plans(
            bundle=bundle,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )
        if llm_candidates:
            best_plan, proxy_score = self._select_best_plan_by_proxy(
                train_df=train_df,
                test_df=test_df,
                target=target,
                candidate_plans=llm_candidates,
                max_features=max_features,
            )
            if best_plan:
                return (
                    self._normalize_plan(best_plan, max_features=max_features),
                    "llm",
                    {
                        "llm_candidate_plans": len(llm_candidates),
                        "llm_cache_hit": cache_hit,
                        "llm_selected_plan_proxy_score": round(float(proxy_score), 6),
                    },
                )

        fallback = self._fallback_plan(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )
        return (
            self._normalize_plan(fallback, max_features=max_features),
            "fallback",
            {"llm_candidate_plans": 0, "llm_cache_hit": cache_hit},
        )

    def _request_llm_candidate_plans(
        self,
        bundle: DataBundle,
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_features: int,
    ) -> tuple[list[list[FeatureSpec]], bool]:
        if self.disable_llm:
            logger.info("LLM usage disabled by FEATURES_AGENT_DISABLE_LLM, using fallback plan.")
            return [], False
        if self.client is None:
            logger.info("LLM client is unavailable for provider '%s', using fallback plan.", self.llm_provider)
            return [], False

        schema_key = self._build_schema_cache_key(
            bundle=bundle,
            llm_provider=self.llm_provider,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )
        if schema_key in _LLM_PLAN_CACHE:
            cached = _LLM_PLAN_CACHE[schema_key]
            logger.info("LLM plan cache hit: schema_key=%s cached_candidates=%s", schema_key[:12], len(cached))
            return cached, True

        prompt = self._build_prompt(
            data_readme=bundle.data_readme,
            schema_summary=bundle.schema_context.to_prompt_summary(),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_features=max_features,
        )

        attempts = max(1, DEFAULT_CONFIG.llm_plan_max_attempts)
        max_candidates = max(1, DEFAULT_CONFIG.llm_self_consistency_plans)
        candidate_plans: list[list[FeatureSpec]] = []
        seen_signatures: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()

        for attempt in range(1, attempts + 1):
            if len(candidate_plans) >= max_candidates:
                break
            if not self._has_budget_for_attempt(prompt):
                logger.warning(
                    "LLM char budget exhausted before attempt %s/%s: used=%s budget=%s",
                    attempt,
                    attempts,
                    self.used_chars,
                    self.char_budget,
                )
                break
            try:
                response = self.client.invoke(prompt)
                response_text = self._response_to_text(response)
                self._consume_budget(len(prompt) + len(response_text))
                parsed = self._parse_plan(
                    response_text=response_text,
                    known_columns=set(numeric_columns + categorical_columns),
                    numeric_columns=set(numeric_columns),
                    categorical_columns=set(categorical_columns),
                )
                if parsed:
                    normalized = self._normalize_plan(parsed, max_features=max_features)
                    signature = self._plan_signature(normalized)
                    if signature not in seen_signatures:
                        candidate_plans.append(normalized)
                        seen_signatures.add(signature)
                        logger.info(
                            "LLM plan received: attempt=%s specs=%s unique_candidates=%s",
                            attempt,
                            len(normalized),
                            len(candidate_plans),
                        )
                    else:
                        logger.info("LLM plan duplicate ignored: attempt=%s", attempt)
                    continue
                logger.warning("LLM returned empty/invalid plan on attempt %s/%s", attempt, attempts)
            except Exception as error:
                logger.warning("LLM plan generation attempt %s/%s failed: %s", attempt, attempts, error)

            if attempt < attempts:
                time.sleep(min(0.25 * attempt, 0.8))

        if candidate_plans:
            _LLM_PLAN_CACHE[schema_key] = candidate_plans
            return candidate_plans, False

        logger.warning("LLM plan generation failed after %s attempts, fallback will be used.", attempts)
        return [], False

    @staticmethod
    def _plan_signature(specs: list[FeatureSpec]) -> tuple[tuple[str, tuple[str, ...]], ...]:
        ordered = sorted(
            ((spec.operation, tuple(spec.columns)) for spec in specs),
            key=lambda item: (item[0], item[1]),
        )
        return tuple(ordered)

    def _has_budget_for_attempt(self, prompt: str) -> bool:
        if self.char_budget <= 0:
            return True
        remaining = self.char_budget - self.used_chars
        min_needed = max(len(prompt) // 2, DEFAULT_CONFIG.llm_min_chars_per_attempt)
        return remaining >= min_needed

    def _consume_budget(self, consumed: int) -> None:
        if consumed <= 0:
            return
        self.used_chars += int(consumed)

    @staticmethod
    def _build_schema_cache_key(
        *,
        bundle: DataBundle,
        llm_provider: str,
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_features: int,
    ) -> str:
        fingerprint_payload = [
            llm_provider,
            bundle.schema_context.dataset_profile.dataset_type,
            bundle.id_column,
            bundle.target_column,
            str(max_features),
            "|".join(sorted(bundle.schema_context.recommended_joins.keys())),
            "|".join(sorted(f"{k}:{v}" for k, v in bundle.schema_context.recommended_joins.items())),
            "|".join(numeric_columns[:40]),
            "|".join(categorical_columns[:40]),
        ]
        raw = "||".join(fingerprint_payload)
        return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()

    def _select_best_plan_by_proxy(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target: pd.Series,
        candidate_plans: list[list[FeatureSpec]],
        max_features: int,
    ) -> tuple[list[FeatureSpec], float]:
        scored: list[tuple[list[FeatureSpec], float]] = []
        for index, specs in enumerate(candidate_plans, start=1):
            score = self._score_plan_proxy(
                train_df=train_df,
                test_df=test_df,
                target=target,
                specs=specs,
                max_features=max_features,
            )
            scored.append((specs, score))
            logger.info("LLM plan proxy score: candidate=%s score=%.6f", index, score)

        if not scored:
            return [], 0.0

        best_specs, best_score = max(scored, key=lambda item: item[1])
        return best_specs, float(best_score)

    def _score_plan_proxy(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target: pd.Series,
        specs: list[FeatureSpec],
        max_features: int,
    ) -> float:
        train_features, test_features, _ = self._apply_plan(
            train_df=train_df,
            test_df=test_df,
            specs=specs,
        )
        if train_features.empty or test_features.empty:
            return -1.0

        selected_columns, selection_scores = self._select_top_columns(
            features=train_features,
            target=target,
            top_k=max_features,
        )
        if not selected_columns:
            return -1.0

        score_values = [selection_scores.get(column, 0.0) for column in selected_columns]
        base_score = float(np.mean(score_values)) if score_values else 0.0
        diversity_penalty = self._diversity_penalty(train_features[selected_columns])
        return base_score - diversity_penalty

    @staticmethod
    def _diversity_penalty(features: pd.DataFrame) -> float:
        if features.shape[1] <= 1:
            return 0.0
        numeric = features.select_dtypes(include=["number"])
        if numeric.shape[1] <= 1:
            return 0.0
        corr = numeric.corr().abs()
        if corr.empty:
            return 0.0
        tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        values = tri.to_numpy().astype("float64").ravel()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0
        max_corr = float(np.max(finite))
        return max(0.0, max_corr - 0.85) * 0.05

    @staticmethod
    def _build_prompt(
        data_readme: str,
        schema_summary: str,
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_features: int,
    ) -> str:
        readme_fragment = data_readme[:3000] if data_readme else "No data readme provided."
        schema_fragment = schema_summary[:3200] if schema_summary else "No schema summary provided."
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

Schema summary:
{schema_fragment}

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
        parsed = safe_json_load(response_text)
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
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        specs: list[FeatureSpec],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
        train_result = pd.DataFrame(index=train_df.index)
        test_result = pd.DataFrame(index=test_df.index)
        feature_operations: dict[str, str] = {}

        used_names: set[str] = set()
        for spec in specs:
            feature_name = self._safe_feature_name(spec.name, used_names=used_names)
            if not feature_name:
                continue

            try:
                train_series, test_series = self._compute_feature(
                    train_df=train_df,
                    test_df=test_df,
                    spec=spec,
                )
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

    def _build_feature_context(self, bundle: DataBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_context = bundle.train.copy()
        test_context = bundle.test.copy()

        for table_name in sorted(bundle.aux_tables.keys()):
            aux_df = bundle.aux_tables[table_name]
            join_key = self._pick_join_key(bundle=bundle, table_name=table_name, aux_df=aux_df)
            if not join_key:
                continue
            prepared_aux = self._prepare_aux_table(
                aux_df=aux_df,
                join_key=join_key,
                table_name=table_name,
                forbidden_columns={bundle.target_column},
            )
            if prepared_aux.empty:
                continue

            train_context = train_context.merge(prepared_aux, on=join_key, how="left")
            test_context = test_context.merge(prepared_aux, on=join_key, how="left")

        return train_context, test_context

    @staticmethod
    def _pick_join_key(bundle: DataBundle, table_name: str, aux_df: pd.DataFrame) -> str | None:
        preferred = bundle.schema_context.recommended_joins.get(table_name)
        if preferred and preferred in aux_df.columns:
            return preferred

        base_keys = sorted(set(bundle.train.columns) & set(bundle.test.columns))
        candidates = [
            column
            for column in base_keys
            if column in aux_df.columns and column not in {bundle.target_column}
        ]
        if not candidates:
            return None
        if bundle.id_column in candidates:
            return bundle.id_column
        return candidates[0]

    @staticmethod
    def _prepare_aux_table(
        aux_df: pd.DataFrame,
        join_key: str,
        table_name: str,
        forbidden_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        safe_prefix = table_name.replace(" ", "_").lower()
        working = aux_df.copy()
        forbidden_normalized = {column.lower() for column in (forbidden_columns or set()) if column}

        renamed: dict[str, str] = {}
        for column in working.columns:
            if column == join_key:
                continue
            if column.lower() in forbidden_normalized:
                continue
            renamed[column] = f"{safe_prefix}__{column}"
        columns_to_keep = [join_key, *renamed.keys()]
        working = working[columns_to_keep].rename(columns=renamed)

        numeric_cols = sorted(
            [column for column in working.columns if column != join_key and pd.api.types.is_numeric_dtype(working[column])]
        )
        categorical_cols = sorted([column for column in working.columns if column != join_key and column not in numeric_cols])

        if not numeric_cols and not categorical_cols:
            return pd.DataFrame()

        agg_spec: dict[str, str] = {}
        for column in numeric_cols:
            agg_spec[column] = "mean"
        for column in categorical_cols:
            agg_spec[column] = "first"

        grouped = working.groupby(join_key, dropna=False).agg(agg_spec).reset_index()
        row_count = (
            working.groupby(join_key, dropna=False).size().rename(f"{safe_prefix}__row_count").reset_index()
        )
        grouped = grouped.merge(row_count, on=join_key, how="left")
        return grouped

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

    def _compute_feature(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        spec: FeatureSpec,
    ) -> tuple[pd.Series | None, pd.Series | None]:
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
