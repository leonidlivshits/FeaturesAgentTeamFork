from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

from src.core.config import DEFAULT_CONFIG
from src.core.runtime import RuntimeBudget
from src.features.contracts import FeatureCandidate, FeatureSet, normalize_feature_name

logger = logging.getLogger(__name__)


CATBOOST_SELECTION_PARAMS = {
    "iterations": 180,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": DEFAULT_CONFIG.random_seed,
    "verbose": 0,
    "thread_count": 1,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


@dataclass
class SelectionResult:
    feature_set: FeatureSet
    selected_candidates: list[FeatureCandidate]
    mean_auc: float
    std_auc: float
    cv_strategy: str
    selection_trace: list[dict[str, Any]] = field(default_factory=list)
    fallback_reason: str = ""
    effective_auc: float = 0.0


class CatBoostFeatureSelector:
    def __init__(self, cv_folds: int = 5, random_seed: int = 42):
        self.cv_folds = cv_folds
        self.random_seed = random_seed

    def prune_candidates(
        self,
        candidates: list[FeatureCandidate],
        target: pd.Series,
        max_candidates: int = 20,
    ) -> list[FeatureCandidate]:
        validated: list[tuple[int, FeatureCandidate]] = []
        seen_signatures: set[tuple[str, str]] = set()
        for input_order, candidate in enumerate(candidates):
            try:
                normalized = self._normalize_candidate(candidate, target=target)
                normalized.validate(expected_train_rows=len(target))
            except Exception as error:
                logger.debug("Skipping invalid candidate %s: %s", getattr(candidate, "name", "<unknown>"), error)
                continue

            signature = self._signature(normalized.train_feature)
            sig_key = (normalized.source_family, signature)
            if sig_key in seen_signatures:
                continue
            seen_signatures.add(sig_key)
            validated.append((input_order, normalized))

        ranked = sorted(
            validated,
            key=lambda item: (
                -float(item[1].proxy_score),
                float(item[1].metadata.get("distribution_shift", 0.0)),
                float(item[1].metadata.get("unseen_ratio", 0.0)),
                float(item[1].missing_ratio),
                float(item[1].compute_cost),
                item[0],
            ),
        )

        kept: list[FeatureCandidate] = []
        for _, candidate in ranked:
            if len(kept) >= max_candidates:
                break
            if any(self._is_highly_redundant(candidate, existing) for existing in kept):
                continue
            kept.append(candidate)
        return kept

    def select_best(
        self,
        candidates: list[FeatureCandidate],
        target: pd.Series,
        max_features: int,
        groups: pd.Series | None = None,
        runtime_budget: RuntimeBudget | None = None,
        min_seconds_per_eval: float = 0.0,
        max_candidates: int | None = None,
    ) -> SelectionResult:
        pruned = self.prune_candidates(
            candidates=candidates,
            target=target,
            max_candidates=max_candidates or DEFAULT_CONFIG.max_candidate_pool_size,
        )
        if not pruned:
            raise ValueError("No valid feature candidates available for selection.")

        splitter, cv_strategy = self._build_splitter(target=target, groups=groups)
        selected: list[FeatureCandidate] = []
        selected_mean = 0.5
        selected_effective = 0.5
        selected_std = math.inf
        trace: list[dict[str, Any]] = []
        remaining = pruned.copy()

        for round_idx in range(1, max_features + 1):
            if not remaining:
                break
            if runtime_budget is not None and selected and not runtime_budget.has_time(min_seconds_per_eval):
                logger.warning(
                    "Stopping forward selection due runtime budget: selected=%s remaining=%.2fs",
                    len(selected),
                    runtime_budget.remaining(),
                )
                break

            round_best: tuple[FeatureCandidate, float, float, dict[str, Any]] | None = None
            search_width = self._search_width(
                round_idx=round_idx,
                total_remaining=len(remaining),
                target_size=len(target),
            )
            llm_selected_count = sum(1 for item in selected if item.source_family == "llm_planner")
            max_llm_features = max(0, int(DEFAULT_CONFIG.selection_max_llm_features))
            round_candidates = sorted(
                remaining,
                key=lambda item: (
                    -float(item.proxy_score),
                    float(item.metadata.get("distribution_shift", 0.0)),
                    float(item.metadata.get("unseen_ratio", 0.0)),
                    float(item.compute_cost),
                    item.name,
                ),
            )[:search_width]
            for candidate in round_candidates:
                if (
                    candidate.source_family == "llm_planner"
                    and llm_selected_count >= max_llm_features
                ):
                    trace.append(
                        {
                            "round": round_idx,
                            "candidate": candidate.name,
                            "candidate_family": candidate.source_family,
                            "skipped": "llm_feature_cap",
                            "max_llm_features": max_llm_features,
                            "llm_selected_count": llm_selected_count,
                            "search_width": search_width,
                        }
                    )
                    continue
                if runtime_budget is not None and selected and not runtime_budget.has_time(min_seconds_per_eval):
                    break
                started_at = time.perf_counter()
                mean_auc, std_auc = self._evaluate_candidate_set(
                    selected + [candidate],
                    target=target,
                    splitter=splitter,
                    groups=groups,
                    cv_strategy=cv_strategy,
                )
                elapsed = time.perf_counter() - started_at
                family_repeat_count = sum(1 for item in selected if item.source_family == candidate.source_family)
                penalized_repeat_count = max(0, family_repeat_count - 1)
                family_penalty = float(DEFAULT_CONFIG.selection_family_repeat_penalty) * penalized_repeat_count
                candidate_set = selected + [candidate]
                mean_shift = float(np.mean([self._distribution_shift(item) for item in candidate_set]))
                mean_unseen_ratio = float(np.mean([self._unseen_ratio(item) for item in candidate_set]))
                cv_std_penalty = float(DEFAULT_CONFIG.selection_cv_std_penalty) * float(std_auc)
                shift_penalty = float(DEFAULT_CONFIG.selection_distribution_shift_penalty) * mean_shift
                unseen_penalty = float(DEFAULT_CONFIG.selection_unseen_ratio_penalty) * mean_unseen_ratio
                effective_auc = float(mean_auc - family_penalty - cv_std_penalty - shift_penalty - unseen_penalty)
                eval_trace = {
                    "round": round_idx,
                    "candidate": candidate.name,
                    "candidate_family": candidate.source_family,
                    "mean_auc": round(mean_auc, 6),
                    "effective_auc": round(effective_auc, 6),
                    "std_auc": round(std_auc, 6),
                    "cv_std_penalty": round(cv_std_penalty, 6),
                    "mean_distribution_shift": round(mean_shift, 6),
                    "mean_unseen_ratio": round(mean_unseen_ratio, 6),
                    "shift_penalty": round(shift_penalty, 6),
                    "unseen_penalty": round(unseen_penalty, 6),
                    "family_repeat_count": family_repeat_count,
                    "penalized_repeat_count": penalized_repeat_count,
                    "family_penalty": round(family_penalty, 6),
                    "elapsed_sec": round(elapsed, 4),
                    "n_features": len(selected) + 1,
                    "search_width": search_width,
                }
                trace.append(eval_trace)
                choice = (candidate, mean_auc, std_auc, eval_trace)
                if round_best is None or self._rank_tuple(choice) > self._rank_tuple(round_best):
                    round_best = choice

            if round_best is None:
                break

            candidate, mean_auc, std_auc, eval_trace = round_best
            effective_auc = float(eval_trace.get("effective_auc", mean_auc))
            improvement = effective_auc - selected_effective
            if selected and improvement < 1e-4:
                min_features = max(1, int(DEFAULT_CONFIG.selection_min_features))
                forced_drop = float(DEFAULT_CONFIG.selection_forced_max_drop)
                if len(selected) < min_features and mean_auc >= selected_mean - forced_drop:
                    eval_trace["selected_under_min_features"] = True
                else:
                    eval_trace["stopped"] = "no_material_improvement"
                    break

            eval_trace["selected"] = True
            selected.append(candidate)
            remaining = [item for item in remaining if item.name != candidate.name]
            selected_mean = mean_auc
            selected_effective = effective_auc
            selected_std = std_auc

        fallback_reason = ""
        if not selected:
            fallback_reason = "forward_selection_empty"
            selected = [pruned[0]]
            selected_mean, selected_std = self._evaluate_candidate_set(
                selected,
                target=target,
                splitter=splitter,
                groups=groups,
                cv_strategy=cv_strategy,
            )
            selected_effective = selected_mean

        feature_set = self._to_feature_set(selected)
        return SelectionResult(
            feature_set=feature_set,
            selected_candidates=selected,
            mean_auc=float(selected_mean),
            std_auc=float(selected_std if np.isfinite(selected_std) else 0.0),
            cv_strategy=cv_strategy,
            selection_trace=trace,
            fallback_reason=fallback_reason,
            effective_auc=float(selected_effective),
        )

    def _normalize_candidate(self, candidate: FeatureCandidate, target: pd.Series) -> FeatureCandidate:
        name = normalize_feature_name(candidate.name)
        train = candidate.train_feature.reset_index(drop=True).copy()
        test = candidate.test_feature.reset_index(drop=True).copy()
        if len(train) != len(target):
            raise ValueError("Target length mismatch.")

        if pd.api.types.is_numeric_dtype(train) and pd.api.types.is_numeric_dtype(test):
            train = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
            test = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            train = train.fillna("__nan__").astype(str)
            test = test.fillna("__nan__").astype(str)

        if train.isna().all() or test.isna().all():
            raise ValueError("All values are missing.")
        if train.nunique(dropna=True) < 2:
            raise ValueError("Feature is constant on train.")

        metadata = candidate.metadata.copy()
        metadata.update(self._stability_metrics(train=train, test=test))
        missing_ratio = candidate.missing_ratio or float(
            np.mean([train.isna().mean() if hasattr(train, "isna") else 0.0, test.isna().mean() if hasattr(test, "isna") else 0.0])
        )
        proxy_score = candidate.proxy_score or score_single_feature_proxy(train, target)
        return FeatureCandidate(
            name=name,
            train_feature=train,
            test_feature=test,
            source_family=candidate.source_family,
            proxy_score=float(proxy_score),
            missing_ratio=float(missing_ratio),
            compute_cost=float(candidate.compute_cost),
            metadata=metadata,
        )

    def _is_highly_redundant(self, left: FeatureCandidate, right: FeatureCandidate) -> bool:
        if self._signature(left.train_feature) == self._signature(right.train_feature):
            return True
        left_series = left.train_feature.reset_index(drop=True)
        right_series = right.train_feature.reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left_series) and pd.api.types.is_numeric_dtype(right_series):
            left_num = pd.to_numeric(left_series, errors="coerce").fillna(-999.0)
            right_num = pd.to_numeric(right_series, errors="coerce").fillna(-999.0)
            if left_num.std() == 0 or right_num.std() == 0:
                return True
            corr = left_num.corr(right_num)
            return bool(np.isfinite(corr) and abs(float(corr)) >= 0.995)
        return False

    @staticmethod
    def _signature(series: pd.Series) -> str:
        filled = series.fillna("__nan__") if not pd.api.types.is_numeric_dtype(series) else pd.to_numeric(series, errors="coerce").fillna(-999.0)
        hashed = pd.util.hash_pandas_object(filled, index=False)
        return str(int(hashed.sum()))

    def _build_splitter(
        self,
        target: pd.Series,
        groups: pd.Series | None,
    ) -> tuple[object, str]:
        y = pd.Series(target).reset_index(drop=True)
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        n_splits = max(2, min(self.cv_folds, min_class_count)) if min_class_count >= 2 else 2
        if len(y) >= 120_000:
            if groups is not None:
                group_series = pd.Series(groups).reset_index(drop=True)
                if group_series.nunique(dropna=True) >= 20:
                    return (
                        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_seed),
                        "group_holdout",
                    )
            return (
                StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_seed),
                "stratified_holdout",
            )
        elif len(y) >= 40_000:
            n_splits = min(n_splits, 3)

        if groups is not None:
            group_series = pd.Series(groups).reset_index(drop=True)
            if group_series.nunique(dropna=True) >= n_splits and group_series.nunique(dropna=True) < len(group_series):
                return (
                    StratifiedGroupKFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=self.random_seed,
                    ),
                    "stratified_group_kfold",
                )

        return (
            StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_seed,
            ),
            "stratified_kfold",
        )

    def _evaluate_candidate_set(
        self,
        candidates: list[FeatureCandidate],
        target: pd.Series,
        splitter: StratifiedKFold | StratifiedGroupKFold,
        groups: pd.Series | None,
        cv_strategy: str,
    ) -> tuple[float, float]:
        if not candidates:
            return 0.5, 0.0

        y = pd.Series(target).reset_index(drop=True)
        X = pd.DataFrame(
            {candidate.name: candidate.train_feature.reset_index(drop=True) for candidate in candidates}
        )
        X, cat_indices = self._prepare_features(X)
        model_params = self._catboost_params_for_rows(n_rows=len(y))
        repeat_count = self._holdout_repeats_for_rows(n_rows=len(y))

        fold_scores: list[float] = []
        if cv_strategy == "group_holdout" and groups is not None:
            group_series = groups.reset_index(drop=True)
            for repeat_idx in range(repeat_count):
                split_seed = self.random_seed + repeat_idx * int(DEFAULT_CONFIG.holdout_seed_stride)
                holdout_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed)
                for train_idx, valid_idx in holdout_splitter.split(X, y, group_series):
                    model = CatBoostClassifier(**model_params)
                    model.fit(
                        X.iloc[train_idx],
                        y.iloc[train_idx],
                        cat_features=cat_indices or None,
                    )
                    probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
                    fold_scores.append(float(roc_auc_score(y.iloc[valid_idx], probabilities)))
        elif cv_strategy == "stratified_holdout":
            for repeat_idx in range(repeat_count):
                split_seed = self.random_seed + repeat_idx * int(DEFAULT_CONFIG.holdout_seed_stride)
                holdout_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed)
                for train_idx, valid_idx in holdout_splitter.split(X, y):
                    model = CatBoostClassifier(**model_params)
                    model.fit(
                        X.iloc[train_idx],
                        y.iloc[train_idx],
                        cat_features=cat_indices or None,
                    )
                    probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
                    fold_scores.append(float(roc_auc_score(y.iloc[valid_idx], probabilities)))
        else:
            if cv_strategy == "stratified_group_kfold" and groups is not None:
                split_args = (X, y, groups.reset_index(drop=True))
            else:
                split_args = (X, y)
            for train_idx, valid_idx in splitter.split(*split_args):
                model = CatBoostClassifier(**model_params)
                model.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx],
                    cat_features=cat_indices or None,
                )
                probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
                fold_scores.append(float(roc_auc_score(y.iloc[valid_idx], probabilities)))

        if not fold_scores:
            return 0.5, 0.0
        return float(np.mean(fold_scores)), float(np.std(fold_scores))

    @staticmethod
    def _prepare_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
        prepared = features.copy()
        cat_indices: list[int] = []
        for idx, column in enumerate(prepared.columns):
            if pd.api.types.is_object_dtype(prepared[column]) or isinstance(prepared[column].dtype, pd.CategoricalDtype):
                prepared[column] = prepared[column].fillna("__nan__").astype(str)
                cat_indices.append(idx)
            else:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(-999.0)
        return prepared, cat_indices

    def _to_feature_set(self, selected: list[FeatureCandidate]) -> FeatureSet:
        train = pd.DataFrame({candidate.name: candidate.train_feature.reset_index(drop=True) for candidate in selected})
        test = pd.DataFrame({candidate.name: candidate.test_feature.reset_index(drop=True) for candidate in selected})
        metadata = {
            "feature_families": [candidate.source_family for candidate in selected],
            "selected_candidates": [candidate.name for candidate in selected],
        }
        return FeatureSet(
            name="selected_feature_bundle",
            train_features=train,
            test_features=test,
            description="Feature bundle selected by forward selection.",
            metadata=metadata,
        )

    @staticmethod
    def _search_width(round_idx: int, total_remaining: int, target_size: int) -> int:
        fast_mode_limit = max(100_000, int(DEFAULT_CONFIG.selection_fast_mode_row_limit))
        if target_size >= fast_mode_limit:
            if round_idx <= 1:
                return min(total_remaining, 2)
            return min(total_remaining, 1)
        if target_size >= 120_000:
            if round_idx <= 1:
                return min(total_remaining, 2)
            if round_idx == 2:
                return min(total_remaining, 2)
            return min(total_remaining, 1)
        if total_remaining >= 12:
            if round_idx <= 1:
                return min(total_remaining, 3)
            if round_idx == 2:
                return min(total_remaining, 2)
            return min(total_remaining, 2)
        if round_idx <= 1:
            return min(total_remaining, 5)
        if round_idx == 2:
            return min(total_remaining, 3)
        return min(total_remaining, 2)

    @staticmethod
    def _rank_tuple(item: tuple[FeatureCandidate, float, float, dict[str, Any]]) -> tuple[float, float, float, float, float, str]:
        candidate, mean_auc, std_auc, eval_trace = item
        effective_auc = float(eval_trace.get("effective_auc", mean_auc))
        join_complexity = float(candidate.metadata.get("join_complexity", 0.0))
        return (
            float(effective_auc),
            float(mean_auc),
            -float(std_auc),
            -float(candidate.missing_ratio),
            -float(join_complexity),
            -float(candidate.compute_cost),
            candidate.name,
        )

    @staticmethod
    def _distribution_shift(candidate: FeatureCandidate) -> float:
        return float(candidate.metadata.get("distribution_shift", 0.0))

    @staticmethod
    def _unseen_ratio(candidate: FeatureCandidate) -> float:
        return float(candidate.metadata.get("unseen_ratio", 0.0))

    @staticmethod
    def _stability_metrics(train: pd.Series, test: pd.Series) -> dict[str, float]:
        if pd.api.types.is_numeric_dtype(train) and pd.api.types.is_numeric_dtype(test):
            train_num = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
            test_num = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan)
            train_values = train_num.dropna().to_numpy(dtype="float64")
            test_values = test_num.dropna().to_numpy(dtype="float64")
            if len(train_values) < 8 or len(test_values) < 8:
                return {"distribution_shift": 0.0, "unseen_ratio": 0.0}

            quantiles = np.array([0.1, 0.5, 0.9], dtype="float64")
            train_q = np.nanquantile(train_values, quantiles)
            test_q = np.nanquantile(test_values, quantiles)
            scale = max(
                float(abs(train_q[2] - train_q[0])),
                float(np.nanstd(train_values)),
                1e-6,
            )
            quantile_shift = float(np.mean(np.abs(train_q - test_q)) / scale)
            mean_shift = float(
                abs(float(np.nanmean(train_values)) - float(np.nanmean(test_values)))
                / max(float(np.nanstd(train_values)), 1e-6)
            )
            distribution_shift = float(min(3.0, 0.65 * quantile_shift + 0.35 * mean_shift))
            return {"distribution_shift": distribution_shift, "unseen_ratio": 0.0}

        train_cat = train.fillna("__nan__").astype(str)
        test_cat = test.fillna("__nan__").astype(str)
        train_freq = train_cat.value_counts(normalize=True, dropna=False)
        test_freq = test_cat.value_counts(normalize=True, dropna=False)
        unseen_ratio = float((~test_cat.isin(train_freq.index)).mean()) if len(test_cat) else 0.0

        merged_index = train_freq.index.union(test_freq.index)
        train_aligned = train_freq.reindex(merged_index, fill_value=0.0)
        test_aligned = test_freq.reindex(merged_index, fill_value=0.0)
        total_variation = float(0.5 * np.abs(train_aligned - test_aligned).sum())
        distribution_shift = float(min(2.0, total_variation + unseen_ratio))
        return {"distribution_shift": distribution_shift, "unseen_ratio": unseen_ratio}

    @staticmethod
    def _catboost_params_for_rows(n_rows: int) -> dict[str, Any]:
        params = CATBOOST_SELECTION_PARAMS.copy()
        if n_rows >= 500_000:
            params["iterations"] = 70
            params["depth"] = 4
            params["learning_rate"] = 0.07
        elif n_rows >= 250_000:
            params["iterations"] = 90
            params["depth"] = 5
            params["learning_rate"] = 0.06
        elif n_rows >= 120_000:
            params["iterations"] = 120
            params["depth"] = 5
            params["learning_rate"] = 0.055
        return params

    @staticmethod
    def _holdout_repeats_for_rows(n_rows: int) -> int:
        repeats = max(1, int(DEFAULT_CONFIG.holdout_repeats))
        if n_rows >= 250_000:
            return 1
        return repeats


def score_single_feature_proxy(feature: pd.Series, target: pd.Series) -> float:
    series = feature.reset_index(drop=True)
    y = pd.Series(target).reset_index(drop=True)
    if len(series) != len(y) or y.nunique(dropna=True) < 2 or series.nunique(dropna=True) < 2:
        return 0.0

    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    else:
        x = pd.Series(pd.factorize(series.fillna("__nan__").astype(str))[0], index=series.index)

    try:
        auc = roc_auc_score(y, x)
    except Exception:
        return 0.0
    return abs(float(auc) - 0.5)
