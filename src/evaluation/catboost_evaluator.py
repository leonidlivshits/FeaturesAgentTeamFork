from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.core.config import DEFAULT_CONFIG
from src.core.runtime import RuntimeBudget
from src.features.contracts import FeatureSet

logger = logging.getLogger(__name__)


@dataclass
class FeatureSetScore:
    feature_set_name: str
    score: float
    n_features: int
    elapsed_sec: float
    cv_std: float = 0.0
    effective_score: float = 0.0
    n_splits: int = 0


class CatBoostFeatureEvaluator:
    def __init__(self, cv_folds: int = 5, random_seed: int = 42, cv_std_penalty: float | None = None):
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.cv_std_penalty = (
            float(cv_std_penalty) if cv_std_penalty is not None else float(DEFAULT_CONFIG.cv_std_penalty)
        )

    def select_best(
        self,
        feature_sets: list[FeatureSet],
        target: pd.Series,
        groups: pd.Series | None = None,
        runtime_budget: RuntimeBudget | None = None,
        min_seconds_per_candidate_eval: float = 0.0,
    ) -> tuple[FeatureSet, list[FeatureSetScore]]:
        if not feature_sets:
            raise ValueError("No feature sets to evaluate.")

        scores: list[FeatureSetScore] = []
        total_candidates = len(feature_sets)
        for index, feature_set in enumerate(feature_sets, start=1):
            if runtime_budget is not None and scores and not runtime_budget.has_time(min_seconds_per_candidate_eval):
                logger.warning(
                    "Stopping candidate evaluation due runtime budget: evaluated=%s/%s, remaining=%.2fs",
                    len(scores),
                    total_candidates,
                    runtime_budget.remaining(),
                )
                break

            started_at = time.perf_counter()
            try:
                auc, auc_std, n_splits = self.evaluate_features_with_stats(
                    feature_set.train_features,
                    target=target,
                    groups=groups,
                )
            except Exception as error:
                logger.warning(
                    "Candidate evaluation failed, fallback score will be used: name=%s error=%s",
                    feature_set.name,
                    error,
                )
                auc = 0.5
                auc_std = 0.0
                n_splits = 0
            effective_score = float(auc - self.cv_std_penalty * auc_std)
            elapsed_sec = time.perf_counter() - started_at
            scores.append(
                FeatureSetScore(
                    feature_set_name=feature_set.name,
                    score=auc,
                    n_features=feature_set.train_features.shape[1],
                    elapsed_sec=elapsed_sec,
                    cv_std=float(auc_std),
                    effective_score=effective_score,
                    n_splits=int(n_splits),
                )
            )
            logger.info(
                "Candidate evaluated %s/%s: name=%s auc=%.6f cv_std=%.6f effective=%.6f folds=%s features=%s elapsed=%.2fs",
                index,
                total_candidates,
                feature_set.name,
                auc,
                auc_std,
                effective_score,
                n_splits,
                feature_set.train_features.shape[1],
                elapsed_sec,
            )

        if not scores:
            fallback_set = feature_sets[0]
            fallback_score = FeatureSetScore(
                feature_set_name=fallback_set.name,
                score=0.5,
                n_features=fallback_set.train_features.shape[1],
                elapsed_sec=0.0,
                cv_std=0.0,
                effective_score=0.5,
                n_splits=0,
            )
            logger.warning(
                "No feature sets were evaluated within runtime budget, selecting first candidate as fallback: %s",
                fallback_set.name,
            )
            return fallback_set, [fallback_score]

        best_score = max(scores, key=lambda score: (score.effective_score, score.score, -score.n_features))
        best_feature_set = next(
            feature_set for feature_set in feature_sets if feature_set.name == best_score.feature_set_name
        )
        return best_feature_set, scores

    def evaluate_features(
        self,
        features: pd.DataFrame,
        *,
        target: pd.Series,
        groups: pd.Series | None = None,
    ) -> float:
        auc, _, _ = self._cross_validated_auc_with_stats(features=features, target=target, groups=groups)
        return auc

    def evaluate_features_with_stats(
        self,
        features: pd.DataFrame,
        *,
        target: pd.Series,
        groups: pd.Series | None = None,
    ) -> tuple[float, float, int]:
        return self._cross_validated_auc_with_stats(features=features, target=target, groups=groups)

    def _cross_validated_auc_with_stats(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        groups: pd.Series | None = None,
    ) -> tuple[float, float, int]:
        if features.empty:
            return 0.0, 0.0, 0

        y = pd.Series(target).reset_index(drop=True)
        y_unique = y.nunique(dropna=True)
        if y_unique < 2:
            return 0.5, 0.0, 0

        X = features.reset_index(drop=True).copy()
        X, cat_feature_indices = self._prepare_features(X)
        groups_series = pd.Series(groups).reset_index(drop=True) if groups is not None else None

        splits = self._build_splits(X=X, y=y, groups=groups_series)
        if not splits:
            return 0.5, 0.0, 0

        fold_scores: list[float] = []
        for train_idx, valid_idx in splits:
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]
            if y_train.nunique(dropna=True) < 2 or y_valid.nunique(dropna=True) < 2:
                continue

            model = CatBoostClassifier(
                random_seed=self.random_seed,
                verbose=0,
                auto_class_weights="Balanced",
            )
            model.fit(
                X_train,
                y_train,
                cat_features=cat_feature_indices or None,
            )

            probabilities = model.predict_proba(X_valid)[:, 1]
            fold_scores.append(float(roc_auc_score(y_valid, probabilities)))

        if not fold_scores:
            return 0.5, 0.0, 0
        return float(np.mean(fold_scores)), float(np.std(fold_scores)), int(len(fold_scores))

    def _build_splits(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is not None:
            unique_groups = int(pd.Series(groups).nunique(dropna=True))
            if unique_groups >= 3:
                n_splits = max(2, min(self.cv_folds, unique_groups))
                gkf = GroupKFold(n_splits=n_splits)
                return list(gkf.split(X, y, groups))

        class_counts = y.value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        if min_class_count < 2:
            return []
        n_splits = max(2, min(self.cv_folds, min_class_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        return list(skf.split(X, y))

    @staticmethod
    def _prepare_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
        prepared = features.copy()
        cat_feature_indices: list[int] = []

        for index, column in enumerate(prepared.columns):
            if pd.api.types.is_object_dtype(prepared[column]) or pd.api.types.is_categorical_dtype(
                prepared[column]
            ):
                prepared[column] = prepared[column].fillna("__nan__").astype(str)
                cat_feature_indices.append(index)
            else:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(-999.0)

        return prepared, cat_feature_indices
