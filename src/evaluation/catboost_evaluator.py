from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features.contracts import FeatureSet


@dataclass
class FeatureSetScore:
    feature_set_name: str
    score: float
    n_features: int


class CatBoostFeatureEvaluator:
    def __init__(self, cv_folds: int = 5, random_seed: int = 42):
        self.cv_folds = cv_folds
        self.random_seed = random_seed

    def select_best(
        self, feature_sets: list[FeatureSet], target: pd.Series
    ) -> tuple[FeatureSet, list[FeatureSetScore]]:
        if not feature_sets:
            raise ValueError("No feature sets to evaluate.")

        scores: list[FeatureSetScore] = []
        for feature_set in feature_sets:
            auc = self._cross_validated_auc(feature_set.train_features, target)
            scores.append(
                FeatureSetScore(
                    feature_set_name=feature_set.name,
                    score=auc,
                    n_features=feature_set.train_features.shape[1],
                )
            )

        best_score = max(scores, key=lambda score: (score.score, -score.n_features))
        best_feature_set = next(
            feature_set for feature_set in feature_sets if feature_set.name == best_score.feature_set_name
        )
        return best_feature_set, scores

    def _cross_validated_auc(self, features: pd.DataFrame, target: pd.Series) -> float:
        if features.empty:
            return 0.0

        y = pd.Series(target).reset_index(drop=True)
        y_unique = y.nunique(dropna=True)
        if y_unique < 2:
            return 0.5

        X = features.reset_index(drop=True).copy()
        X, cat_feature_indices = self._prepare_features(X)

        class_counts = y.value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        if min_class_count < 2:
            return 0.5

        n_splits = max(2, min(self.cv_folds, min_class_count))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        fold_scores: list[float] = []
        for train_idx, valid_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

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

        return float(np.mean(fold_scores)) if fold_scores else 0.0

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

