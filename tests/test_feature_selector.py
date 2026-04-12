from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.feature_selector import CatBoostFeatureSelector
from src.features.contracts import FeatureCandidate


def _candidate(name: str, values: np.ndarray, family: str = "test") -> FeatureCandidate:
    series = pd.Series(values)
    return FeatureCandidate(
        name=name,
        train_feature=series,
        test_feature=series.iloc[: len(series) // 2].reset_index(drop=True).reindex(range(len(series) // 2), fill_value=0),
        source_family=family,
        compute_cost=0.1,
    )


def test_pruning_removes_constant_duplicate_and_correlated_features() -> None:
    target = pd.Series([0, 1] * 20)
    base = np.array([0, 1] * 20, dtype=float)
    selector = CatBoostFeatureSelector(cv_folds=3, random_seed=42)
    candidates = [
        FeatureCandidate("signal", pd.Series(base), pd.Series(base[:20]), "family", compute_cost=0.1),
        FeatureCandidate("signal_dup", pd.Series(base), pd.Series(base[:20]), "family", compute_cost=0.1),
        FeatureCandidate("constant", pd.Series(np.ones_like(base)), pd.Series(np.ones(20)), "family", compute_cost=0.1),
        FeatureCandidate("correlated", pd.Series(base * 2), pd.Series(base[:20] * 2), "family", compute_cost=0.1),
        FeatureCandidate("noise", pd.Series(np.arange(len(base))), pd.Series(np.arange(20)), "family", compute_cost=0.1),
    ]

    pruned = selector.prune_candidates(candidates, target=target, max_candidates=10)
    names = [candidate.name for candidate in pruned]

    assert "constant" not in names
    assert "gen_signal" in names
    assert "gen_signal_dup" not in names
    assert "gen_correlated" not in names


def test_forward_selection_uses_group_kfold_and_returns_max_five_features() -> None:
    rng = np.random.default_rng(42)
    size = 60
    target = pd.Series(([0, 1] * 30))
    groups = pd.Series([index // 3 for index in range(size)])
    signal = target.astype(float) + rng.normal(0, 0.05, size=size)
    noise = rng.normal(0, 1, size=size)
    candidates = []
    for idx in range(6):
        values = signal if idx < 3 else noise + idx
        candidates.append(
            FeatureCandidate(
                name=f"feature_{idx}",
                train_feature=pd.Series(values),
                test_feature=pd.Series(values[:20]),
                source_family="synthetic",
                compute_cost=0.1 + idx * 0.01,
            )
        )

    selector = CatBoostFeatureSelector(cv_folds=3, random_seed=42)
    result = selector.select_best(
        candidates=candidates,
        target=target,
        max_features=5,
        groups=groups,
        max_candidates=6,
    )

    assert result.cv_strategy == "stratified_group_kfold"
    assert 1 <= len(result.selected_candidates) <= 5
    assert list(result.feature_set.train_features.columns) == list(result.feature_set.test_features.columns)
