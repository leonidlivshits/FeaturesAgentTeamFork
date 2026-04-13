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


def test_pruning_adds_stability_metadata_for_numeric_and_categorical() -> None:
    target = pd.Series([0, 1] * 10)
    selector = CatBoostFeatureSelector(cv_folds=3, random_seed=42)
    candidates = [
        FeatureCandidate(
            name="numeric_shifted",
            train_feature=pd.Series(np.linspace(0.0, 1.0, num=20)),
            test_feature=pd.Series(np.linspace(2.0, 3.0, num=10)),
            source_family="synthetic",
            compute_cost=0.1,
        ),
        FeatureCandidate(
            name="categorical_unseen",
            train_feature=pd.Series(["a", "b"] * 10),
            test_feature=pd.Series(["a", "z", "a", "z", "a", "z", "a", "z", "a", "z"]),
            source_family="synthetic",
            compute_cost=0.1,
        ),
    ]

    pruned = selector.prune_candidates(candidates, target=target, max_candidates=10)
    metrics = {candidate.name: candidate.metadata for candidate in pruned}

    assert metrics["gen_numeric_shifted"]["distribution_shift"] > 0.0
    assert metrics["gen_numeric_shifted"]["unseen_ratio"] == 0.0
    assert metrics["gen_categorical_unseen"]["distribution_shift"] > 0.0
    assert metrics["gen_categorical_unseen"]["unseen_ratio"] > 0.0


def test_forward_selection_caps_llm_features() -> None:
    rng = np.random.default_rng(42)
    size = 80
    target = pd.Series(([0, 1] * (size // 2)))
    signal = target.astype(float)
    candidates = [
        FeatureCandidate(
            name="llm_feature_1",
            train_feature=pd.Series(signal + rng.normal(0, 0.05, size=size)),
            test_feature=pd.Series(signal[:20] + rng.normal(0, 0.05, size=20)),
            source_family="llm_planner",
            compute_cost=0.15,
        ),
        FeatureCandidate(
            name="llm_feature_2",
            train_feature=pd.Series(signal + rng.normal(0, 0.07, size=size)),
            test_feature=pd.Series(signal[:20] + rng.normal(0, 0.07, size=20)),
            source_family="llm_planner",
            compute_cost=0.15,
        ),
        FeatureCandidate(
            name="llm_feature_3",
            train_feature=pd.Series(signal + rng.normal(0, 0.09, size=size)),
            test_feature=pd.Series(signal[:20] + rng.normal(0, 0.09, size=20)),
            source_family="llm_planner",
            compute_cost=0.15,
        ),
        FeatureCandidate(
            name="heuristic_feature_1",
            train_feature=pd.Series(signal + rng.normal(0, 0.14, size=size)),
            test_feature=pd.Series(signal[:20] + rng.normal(0, 0.14, size=20)),
            source_family="heuristic",
            compute_cost=0.1,
        ),
        FeatureCandidate(
            name="heuristic_feature_2",
            train_feature=pd.Series(signal + rng.normal(0, 0.16, size=size)),
            test_feature=pd.Series(signal[:20] + rng.normal(0, 0.16, size=20)),
            source_family="heuristic",
            compute_cost=0.1,
        ),
    ]

    selector = CatBoostFeatureSelector(cv_folds=3, random_seed=42)
    result = selector.select_best(
        candidates=candidates,
        target=target,
        max_features=4,
        max_candidates=5,
    )
    llm_count = sum(1 for item in result.selected_candidates if item.source_family == "llm_planner")
    assert llm_count <= 2
