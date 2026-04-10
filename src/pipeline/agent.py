from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.core.config import DATA_DIR, DEFAULT_CONFIG, OUTPUT_DIR
from src.data.loaders import DataBundle, load_data_bundle
from src.evaluation.catboost_evaluator import CatBoostFeatureEvaluator, FeatureSetScore
from src.features.contracts import FeatureSet
from src.features.registry import build_generators
from src.io.writer import write_submission

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    best_feature_set: str
    best_cv_auc: float
    generated_feature_count: int
    id_column: str
    target_column: str
    all_scores: list[FeatureSetScore]


def run_pipeline() -> PipelineResult:
    bundle = load_data_bundle(DATA_DIR)
    feature_sets = generate_feature_sets(bundle=bundle, max_features=DEFAULT_CONFIG.max_features)
    selected_feature_set, scores = select_best_feature_set(feature_sets=feature_sets, bundle=bundle)

    write_submission(
        train_base=bundle.train,
        test_base=bundle.test,
        selected_feature_set=selected_feature_set,
        output_dir=OUTPUT_DIR,
    )

    score_lookup = {score.feature_set_name: score for score in scores}
    best_score = score_lookup[selected_feature_set.name]
    logger.info(
        "Pipeline finished",
        extra={
            "best_feature_set": selected_feature_set.name,
            "best_cv_auc": round(best_score.score, 6),
            "n_features": selected_feature_set.train_features.shape[1],
        },
    )

    return PipelineResult(
        best_feature_set=selected_feature_set.name,
        best_cv_auc=best_score.score,
        generated_feature_count=selected_feature_set.train_features.shape[1],
        id_column=bundle.id_column,
        target_column=bundle.target_column,
        all_scores=scores,
    )


def generate_feature_sets(bundle: DataBundle, max_features: int) -> list[FeatureSet]:
    feature_sets: list[FeatureSet] = []
    for generator in build_generators():
        generated_sets = generator.generate(bundle=bundle, max_features=max_features)
        for feature_set in generated_sets:
            normalized_set = normalize_feature_set(feature_set, max_features=max_features)
            normalized_set.validate()
            if normalized_set.train_features.empty:
                continue
            feature_sets.append(normalized_set)

    if not feature_sets:
        fallback = build_fallback_feature_set(bundle=bundle)
        feature_sets.append(fallback)

    return feature_sets


def select_best_feature_set(feature_sets: list[FeatureSet], bundle: DataBundle) -> tuple[FeatureSet, list[FeatureSetScore]]:
    evaluator = CatBoostFeatureEvaluator(
        cv_folds=DEFAULT_CONFIG.cv_folds,
        random_seed=DEFAULT_CONFIG.random_seed,
    )
    target = bundle.train[bundle.target_column]
    return evaluator.select_best(feature_sets=feature_sets, target=target)


def normalize_feature_set(feature_set: FeatureSet, max_features: int) -> FeatureSet:
    columns = list(feature_set.train_features.columns)[:max_features]
    train_features = feature_set.train_features[columns].copy()
    test_features = feature_set.test_features[columns].copy()
    return FeatureSet(
        name=feature_set.name,
        train_features=train_features,
        test_features=test_features,
        description=feature_set.description,
        metadata=feature_set.metadata,
    )


def build_fallback_feature_set(bundle: DataBundle) -> FeatureSet:
    train_hash = pd.util.hash_pandas_object(bundle.train[bundle.id_column], index=False).astype("uint64")
    test_hash = pd.util.hash_pandas_object(bundle.test[bundle.id_column], index=False).astype("uint64")

    train_features = pd.DataFrame({"gen_id_hash": (train_hash % 100000).astype("float64") / 100000.0})
    test_features = pd.DataFrame({"gen_id_hash": (test_hash % 100000).astype("float64") / 100000.0})

    return FeatureSet(
        name="fallback_id_hash",
        train_features=train_features,
        test_features=test_features,
        description="Fallback deterministic feature from id hash.",
    )
