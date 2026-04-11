from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.core.config import DATA_DIR, DEFAULT_CONFIG, OUTPUT_DIR
from src.core.runtime import RuntimeBudget, ensure_input_contract, ensure_output_contract, prepare_output_dir
from src.data.loaders import DataBundle, load_data_bundle
from src.evaluation.catboost_evaluator import CatBoostFeatureEvaluator, FeatureSetScore
from src.features.contracts import FeatureSet
from src.features.registry import build_generators, get_effective_generator_mode, get_generator_mode
from src.io.contracts import validate_output_contract
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
    decision_trace: dict[str, Any]


def run_pipeline() -> PipelineResult:
    ensure_input_contract(DATA_DIR)
    prepare_output_dir(OUTPUT_DIR)
    runtime_budget = RuntimeBudget(
        total_sec=DEFAULT_CONFIG.internal_time_budget_sec,
        reserve_sec=DEFAULT_CONFIG.runtime_reserve_sec,
    )

    bundle = load_data_bundle(DATA_DIR)
    feature_sets = generate_feature_sets(
        bundle=bundle,
        max_features=DEFAULT_CONFIG.max_features,
        runtime_budget=runtime_budget,
    )
    selected_feature_set, scores = select_best_feature_set(
        feature_sets=feature_sets,
        bundle=bundle,
        runtime_budget=runtime_budget,
    )

    output_train_path, output_test_path = write_submission(
        train_base=bundle.train,
        test_base=bundle.test,
        selected_feature_set=selected_feature_set,
        output_dir=OUTPUT_DIR,
    )
    validate_output_contract(
        input_train=bundle.train,
        input_test=bundle.test,
        output_train_path=output_train_path,
        output_test_path=output_test_path,
        max_features=DEFAULT_CONFIG.max_features,
    )
    ensure_output_contract(OUTPUT_DIR)

    score_lookup = {score.feature_set_name: score for score in scores}
    best_score = score_lookup.get(selected_feature_set.name, max(scores, key=lambda score: score.score))
    selected_metadata = selected_feature_set.metadata if isinstance(selected_feature_set.metadata, dict) else {}
    decision_trace: dict[str, Any] = {
        "requested_mode": get_generator_mode(),
        "effective_mode": get_effective_generator_mode(),
        "candidate_sets_generated": len(feature_sets),
        "candidate_sets_evaluated": len(scores),
        "selected_feature_set": selected_feature_set.name,
        "selected_cv_auc": round(float(best_score.score), 6),
        "selected_n_features": int(selected_feature_set.train_features.shape[1]),
        "plan_source": selected_metadata.get("plan_source", "n/a"),
    }
    ranked_scores = sorted(scores, key=lambda score: score.score, reverse=True)
    for rank, score in enumerate(ranked_scores, start=1):
        logger.info(
            "Candidate rank #%s: name=%s auc=%.6f features=%s elapsed=%.2fs",
            rank,
            score.feature_set_name,
            score.score,
            score.n_features,
            score.elapsed_sec,
        )

    logger.info(
        "Pipeline finished: best=%s auc=%.6f features=%s elapsed=%.2fs remaining=%.2fs",
        selected_feature_set.name,
        best_score.score,
        selected_feature_set.train_features.shape[1],
        runtime_budget.elapsed(),
        runtime_budget.remaining(),
    )
    logger.info("Decision trace: %s", decision_trace)

    return PipelineResult(
        best_feature_set=selected_feature_set.name,
        best_cv_auc=best_score.score,
        generated_feature_count=selected_feature_set.train_features.shape[1],
        id_column=bundle.id_column,
        target_column=bundle.target_column,
        all_scores=scores,
        decision_trace=decision_trace,
    )


def generate_feature_sets(
    bundle: DataBundle,
    max_features: int,
    runtime_budget: RuntimeBudget,
) -> list[FeatureSet]:
    feature_sets: list[FeatureSet] = []
    for generator in build_generators():
        if not runtime_budget.has_time(DEFAULT_CONFIG.min_seconds_per_generator):
            logger.warning(
                "Stopping generators due runtime budget: generated=%s remaining=%.2fs",
                len(feature_sets),
                runtime_budget.remaining(),
            )
            break

        started_at = time.perf_counter()
        try:
            generated_sets = generator.generate(bundle=bundle, max_features=max_features)
        except Exception as error:
            logger.exception("Generator failed: %s", error)
            continue
        elapsed_sec = time.perf_counter() - started_at
        logger.info(
            "Generator completed: name=%s produced_sets=%s elapsed=%.2fs remaining=%.2fs",
            getattr(generator, "name", generator.__class__.__name__),
            len(generated_sets),
            elapsed_sec,
            runtime_budget.remaining(),
        )

        for feature_set in generated_sets:
            try:
                normalized_set = normalize_feature_set(feature_set, max_features=max_features)
                normalized_set.validate(
                    expected_train_rows=len(bundle.train),
                    expected_test_rows=len(bundle.test),
                )
            except Exception as error:
                logger.warning(
                    "Skipping invalid feature set: name=%s error=%s",
                    getattr(feature_set, "name", "<unknown>"),
                    error,
                )
                continue
            if normalized_set.train_features.empty:
                continue
            feature_sets.append(normalized_set)
            if len(feature_sets) >= DEFAULT_CONFIG.max_candidate_feature_sets:
                logger.warning(
                    "Reached candidate cap=%s, stopping generation.",
                    DEFAULT_CONFIG.max_candidate_feature_sets,
                )
                break
        if len(feature_sets) >= DEFAULT_CONFIG.max_candidate_feature_sets:
            break

    if not feature_sets:
        fallback = build_fallback_feature_set(bundle=bundle)
        feature_sets.append(fallback)

    return feature_sets


def select_best_feature_set(
    feature_sets: list[FeatureSet],
    bundle: DataBundle,
    runtime_budget: RuntimeBudget,
) -> tuple[FeatureSet, list[FeatureSetScore]]:
    evaluator = CatBoostFeatureEvaluator(
        cv_folds=DEFAULT_CONFIG.cv_folds,
        random_seed=DEFAULT_CONFIG.random_seed,
    )
    target = bundle.train[bundle.target_column]
    try:
        return evaluator.select_best(
            feature_sets=feature_sets,
            target=target,
            runtime_budget=runtime_budget,
            min_seconds_per_candidate_eval=DEFAULT_CONFIG.min_seconds_per_candidate_eval,
        )
    except Exception as error:
        logger.warning("Evaluator failed, using proxy fallback selector: %s", error)
        return select_best_feature_set_fallback(feature_sets=feature_sets, target=target)


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


def select_best_feature_set_fallback(
    feature_sets: list[FeatureSet],
    target: pd.Series,
) -> tuple[FeatureSet, list[FeatureSetScore]]:
    if not feature_sets:
        raise ValueError("No feature sets available for fallback selection.")

    scores: list[FeatureSetScore] = []
    started_at = time.perf_counter()
    for feature_set in feature_sets:
        proxy_score = score_feature_set_proxy(feature_set.train_features, target)
        scores.append(
            FeatureSetScore(
                feature_set_name=feature_set.name,
                score=proxy_score,
                n_features=feature_set.train_features.shape[1],
                elapsed_sec=0.0,
            )
        )

    best_score = max(scores, key=lambda score: (score.score, -score.n_features))
    elapsed_total = time.perf_counter() - started_at
    for score in scores:
        score.elapsed_sec = elapsed_total / max(len(scores), 1)

    best_feature_set = next(
        feature_set for feature_set in feature_sets if feature_set.name == best_score.feature_set_name
    )
    return best_feature_set, scores


def score_feature_set_proxy(features: pd.DataFrame, target: pd.Series) -> float:
    if features.empty:
        return 0.0

    y = pd.Series(target).reset_index(drop=True)
    if y.nunique(dropna=True) < 2:
        return 0.5

    signals: list[float] = []
    for column in features.columns:
        series = features[column].reset_index(drop=True)
        if series.nunique(dropna=True) < 2:
            continue

        if pd.api.types.is_numeric_dtype(series):
            x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-999.0)
        else:
            x = pd.Series(pd.factorize(series.fillna("__nan__").astype(str))[0], index=series.index)

        try:
            auc = roc_auc_score(y, x)
            signals.append(abs(float(auc) - 0.5))
        except Exception:
            continue

    if not signals:
        return 0.5
    mean_signal = float(np.mean(signals))
    return max(0.0, min(1.0, 0.5 + mean_signal))
