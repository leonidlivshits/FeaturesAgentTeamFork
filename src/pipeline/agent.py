from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.config import DATA_DIR, DEFAULT_CONFIG, OUTPUT_DIR
from src.core.runtime import RuntimeBudget, ensure_input_contract, ensure_output_contract, prepare_output_dir
from src.data.loaders import load_data_bundle
from src.evaluation.feature_selector import CatBoostFeatureSelector, SelectionResult
from src.features.contracts import FeatureCandidate, FeatureSet
from src.features.factory import SchemaAwareFeatureFactory
from src.io.contracts import validate_output_contract
from src.io.writer import write_submission
from src.schema.profiler import build_dataset_profile

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    best_feature_set: str
    best_cv_auc: float
    generated_feature_count: int
    id_column: str
    target_column: str
    all_scores: list[dict[str, Any]]
    decision_trace: dict[str, Any]


def run_pipeline() -> PipelineResult:
    ensure_input_contract(DATA_DIR)
    prepare_output_dir(OUTPUT_DIR)
    runtime_budget = RuntimeBudget(
        total_sec=DEFAULT_CONFIG.internal_time_budget_sec,
        reserve_sec=DEFAULT_CONFIG.runtime_reserve_sec,
    )

    bundle = load_data_bundle(DATA_DIR)
    profile = build_dataset_profile(bundle)
    feature_factory = SchemaAwareFeatureFactory()
    candidates = feature_factory.generate_candidates(
        bundle=bundle,
        profile=profile,
        runtime_budget=runtime_budget,
    )
    if not candidates:
        candidates = [build_fallback_candidate(bundle)]

    groups = None
    if profile.group_column and profile.group_column in bundle.train.columns:
        group_series = bundle.train[profile.group_column]
        if group_series.nunique(dropna=True) < len(group_series):
            groups = group_series

    selector = CatBoostFeatureSelector(
        cv_folds=DEFAULT_CONFIG.cv_folds,
        random_seed=DEFAULT_CONFIG.random_seed,
    )
    try:
        selection_result = selector.select_best(
            candidates=candidates,
            target=bundle.train[bundle.target_column],
            max_features=DEFAULT_CONFIG.max_features,
            groups=groups,
            runtime_budget=runtime_budget,
            min_seconds_per_eval=DEFAULT_CONFIG.min_seconds_per_candidate_eval,
            max_candidates=DEFAULT_CONFIG.max_candidate_pool_size,
        )
    except Exception as error:
        logger.warning("Selection failed, using fallback candidate: %s", error)
        selection_result = SelectionResult(
            feature_set=build_fallback_candidate(bundle).to_feature_set(),
            selected_candidates=[build_fallback_candidate(bundle)],
            mean_auc=0.5,
            std_auc=0.0,
            cv_strategy="fallback",
            fallback_reason="selection_failure",
        )

    output_train_path, output_test_path = write_submission(
        train_base=bundle.train,
        test_base=bundle.test,
        selected_feature_set=selection_result.feature_set,
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

    selected_families = [candidate.source_family for candidate in selection_result.selected_candidates]
    llm_in_final = any(candidate.source_family == "llm_planner" for candidate in selection_result.selected_candidates)
    llm_candidates = [candidate for candidate in candidates if candidate.source_family == "llm_planner"]
    decision_trace: dict[str, Any] = {
        "dataset_type": profile.dataset_type,
        "group_column": profile.group_column or "n/a",
        "time_column": profile.time_column or "n/a",
        "candidate_pool_size": len(candidates),
        "selected_feature_set": selection_result.feature_set.name,
        "selected_cv_auc": round(float(selection_result.mean_auc), 6),
        "selected_cv_std": round(float(selection_result.std_auc), 6),
        "selected_n_features": int(selection_result.feature_set.train_features.shape[1]),
        "selected_feature_families": selected_families,
        "cv_strategy": selection_result.cv_strategy,
        "fallback_reason": selection_result.fallback_reason or "n/a",
        "llm_candidate_count": len(llm_candidates),
        "llm_in_final": llm_in_final,
        "tables_profiled": sorted(profile.table_profiles),
        "join_plan_count": len(profile.join_plans),
    }

    logger.info(
        "Pipeline finished: dataset_type=%s best_auc=%.6f std=%.6f features=%s remaining=%.2fs",
        profile.dataset_type,
        selection_result.mean_auc,
        selection_result.std_auc,
        selection_result.feature_set.train_features.shape[1],
        runtime_budget.remaining(),
    )
    logger.info("Decision trace: %s", decision_trace)

    return PipelineResult(
        best_feature_set=selection_result.feature_set.name,
        best_cv_auc=selection_result.mean_auc,
        generated_feature_count=selection_result.feature_set.train_features.shape[1],
        id_column=bundle.id_column,
        target_column=bundle.target_column,
        all_scores=selection_result.selection_trace,
        decision_trace=decision_trace,
    )


def build_fallback_candidate(bundle) -> FeatureCandidate:
    train_hash = pd.util.hash_pandas_object(bundle.train[bundle.id_column], index=False).astype("uint64")
    test_hash = pd.util.hash_pandas_object(bundle.test[bundle.id_column], index=False).astype("uint64")
    return FeatureCandidate(
        name="gen_id_hash",
        train_feature=(train_hash % 100000).astype("float64") / 100000.0,
        test_feature=(test_hash % 100000).astype("float64") / 100000.0,
        source_family="fallback",
        proxy_score=0.0,
        missing_ratio=0.0,
        compute_cost=0.01,
        metadata={"join_complexity": 0.0},
    )


def build_fallback_feature_set(bundle) -> FeatureSet:
    return build_fallback_candidate(bundle).to_feature_set()
