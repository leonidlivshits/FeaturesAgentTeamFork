"""Entrypoint for the feature generation agent."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from src.core.logging import configure_logging
from src.core.runtime import apply_reproducibility
from src.core.config import DATA_DIR, DEFAULT_CONFIG, OUTPUT_DIR
from src.data.loaders import infer_key_columns, infer_key_columns_fallback, read_csv_auto
from src.pipeline.agent import run_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    load_dotenv()
    apply_reproducibility(DEFAULT_CONFIG.random_seed)

    try:
        result = run_pipeline()
    except Exception as error:
        logger.exception("Pipeline failed, generating emergency fallback output: %s", error)
        _write_emergency_submission(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
        print("Best feature set:", "emergency_fallback")
        print("Best CV AUC:", f"{0.0:.6f}")
        print("Generated features:", 1)
        print("ID column:", "n/a")
        print("Target column:", "n/a")
        print("Dataset type:", "n/a")
        print("Generator mode:", "fallback")
        print("Group column:", "n/a")
        print("Time column:", "n/a")
        print("Candidate pool size:", 0)
        print("CV strategy:", "fallback")
        print("Selected families:", "fallback")
        print("Selection strategy:", "fallback")
        print("Group CV used:", False)
        print("Selected plan source:", "n/a")
        print("Selected LLM provider:", "n/a")
        print("LLM was used:", False)
        print("LLM candidate sets:", 0)
        print("LLM in final:", False)
        print("LLM candidate count:", 0)
        print("LLM providers seen:", "n/a")
        return

    print("Best feature set:", result.best_feature_set)
    print("Best CV AUC:", f"{result.best_cv_auc:.6f}")
    if "selected_cv_std" in result.decision_trace:
        print("Best CV STD:", result.decision_trace.get("selected_cv_std"))
    if "selected_effective_score" in result.decision_trace:
        print("Effective score:", result.decision_trace.get("selected_effective_score"))
    print("Generated features:", result.generated_feature_count)
    print("ID column:", result.id_column)
    print("Target column:", result.target_column)
    print("Dataset type:", result.decision_trace.get("dataset_type", "n/a"))
    print(
        "Generator mode:",
        result.decision_trace.get(
            "effective_mode",
            result.decision_trace.get("requested_mode", "n/a"),
        ),
    )
    print("Group column:", result.decision_trace.get("group_column", "n/a"))
    print("Time column:", result.decision_trace.get("time_column", "n/a"))
    print("Candidate pool size:", result.decision_trace.get("candidate_pool_size", 0))
    print("CV strategy:", result.decision_trace.get("cv_strategy", "n/a"))
    print("Selected families:", ", ".join(result.decision_trace.get("selected_feature_families", [])))
    print("Selection strategy:", result.decision_trace.get("selection_strategy", "n/a"))
    print("Group CV used:", result.decision_trace.get("group_cv_used", False))
    print("Selected plan source:", result.decision_trace.get("selected_plan_source", "n/a"))
    print("Selected LLM provider:", result.decision_trace.get("selected_llm_provider", "n/a"))
    print("LLM was used:", result.decision_trace.get("llm_was_used", False))
    print(
        "LLM candidate sets:",
        result.decision_trace.get(
            "llm_candidate_sets",
            result.decision_trace.get("llm_candidate_count", 0),
        ),
    )
    print("LLM in final:", result.decision_trace.get("llm_in_final", False))
    print("LLM candidate count:", result.decision_trace.get("llm_candidate_count", 0))
    print("LLM providers seen:", ", ".join(result.decision_trace.get("llm_providers_seen", ["n/a"])))


def _write_emergency_submission(data_dir: Path, output_dir: Path) -> None:
    train = read_csv_auto(data_dir / "train.csv")
    test = read_csv_auto(data_dir / "test.csv")
    try:
        id_column, target_column = infer_key_columns(train=train, test=test)
    except Exception:
        train, test, id_column, target_column = infer_key_columns_fallback(train=train, test=test)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = train[[id_column, target_column]].copy()
    test_out = test[[id_column]].copy()

    # Stronger emergency fallback than id-hash: robust row-wise statistics from shared columns.
    shared = [column for column in train.columns if column in test.columns and column != id_column]
    numeric_shared = [column for column in shared if pd.api.types.is_numeric_dtype(train[column])]
    categorical_shared = [column for column in shared if column not in numeric_shared]

    feature_count = 0
    if numeric_shared:
        num_cols = numeric_shared[: min(8, len(numeric_shared))]
        train_num = train[num_cols].apply(pd.to_numeric, errors="coerce")
        test_num = test[num_cols].apply(pd.to_numeric, errors="coerce")
        train_out["gen_emergency_row_num_mean"] = train_num.mean(axis=1).fillna(0.0)
        test_out["gen_emergency_row_num_mean"] = test_num.mean(axis=1).fillna(0.0)
        feature_count += 1
        if feature_count < 5:
            train_out["gen_emergency_row_num_std"] = train_num.std(axis=1).fillna(0.0)
            test_out["gen_emergency_row_num_std"] = test_num.std(axis=1).fillna(0.0)
            feature_count += 1
        if feature_count < 5:
            train_out["gen_emergency_missing_ratio"] = train_num.isna().mean(axis=1)
            test_out["gen_emergency_missing_ratio"] = test_num.isna().mean(axis=1)
            feature_count += 1

    for column in categorical_shared[:2]:
        if feature_count >= 5:
            break
        freq = train[column].fillna("__nan__").astype(str).value_counts(normalize=True, dropna=False)
        feature_name = f"gen_emergency_{column}_freq"
        train_out[feature_name] = train[column].fillna("__nan__").astype(str).map(freq).fillna(0.0)
        test_out[feature_name] = test[column].fillna("__nan__").astype(str).map(freq).fillna(0.0)
        feature_count += 1

    if feature_count == 0:
        train_hash_source = train[id_column] if id_column in train.columns else pd.Series(range(len(train)))
        test_hash_source = test[id_column] if id_column in test.columns else pd.Series(range(len(test)))
        train_out["gen_emergency_id_hash"] = (
            (pd.util.hash_pandas_object(train_hash_source, index=False).astype("uint64") % 100000).astype("float64")
            / 100000.0
        )
        test_out["gen_emergency_id_hash"] = (
            (pd.util.hash_pandas_object(test_hash_source, index=False).astype("uint64") % 100000).astype("float64")
            / 100000.0
        )

    train_out.to_csv(output_dir / "train.csv", index=False)
    test_out.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
