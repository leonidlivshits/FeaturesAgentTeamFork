"""Entrypoint for the feature generation agent."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from src.core.logging import configure_logging
from src.core.runtime import apply_reproducibility
from src.core.config import DATA_DIR, DEFAULT_CONFIG, OUTPUT_DIR
from src.data.loaders import read_csv_auto
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

    common = [column for column in train.columns if column in test.columns]
    if not common and train.shape[1] > 0 and test.shape[1] > 0:
        # Force at least one shared id-like column by position.
        train_first = str(train.columns[0])
        test_first = str(test.columns[0])
        if test_first != train_first and train_first not in test.columns:
            test = test.rename(columns={test_first: train_first})
        common = [column for column in train.columns if column in test.columns]

    id_column = common[0] if common else str(train.columns[0])
    train_hash_source = train[id_column] if id_column in train.columns else pd.Series(range(len(train)))
    test_hash_source = test[id_column] if id_column in test.columns else pd.Series(range(len(test)))

    train_feature = (pd.util.hash_pandas_object(train_hash_source, index=False).astype("uint64") % 100000).astype("float64") / 100000.0
    test_feature = (pd.util.hash_pandas_object(test_hash_source, index=False).astype("uint64") % 100000).astype("float64") / 100000.0

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = train.copy()
    test_out = test.copy()
    feature_name = "gen_emergency_id_hash"
    train_out[feature_name] = train_feature.reset_index(drop=True)
    test_out[feature_name] = test_feature.reset_index(drop=True)
    train_out.to_csv(output_dir / "train.csv", index=False)
    test_out.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
