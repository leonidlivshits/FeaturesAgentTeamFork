"""Entrypoint for the feature generation agent."""

from __future__ import annotations

from dotenv import load_dotenv

from src.core.logging import configure_logging
from src.core.runtime import apply_reproducibility
from src.core.config import DEFAULT_CONFIG
from src.pipeline.agent import run_pipeline


def main() -> None:
    configure_logging()
    load_dotenv()
    apply_reproducibility(DEFAULT_CONFIG.random_seed)

    result = run_pipeline()
    print("Best feature set:", result.best_feature_set)
    print("Best CV AUC:", f"{result.best_cv_auc:.6f}")
    print("Generated features:", result.generated_feature_count)
    print("ID column:", result.id_column)
    print("Target column:", result.target_column)
    print("Dataset type:", result.decision_trace.get("dataset_type", "n/a"))
    print("Group column:", result.decision_trace.get("group_column", "n/a"))
    print("Time column:", result.decision_trace.get("time_column", "n/a"))
    print("Candidate pool size:", result.decision_trace.get("candidate_pool_size", 0))
    print("CV strategy:", result.decision_trace.get("cv_strategy", "n/a"))
    print("Selected families:", ", ".join(result.decision_trace.get("selected_feature_families", [])))
    print("LLM candidate count:", result.decision_trace.get("llm_candidate_count", 0))
    print("LLM in final:", result.decision_trace.get("llm_in_final", False))


if __name__ == "__main__":
    main()
