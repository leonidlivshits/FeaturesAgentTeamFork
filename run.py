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
    print("Generator mode:", result.decision_trace.get("effective_mode", "n/a"))
    print("Dataset type:", result.decision_trace.get("dataset_type", "n/a"))
    print("Dataset type confidence:", result.decision_trace.get("dataset_type_confidence", "n/a"))
    print("Selected plan source:", result.decision_trace.get("selected_plan_source", "n/a"))
    print("Selected LLM provider:", result.decision_trace.get("selected_llm_provider", "n/a"))
    print("LLM was used:", result.decision_trace.get("llm_was_used", False))
    print("LLM candidate sets:", result.decision_trace.get("llm_candidate_sets", 0))
    print("LLM providers seen:", ", ".join(result.decision_trace.get("llm_providers_seen", ["n/a"])))


if __name__ == "__main__":
    main()
