"""Entrypoint for the feature generation agent."""

from __future__ import annotations

from dotenv import load_dotenv

from src.core.logging import configure_logging
from src.pipeline.agent import run_pipeline


def main() -> None:
    configure_logging()
    load_dotenv()

    result = run_pipeline()
    print("Best feature set:", result.best_feature_set)
    print("Best CV AUC:", f"{result.best_cv_auc:.6f}")
    print("Generated features:", result.generated_feature_count)
    print("ID column:", result.id_column)
    print("Target column:", result.target_column)


if __name__ == "__main__":
    main()
