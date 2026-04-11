from __future__ import annotations

import logging
import os

from src.core.config import DEFAULT_CONFIG
from src.core.llm import is_llm_available
from src.features.contracts import FeatureGenerator
from src.features.generators.categorical_frequency import CategoricalFrequencyGenerator
from src.features.generators.heuristic_numeric import NumericHeuristicGenerator
from src.features.generators.llm_guided import LlmGuidedGenerator
from src.features.generators.missingness import MissingnessGenerator
from src.features.generators.relational_aggregates import RelationalAggregatesGenerator

logger = logging.getLogger(__name__)


def get_generator_mode() -> str:
    raw_mode = (os.getenv("FEATURES_AGENT_MODE") or DEFAULT_CONFIG.default_generator_mode).strip().lower()
    supported = {"auto", "hybrid", "heuristic", "llm"}
    if raw_mode not in supported:
        logger.warning("Unsupported FEATURES_AGENT_MODE='%s', fallback to 'auto'", raw_mode)
        return "auto"
    return raw_mode


def get_effective_generator_mode() -> str:
    mode = get_generator_mode()
    if mode == "auto":
        return "hybrid" if is_llm_available() else "heuristic"
    return mode


def build_generators() -> list[FeatureGenerator]:
    mode = get_effective_generator_mode()
    logger.info("Generator mode: requested=%s effective=%s", get_generator_mode(), mode)

    heuristic_generators: list[FeatureGenerator] = [
        NumericHeuristicGenerator(),
        RelationalAggregatesGenerator(),
        CategoricalFrequencyGenerator(),
        MissingnessGenerator(),
    ]

    if mode == "llm":
        return [LlmGuidedGenerator()]
    if mode == "heuristic":
        return heuristic_generators
    return [LlmGuidedGenerator(), *heuristic_generators]
