from __future__ import annotations

from src.features.contracts import FeatureGenerator
from src.features.generators.categorical_frequency import CategoricalFrequencyGenerator
from src.features.generators.heuristic_numeric import NumericHeuristicGenerator
from src.features.generators.missingness import MissingnessGenerator


def build_generators() -> list[FeatureGenerator]:
    return [
        NumericHeuristicGenerator(),
        CategoricalFrequencyGenerator(),
        MissingnessGenerator(),
    ]

