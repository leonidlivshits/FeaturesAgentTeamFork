from __future__ import annotations

from src.features.factory import LlmCandidatePlanner


def test_llm_parser_rejects_unknown_operation_and_malformed_json() -> None:
    planner = LlmCandidatePlanner()
    malformed = planner._parse_specs(
        response_text="{not-json",
        known_columns={"a", "b"},
        numeric_columns={"a", "b"},
        categorical_columns=set(),
    )
    invalid_op = planner._parse_specs(
        response_text='{"features": [{"name": "x", "operation": "unknown", "columns": ["a"]}]}',
        known_columns={"a", "b"},
        numeric_columns={"a", "b"},
        categorical_columns=set(),
    )

    assert malformed == []
    assert invalid_op == []


def test_llm_parser_accepts_valid_spec() -> None:
    planner = LlmCandidatePlanner()
    specs = planner._parse_specs(
        response_text='{"features": [{"name": "ratio_ab", "operation": "ratio", "columns": ["a", "b"]}]}',
        known_columns={"a", "b", "c"},
        numeric_columns={"a", "b"},
        categorical_columns={"c"},
    )

    assert len(specs) == 1
    assert specs[0].operation == "ratio"
    assert specs[0].columns == ("a", "b")
