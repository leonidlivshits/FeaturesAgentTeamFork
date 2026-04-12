from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JoinEdge:
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    coverage_left: float
    coverage_right: float
    name_score: float
    readme_score: float
    total_score: float


@dataclass(frozen=True)
class JoinPlan:
    path: tuple[JoinEdge, ...]
    output_table: str
    output_key: str
    hop_count: int
    total_score: float


@dataclass
class TableProfile:
    name: str
    role: str
    row_count: int
    columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    time_columns: tuple[str, ...]
    key_candidates: tuple[str, ...]
    repeated_key_candidates: tuple[str, ...]
    leakage_risk_columns: tuple[str, ...]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    dataset_type: str
    id_column: str
    target_column: str
    group_column: str | None
    time_column: str | None
    base_table: str
    table_profiles: dict[str, TableProfile]
    join_edges: list[JoinEdge]
    join_plans: list[JoinPlan]
    leakage_risk_columns: set[str]
    metadata: dict[str, object] = field(default_factory=dict)
