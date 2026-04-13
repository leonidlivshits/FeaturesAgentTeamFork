from __future__ import annotations

import pandas as pd

from src.data.loaders import _looks_like_target_table, _should_skip_aux_file


def test_should_skip_aux_file_detects_reserved_names() -> None:
    assert _should_skip_aux_file("train.csv")
    assert _should_skip_aux_file("test_labels.csv")
    assert _should_skip_aux_file("any_labels.csv")
    assert not _should_skip_aux_file("client_data.csv")


def test_detects_target_like_aux_table_by_schema() -> None:
    frame = pd.DataFrame(
        {
            "client_id": ["1", "2", "3"],
            "target": [0, 1, 0],
        }
    )
    assert _looks_like_target_table(
        frame=frame,
        table_name="labels_dump",
        id_column="client_id",
        target_column="target",
    )


def test_does_not_flag_normal_feature_table() -> None:
    frame = pd.DataFrame(
        {
            "client_id": ["1", "2", "3"],
            "age": [45, 33, 28],
            "income": [1000, 2300, 1800],
            "segment": ["a", "b", "a"],
        }
    )
    assert not _looks_like_target_table(
        frame=frame,
        table_name="client_data",
        id_column="client_id",
        target_column="target",
    )
