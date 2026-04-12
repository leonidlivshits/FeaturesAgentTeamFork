from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.pipeline.agent as pipeline_agent


def test_pipeline_smoke_runs_on_synthetic_dataset(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()

    train = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5, 6],
            "user_id": [1, 1, 2, 2, 3, 3],
            "amount": [10.0, 12.0, 8.0, 7.5, 20.0, 21.0],
            "event_time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-01",
                "2024-01-03",
                "2024-01-02",
                "2024-01-04",
            ],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "row_id": [7, 8],
            "user_id": [3, 4],
            "amount": [22.0, 9.0],
            "event_time": ["2024-01-05", "2024-01-06"],
        }
    )
    orders = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 4],
            "order_id": [101, 102, 201, 301, 401],
            "basket_value": [50.0, 60.0, 30.0, 40.0, 20.0],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": [101, 101, 102, 201, 301, 401],
            "product_id": [1, 2, 3, 2, 4, 5],
            "item_price": [5.0, 6.0, 7.0, 8.0, 4.0, 3.0],
        }
    )

    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    orders.to_csv(data_dir / "orders.csv", index=False)
    order_items.to_csv(data_dir / "order_items.csv", index=False)
    (data_dir / "readme.txt").write_text("Repeat purchase prediction with users, orders and order items.", encoding="utf-8")

    monkeypatch.setattr(pipeline_agent, "DATA_DIR", data_dir)
    monkeypatch.setattr(pipeline_agent, "OUTPUT_DIR", output_dir)
    monkeypatch.setenv("FEATURES_AGENT_DISABLE_LLM", "1")

    result = pipeline_agent.run_pipeline()

    assert 1 <= result.generated_feature_count <= 5
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "test.csv").exists()
