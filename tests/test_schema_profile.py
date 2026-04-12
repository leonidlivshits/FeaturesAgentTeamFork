from __future__ import annotations

import pandas as pd

from src.data.loaders import DataBundle
from src.schema.profiler import build_dataset_profile


def test_profile_detects_deposit_like_and_join_plan() -> None:
    train = pd.DataFrame(
        {
            "application_id": [1, 2, 3, 4],
            "client_id": [10, 10, 20, 30],
            "income": [100.0, 120.0, 80.0, 95.0],
            "balance": [1000.0, 900.0, 700.0, 1100.0],
            "target": [1, 0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "application_id": [5, 6],
            "client_id": [20, 40],
            "income": [85.0, 130.0],
            "balance": [750.0, 1200.0],
        }
    )
    aux_tables = {
        "client_history": pd.DataFrame(
            {
                "client_id": [10, 10, 20, 30, 40],
                "payment_amount": [100.0, 120.0, 80.0, 95.0, 140.0],
                "region": ["a", "a", "b", "b", "c"],
            }
        )
    }
    bundle = DataBundle(
        train=train,
        test=test,
        aux_tables=aux_tables,
        data_readme="Deposit applications with clients and payment history.",
        id_column="application_id",
        target_column="target",
    )

    profile = build_dataset_profile(bundle)

    assert profile.dataset_type == "deposit_like"
    assert profile.group_column == "client_id"
    assert profile.join_plans
    assert profile.table_profiles["client_history"].role in {"application", "user", "generic"}


def test_profile_detects_repeat_purchase_like_with_two_hop_join() -> None:
    train = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5, 6],
            "user_id": [1, 1, 2, 2, 3, 3],
            "event_time": [
                "2024-01-01",
                "2024-01-03",
                "2024-01-02",
                "2024-01-04",
                "2024-01-02",
                "2024-01-05",
            ],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "row_id": [7, 8],
            "user_id": [1, 4],
            "event_time": ["2024-01-06", "2024-01-07"],
        }
    )
    aux_tables = {
        "orders": pd.DataFrame(
            {
                "user_id": [1, 1, 2, 3, 4],
                "order_id": [100, 101, 200, 300, 400],
                "order_time": ["2023-12-30", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-04"],
            }
        ),
        "order_items": pd.DataFrame(
            {
                "order_id": [100, 100, 101, 200, 300, 400],
                "product_id": [10, 11, 12, 10, 14, 15],
                "item_price": [5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
            }
        ),
    }
    bundle = DataBundle(
        train=train,
        test=test,
        aux_tables=aux_tables,
        data_readme="Repeat purchase prediction with users, orders and order items.",
        id_column="row_id",
        target_column="target",
    )

    profile = build_dataset_profile(bundle)

    assert profile.dataset_type == "repeat_purchase_like"
    assert profile.group_column == "user_id"
    assert profile.time_column == "event_time"
    assert any(plan.hop_count == 2 for plan in profile.join_plans)


def test_profile_marks_leakage_columns() -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature_a": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )
    test = pd.DataFrame({"id": [4, 5], "feature_a": [1.5, 2.5]})
    aux_tables = {
        "predictions": pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "target_prediction": [0.1, 0.9, 0.2, 0.3, 0.4],
            }
        )
    }
    bundle = DataBundle(
        train=train,
        test=test,
        aux_tables=aux_tables,
        data_readme="Generic dataset.",
        id_column="id",
        target_column="target",
    )

    profile = build_dataset_profile(bundle)

    assert "target_prediction" in profile.table_profiles["predictions"].leakage_risk_columns
