from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": 1,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate output features against holdout labels.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to dataset directory with test_labels.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Path to output/ with train.csv and test.csv.")
    return parser


def infer_columns(train_out: pd.DataFrame, test_out: pd.DataFrame, labels: pd.DataFrame) -> tuple[str, str]:
    shared_ids = set(train_out.columns) & set(test_out.columns) & set(labels.columns)
    if not shared_ids:
        raise ValueError("Failed to infer id column: no common column in output train/test and labels.")
    preferred = ("client_id", "row_id", "id")
    id_column = ""
    lowered = {column.lower(): column for column in shared_ids}
    for token in preferred:
        if token in lowered:
            id_column = lowered[token]
            break
    if not id_column:
        id_column = sorted(shared_ids)[0]

    target_candidates = [column for column in train_out.columns if column not in test_out.columns]
    target_column = "target" if "target" in train_out.columns else (target_candidates[0] if target_candidates else "")
    if not target_column:
        raise ValueError("Failed to infer target column in output/train.csv.")
    return id_column, target_column


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    train_out = pd.read_csv(output_dir / "train.csv")
    test_out = pd.read_csv(output_dir / "test.csv")
    labels = pd.read_csv(dataset_dir / "test_labels.csv")

    id_column, target_column = infer_columns(train_out=train_out, test_out=test_out, labels=labels)

    feature_cols = [column for column in train_out.columns if column not in {id_column, target_column}]
    if not feature_cols:
        raise ValueError("No generated feature columns found in output/train.csv.")

    X_train = train_out[feature_cols].copy()
    y_train = train_out[target_column].copy()
    X_test = test_out[feature_cols].copy()

    cat_features: list[int] = []
    for idx, column in enumerate(feature_cols):
        if X_train[column].dtype == "object":
            X_train[column] = X_train[column].fillna("__nan__").astype(str)
            X_test[column] = X_test[column].fillna("__nan__").astype(str)
            cat_features.append(idx)
        else:
            X_train[column] = pd.to_numeric(X_train[column], errors="coerce").fillna(-999.0)
            X_test[column] = pd.to_numeric(X_test[column], errors="coerce").fillna(-999.0)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, cat_features=cat_features or None)

    merged = test_out[[id_column] + feature_cols].merge(labels[[id_column, "target"]], on=id_column, how="inner")
    if merged.empty:
        raise ValueError("No matching rows between output/test.csv and dataset test_labels.csv.")
    y_test = merged["target"]
    X_eval = merged[feature_cols].copy()
    for column in feature_cols:
        if column in X_eval.columns and X_eval[column].dtype == "object":
            X_eval[column] = X_eval[column].fillna("__nan__").astype(str)
        else:
            X_eval[column] = pd.to_numeric(X_eval[column], errors="coerce").fillna(-999.0)

    holdout_proba = model.predict_proba(X_eval)[:, 1]
    holdout_auc = float(roc_auc_score(y_test, holdout_proba))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(
        CatBoostClassifier(**CATBOOST_PARAMS),
        X_train,
        y_train,
        cv=skf,
        scoring="roc_auc",
    )

    print(f"id_column={id_column}")
    print(f"target_column={target_column}")
    print(f"n_features={len(feature_cols)}")
    print(f"train_rows={len(X_train)}")
    print(f"holdout_rows={len(merged)}")
    print(f"cv_auc_mean={cv_auc.mean():.6f}")
    print(f"cv_auc_std={cv_auc.std():.6f}")
    print(f"holdout_auc={holdout_auc:.6f}")


if __name__ == "__main__":
    main()
