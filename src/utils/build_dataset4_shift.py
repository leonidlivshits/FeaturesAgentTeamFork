from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


RANDOM_SEED = 42


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    return path.read_text(errors="ignore")


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    return [], []
                headers = [str(header) for header in reader.fieldnames]
                rows = [dict(row) for row in reader]
                return headers, rows
        except Exception as error:
            last_error = error
            continue
    if last_error is not None:
        raise last_error
    return [], []


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _parse_target(value: str) -> int | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"", "none", "null", "nan"}:
        return None
    try:
        parsed = int(float(token))
    except Exception:
        return None
    if parsed not in (0, 1):
        return None
    return parsed


def _collect_labeled_rows(source_dir: Path) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for file_name in ("train.csv", "test.csv"):
        headers, rows = _read_csv_rows(source_dir / file_name)
        if not headers:
            continue
        required = {"user_id", "product_id", "target"}
        if not required.issubset(set(headers)):
            raise ValueError(f"{file_name} must contain columns {sorted(required)}")
        for row in rows:
            target = _parse_target(row.get("target", ""))
            if target is None:
                continue
            frames.append(
                {
                    "user_id": str(row["user_id"]),
                    "product_id": str(row["product_id"]),
                    "target": str(target),
                }
            )
    return frames


def _load_product_department(source_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    headers, rows = _read_csv_rows(source_dir / "products.csv")
    if not headers:
        raise ValueError("products.csv is missing or empty.")
    required = {"product_id", "department_id"}
    if not required.issubset(set(headers)):
        raise ValueError("products.csv must contain product_id and department_id.")

    product_to_dept: dict[str, str] = {}
    for row in rows:
        product_to_dept[str(row["product_id"])] = str(row["department_id"])

    dept_labels: dict[str, str] = {}
    dept_headers, dept_rows = _read_csv_rows(source_dir / "departments.csv")
    if dept_headers and {"department_id", "department"}.issubset(set(dept_headers)):
        for row in dept_rows:
            dept_labels[str(row["department_id"])] = str(row["department"])
    return product_to_dept, dept_labels


def _pick_holdout_departments(
    records: list[dict[str, str]],
    product_to_dept: dict[str, str],
) -> set[str]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "pos": 0})
    total_count = 0
    total_pos = 0
    for row in records:
        dept_id = product_to_dept.get(row["product_id"], "__missing__")
        target = int(row["target"])
        stats[dept_id]["count"] += 1
        stats[dept_id]["pos"] += target
        total_count += 1
        total_pos += target
    if not stats:
        return set()

    global_rate = total_pos / max(1, total_count)
    min_rows = max(3000, int(total_count * 0.03))
    scored: list[tuple[float, str]] = []
    for dept_id, item in stats.items():
        count = int(item["count"])
        if count < min_rows:
            continue
        rate = item["pos"] / max(1, count)
        score = abs(rate - global_rate) * math.log1p(count)
        scored.append((score, dept_id))
    scored.sort(reverse=True)
    selected = [dept for _, dept in scored[:2]]
    return set(selected)


def _copy_aux_tables(source_dir: Path, target_dir: Path) -> None:
    for csv_path in sorted(source_dir.glob("*.csv")):
        if csv_path.name in {"train.csv", "test.csv"}:
            continue
        shutil.copy2(csv_path, target_dir / csv_path.name)


def build_dataset4(source_dir: Path, target_dir: Path) -> dict[str, object]:
    records = _collect_labeled_rows(source_dir)
    if len(records) < 100000:
        raise ValueError(f"Not enough labeled records in dataset_1: {len(records)}")

    product_to_dept, dept_labels = _load_product_department(source_dir)
    holdout_departments = _pick_holdout_departments(records=records, product_to_dept=product_to_dept)
    if not holdout_departments:
        raise ValueError("Failed to find holdout departments for dataset_4.")

    train_rows_raw: list[dict[str, str]] = []
    test_rows_raw: list[dict[str, str]] = []
    for row in records:
        dept_id = product_to_dept.get(row["product_id"], "__missing__")
        if dept_id in holdout_departments:
            test_rows_raw.append(row)
        else:
            train_rows_raw.append(row)

    if len(train_rows_raw) < 100000 or len(test_rows_raw) < 50000:
        raise ValueError(
            f"Split too imbalanced train={len(train_rows_raw)} test={len(test_rows_raw)} for dataset_4."
        )

    # Reassign unique row_id across full split to avoid collisions in local holdout evaluation.
    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    test_labels: list[dict[str, str]] = []
    current_id = 1
    for row in train_rows_raw:
        train_rows.append(
            {
                "row_id": str(current_id),
                "user_id": row["user_id"],
                "product_id": row["product_id"],
                "target": row["target"],
            }
        )
        current_id += 1
    for row in test_rows_raw:
        row_id = str(current_id)
        test_rows.append(
            {
                "row_id": row_id,
                "user_id": row["user_id"],
                "product_id": row["product_id"],
            }
        )
        test_labels.append({"row_id": row_id, "target": row["target"]})
        current_id += 1

    target_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(target_dir / "train.csv", ["row_id", "user_id", "product_id", "target"], train_rows)
    _write_csv(target_dir / "test.csv", ["row_id", "user_id", "product_id"], test_rows)
    _write_csv(target_dir / "test_labels.csv", ["row_id", "target"], test_labels)
    _copy_aux_tables(source_dir=source_dir, target_dir=target_dir)

    source_readme = _read_text(source_dir / "readme.txt")
    holdout_dept_names = [dept_labels.get(dept, dept) for dept in sorted(holdout_departments)]
    readme_note = (
        "\n\n"
        "Synthetic stress split for generalization checks:\n"
        "- built from dataset_1 labeled pool (train + test targets)\n"
        "- test is OOD holdout by department_id (product domain shift)\n"
        f"- holdout departments: {', '.join(holdout_dept_names)}\n"
        "- test labels are stored in test_labels.csv for offline validation\n"
        "- pipeline input test.csv has no target column\n"
    )
    (target_dir / "readme.txt").write_text(source_readme + readme_note, encoding="utf-8")

    train_pos = sum(int(item["target"]) for item in train_rows)
    test_pos = sum(int(item["target"]) for item in test_labels)
    metadata = {
        "source_dataset": str(source_dir),
        "target_dataset": str(target_dir),
        "random_seed": RANDOM_SEED,
        "rows_total_labeled": len(records),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_positive_rate": round(train_pos / max(1, len(train_rows)), 6),
        "test_positive_rate": round(test_pos / max(1, len(test_rows)), 6),
        "holdout_department_ids": sorted(holdout_departments),
        "holdout_department_names": holdout_dept_names,
    }
    (target_dir / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source_dir = root / "datasets" / "dataset_1"
    target_dir = root / "datasets" / "dataset_4"
    metadata = build_dataset4(source_dir=source_dir, target_dir=target_dir)
    print("dataset_4 ready")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
