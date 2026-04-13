from __future__ import annotations

import csv
import json
import math
import random
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


def _infer_id_column(headers: list[str]) -> str:
    lowered = {header.lower(): header for header in headers}
    if "client_id" in lowered:
        return lowered["client_id"]
    if "id" in lowered:
        return lowered["id"]
    for header in headers:
        token = header.lower()
        if "client" in token or "customer" in token or token.endswith("_id"):
            return header
    raise ValueError("Unable to infer id column.")


def _parse_target(value: str) -> int | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"", "nan", "none", "null"}:
        return None
    try:
        parsed = int(float(token))
    except Exception:
        return None
    if parsed not in (0, 1):
        return None
    return parsed


def _collect_labels(dataset_dir: Path, id_column: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    for file_name in ("train.csv", "test.csv"):
        headers, rows = _read_csv_rows(dataset_dir / file_name)
        if not headers:
            continue
        if id_column not in headers or "target" not in headers:
            continue
        for row in rows:
            sample_id = str(row.get(id_column, "")).strip()
            if not sample_id:
                continue
            target = _parse_target(row.get("target", ""))
            if target is None:
                continue
            labels[sample_id] = target
    return labels


def _month_score(count: int, positive_rate: float, global_rate: float) -> float:
    gap = abs(positive_rate - global_rate)
    size_bonus = math.log1p(max(1, count))
    return gap * size_bonus


def _pick_holdout_months(month_stats: dict[str, dict[str, int]], total_rows: int) -> set[str]:
    min_month_rows = max(120, int(total_rows * 0.03))
    total_pos = sum(item["pos"] for item in month_stats.values())
    global_rate = total_pos / max(1, total_rows)
    scored: list[tuple[float, str]] = []
    for month, stats in month_stats.items():
        count = int(stats["count"])
        if count < min_month_rows:
            continue
        rate = stats["pos"] / max(1, count)
        scored.append((_month_score(count, rate, global_rate), month))
    scored.sort(reverse=True)
    if not scored:
        return set()
    selected = [month for _, month in scored[:2]]
    return set(selected)


def _stratified_random_split(records: list[dict[str, str]], test_ratio: float = 0.25) -> tuple[set[str], set[str]]:
    rng = random.Random(RANDOM_SEED)
    by_target: dict[int, list[str]] = defaultdict(list)
    for row in records:
        by_target[int(row["target"])].append(row["client_id"])

    test_ids: set[str] = set()
    for target, sample_ids in by_target.items():
        shuffled = sample_ids[:]
        rng.shuffle(shuffled)
        k = max(1, int(len(shuffled) * test_ratio))
        chosen = shuffled[:k]
        test_ids.update(chosen)

    all_ids = {row["client_id"] for row in records}
    train_ids = all_ids - test_ids
    if not train_ids or not test_ids:
        # Final fallback: deterministic contiguous split.
        ordered = sorted(all_ids)
        border = max(1, int(len(ordered) * (1.0 - test_ratio)))
        train_ids = set(ordered[:border])
        test_ids = set(ordered[border:])
    return train_ids, test_ids


def build_dataset3(source_dir: Path, target_dir: Path) -> dict[str, object]:
    client_headers_raw, client_rows_raw = _read_csv_rows(source_dir / "client_data.csv")
    if not client_headers_raw or not client_rows_raw:
        raise ValueError("Source client_data.csv is empty or invalid.")

    artifact_columns = {
        header
        for header in client_headers_raw
        if not str(header).strip() or str(header).strip().lower().startswith("unnamed")
    }
    client_headers = [header for header in client_headers_raw if header not in artifact_columns]
    id_column = _infer_id_column(client_headers)
    month_column = "month" if "month" in client_headers else None

    label_map = _collect_labels(source_dir, id_column=id_column)
    if not label_map:
        raise ValueError("No labels found in dataset_2 train/test.")

    labeled_rows: list[dict[str, str]] = []
    month_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "pos": 0})
    for raw in client_rows_raw:
        row = {header: raw.get(header, "") for header in client_headers}
        sample_id = str(row.get(id_column, "")).strip()
        if not sample_id:
            continue
        target = label_map.get(sample_id)
        if target is None:
            continue
        month_value = str(row.get(month_column, "__missing__")).strip().lower() if month_column else "__missing__"
        labeled_rows.append(
            {
                "client_id": sample_id,
                "target": str(target),
                "month": month_value,
            }
        )
        month_stats[month_value]["count"] += 1
        month_stats[month_value]["pos"] += target

    if len(labeled_rows) < 200:
        raise ValueError(f"Too few labeled rows to build stress dataset: {len(labeled_rows)}")

    holdout_months = _pick_holdout_months(month_stats=month_stats, total_rows=len(labeled_rows))
    if holdout_months:
        train_ids = {row["client_id"] for row in labeled_rows if row["month"] not in holdout_months}
        test_ids = {row["client_id"] for row in labeled_rows if row["month"] in holdout_months}
    else:
        train_ids, test_ids = _stratified_random_split(labeled_rows)

    if len(train_ids) < 500 or len(test_ids) < 200:
        train_ids, test_ids = _stratified_random_split(labeled_rows)
        holdout_months = set()

    labels_by_id = {row["client_id"]: row["target"] for row in labeled_rows}
    train_rows = [{"client_id": sample_id, "target": labels_by_id[sample_id]} for sample_id in sorted(train_ids)]
    test_rows = [{"client_id": sample_id} for sample_id in sorted(test_ids)]
    test_label_rows = [{"client_id": sample_id, "target": labels_by_id[sample_id]} for sample_id in sorted(test_ids)]

    keep_ids = train_ids | test_ids
    client_rows = []
    for raw in client_rows_raw:
        row = {header: raw.get(header, "") for header in client_headers}
        sample_id = str(row.get(id_column, "")).strip()
        if sample_id in keep_ids:
            client_rows.append(row)

    # Re-order columns to keep id first and avoid accidental target leakage from aux table.
    aux_headers = [id_column] + [header for header in client_headers if header not in {id_column, "target"}]
    _write_csv(target_dir / "train.csv", ["client_id", "target"], train_rows)
    _write_csv(target_dir / "test.csv", ["client_id"], test_rows)
    _write_csv(target_dir / "test_labels.csv", ["client_id", "target"], test_label_rows)
    _write_csv(target_dir / "client_data.csv", aux_headers, client_rows)

    base_readme = _read_text(source_dir / "readme.txt")
    readme_note = (
        "\n\n"
        "Synthetic stress split for generalization checks:\n"
        "- built from dataset_2 labeled pool\n"
        "- test is OOD holdout by month when possible\n"
        "- test labels are stored separately in test_labels.csv for offline validation\n"
        "- pipeline input test.csv contains client_id only (no target)\n"
    )
    (target_dir / "readme.txt").write_text(base_readme + readme_note, encoding="utf-8")

    metadata = {
        "source_dataset": str(source_dir),
        "target_dataset": str(target_dir),
        "rows_total_labeled": len(labeled_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "holdout_months": sorted(holdout_months),
        "month_stats": month_stats,
        "random_seed": RANDOM_SEED,
    }
    (target_dir / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source_dir = root / "datasets" / "dataset_2"
    target_dir = root / "datasets" / "dataset_3"
    metadata = build_dataset3(source_dir=source_dir, target_dir=target_dir)
    print("dataset_3 ready")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
