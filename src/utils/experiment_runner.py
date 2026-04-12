from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import DATA_DIR, ROOT_DIR, DEFAULT_CONFIG
from src.core.logging import configure_logging
from src.core.runtime import apply_reproducibility
from src.pipeline.agent import run_pipeline
from src.utils.switch_dataset import (
    backup_current_data,
    clear_directory,
    copy_tree_contents,
    ensure_source_layout,
    resolve_path,
)


DEFAULT_DATASETS = ("datasets/dataset_1", "datasets/dataset_2")
DEFAULT_MODES = ("heuristic", "hybrid", "llm")
DEFAULT_MD_REPORT = ROOT_DIR / "docs" / "EXPERIMENTS_REPORT.md"
DEFAULT_JSONL_REPORT = ROOT_DIR / "docs" / "EXPERIMENTS_LOG.jsonl"
DEFAULT_BACKUP_ROOT = ROOT_DIR / "datasets" / "_backups"


@dataclass
class ExperimentRecord:
    timestamp: str
    dataset: str
    mode: str
    status: str
    elapsed_sec: float
    best_feature_set: str
    best_cv_auc: float
    selected_cv_std: float
    selected_effective_score: float
    generated_feature_count: int
    selection_strategy: str
    group_cv_used: bool
    dataset_type: str
    llm_was_used: bool
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ablations and stress tests across multiple datasets and modes."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=f"Dataset directories (default: {' '.join(DEFAULT_DATASETS)}).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        help=f"Generator modes for ablations (default: {' '.join(DEFAULT_MODES)}).",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=DEFAULT_MD_REPORT,
        help=f"Markdown report output path (default: {DEFAULT_MD_REPORT}).",
    )
    parser.add_argument(
        "--report-jsonl",
        type=Path,
        default=DEFAULT_JSONL_REPORT,
        help=f"JSONL report output path (default: {DEFAULT_JSONL_REPORT}).",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Do not restore original data/ after experiments.",
    )
    return parser


def run_single_experiment(dataset_dir: Path, mode: str) -> ExperimentRecord:
    timestamp = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    os.environ["FEATURES_AGENT_MODE"] = mode
    if mode == "llm":
        os.environ["FEATURES_AGENT_DISABLE_LLM"] = "0"

    try:
        result = run_pipeline()
        elapsed = time.perf_counter() - started
        trace = result.decision_trace
        return ExperimentRecord(
            timestamp=timestamp,
            dataset=str(dataset_dir),
            mode=mode,
            status="ok",
            elapsed_sec=round(float(elapsed), 3),
            best_feature_set=str(result.best_feature_set),
            best_cv_auc=round(float(result.best_cv_auc), 6),
            selected_cv_std=float(trace.get("selected_cv_std", 0.0)),
            selected_effective_score=float(trace.get("selected_effective_score", result.best_cv_auc)),
            generated_feature_count=int(result.generated_feature_count),
            selection_strategy=str(trace.get("selection_strategy", "n/a")),
            group_cv_used=bool(trace.get("group_cv_used", False)),
            dataset_type=str(trace.get("dataset_type", "n/a")),
            llm_was_used=bool(trace.get("llm_was_used", False)),
        )
    except Exception as error:
        elapsed = time.perf_counter() - started
        return ExperimentRecord(
            timestamp=timestamp,
            dataset=str(dataset_dir),
            mode=mode,
            status="error",
            elapsed_sec=round(float(elapsed), 3),
            best_feature_set="n/a",
            best_cv_auc=0.0,
            selected_cv_std=0.0,
            selected_effective_score=0.0,
            generated_feature_count=0,
            selection_strategy="n/a",
            group_cv_used=False,
            dataset_type="n/a",
            llm_was_used=False,
            error=str(error),
        )


def write_jsonl(records: list[ExperimentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_markdown(records: list[ExperimentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Experiments Report")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- random_seed: {DEFAULT_CONFIG.random_seed}")
    lines.append("")
    lines.append("| dataset | mode | status | auc | cv_std | effective | features | strategy | group_cv | llm_used | elapsed_sec |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for record in records:
        lines.append(
            "| {dataset} | {mode} | {status} | {auc:.6f} | {std:.6f} | {eff:.6f} | {nfeat} | {strategy} | {group_cv} | {llm_used} | {elapsed:.3f} |".format(
                dataset=Path(record.dataset).name,
                mode=record.mode,
                status=record.status,
                auc=record.best_cv_auc,
                std=record.selected_cv_std,
                eff=record.selected_effective_score,
                nfeat=record.generated_feature_count,
                strategy=record.selection_strategy,
                group_cv=int(record.group_cv_used),
                llm_used=int(record.llm_was_used),
                elapsed=record.elapsed_sec,
            )
        )
        if record.error:
            lines.append(f"|  |  | error | `{record.error}` |  |  |  |  |  |  |  |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report is produced by `src/utils/experiment_runner.py`.")
    lines.append("- Runs datasets sequentially to stress-test portability.")
    lines.append("- Modes represent ablations of generation strategy.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()

    configure_logging()
    load_dotenv()
    apply_reproducibility(DEFAULT_CONFIG.random_seed)

    dataset_dirs = [resolve_path(Path(path)) for path in args.datasets]
    for dataset_dir in dataset_dirs:
        ensure_source_layout(dataset_dir)

    data_dir = resolve_path(DATA_DIR)
    backup_root = resolve_path(DEFAULT_BACKUP_ROOT)
    original_backup = backup_current_data(data_dir=data_dir, backup_root=backup_root)

    original_mode = os.environ.get("FEATURES_AGENT_MODE")
    original_disable_llm = os.environ.get("FEATURES_AGENT_DISABLE_LLM")

    records: list[ExperimentRecord] = []
    try:
        for dataset_dir in dataset_dirs:
            clear_directory(data_dir)
            copy_tree_contents(dataset_dir, data_dir)
            for mode in args.modes:
                mode_name = mode.strip().lower()
                records.append(run_single_experiment(dataset_dir=dataset_dir, mode=mode_name))
    finally:
        if original_mode is None:
            os.environ.pop("FEATURES_AGENT_MODE", None)
        else:
            os.environ["FEATURES_AGENT_MODE"] = original_mode

        if original_disable_llm is None:
            os.environ.pop("FEATURES_AGENT_DISABLE_LLM", None)
        else:
            os.environ["FEATURES_AGENT_DISABLE_LLM"] = original_disable_llm

        if not args.keep_data and original_backup is not None:
            clear_directory(data_dir)
            copy_tree_contents(original_backup, data_dir)

    report_md = resolve_path(args.report_md)
    report_jsonl = resolve_path(args.report_jsonl)
    write_markdown(records, report_md)
    write_jsonl(records, report_jsonl)

    ok_count = sum(1 for record in records if record.status == "ok")
    print("OK: experiments finished")
    print(f"Runs total: {len(records)}")
    print(f"Runs successful: {ok_count}")
    print(f"Markdown report: {report_md}")
    print(f"JSONL report: {report_jsonl}")
    if original_backup is not None and not args.keep_data:
        print(f"Restored original data/ from backup: {original_backup}")


if __name__ == "__main__":
    main()
