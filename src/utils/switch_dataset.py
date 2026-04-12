from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_BACKUP_ROOT = ROOT / "datasets" / "_backups"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely switch active dataset in data/ with optional backup."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to dataset directory (must contain train.csv, test.csv, readme.txt).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Target data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=DEFAULT_BACKUP_ROOT,
        help=f"Backup root for previous data/ content (default: {DEFAULT_BACKUP_ROOT}).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup current data/ before switching.",
    )
    return parser


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def ensure_source_layout(source_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    required = ("train.csv", "test.csv", "readme.txt")
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Source dataset is incomplete: missing {missing} in {source_dir}"
        )


def has_any_content(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())


def copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def backup_current_data(data_dir: Path, backup_root: Path) -> Path | None:
    if not has_any_content(data_dir):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"data_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    copy_tree_contents(data_dir, backup_dir)
    return backup_dir


def main() -> None:
    args = build_parser().parse_args()

    source_dir = resolve_path(args.source)
    data_dir = resolve_path(args.data_dir)
    backup_root = resolve_path(args.backup_root)

    ensure_source_layout(source_dir)

    backup_dir: Path | None = None
    if not args.no_backup:
        backup_dir = backup_current_data(data_dir=data_dir, backup_root=backup_root)

    clear_directory(data_dir)
    copy_tree_contents(source_dir, data_dir)

    print("OK: dataset switched")
    print(f"Source: {source_dir}")
    print(f"Active data dir: {data_dir}")
    if backup_dir is not None:
        print(f"Backup of previous data: {backup_dir}")
    else:
        print("Backup of previous data: skipped (data/ was empty)")


if __name__ == "__main__":
    main()
