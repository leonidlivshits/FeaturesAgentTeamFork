from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "output"


def run_for_dataset(dataset_dir: Path) -> tuple[int, float, str, str]:
    env = os.environ.copy()
    env["FEATURES_AGENT_DATA_DIR"] = str(dataset_dir)
    env["FEATURES_AGENT_OUTPUT_DIR"] = str(OUTPUT_ROOT / dataset_dir.name)

    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "run.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    return proc.returncode, elapsed, proc.stdout, proc.stderr


def main() -> None:
    dataset_dirs = sorted(path for path in DATA_ROOT.glob("dataset_*") if path.is_dir())
    if not dataset_dirs:
        raise FileNotFoundError("No dataset_* directories found in data/.")

    for dataset_dir in dataset_dirs:
        code, elapsed, stdout, stderr = run_for_dataset(dataset_dir)
        print(f"=== {dataset_dir.name} ===")
        print(f"runtime_sec={elapsed:.2f}")
        print(f"returncode={code}")
        if stdout.strip():
            print("stdout:")
            print(stdout[-3000:])
        if stderr.strip():
            print("stderr:")
            print(stderr[-3000:])


if __name__ == "__main__":
    main()
