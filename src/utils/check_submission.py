from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
PYPROJECT_PATH = ROOT / "pyproject.toml"
RUN_PATH = ROOT / "run.py"
ENV_PATH = ROOT / ".env"

MAX_RUNTIME_SEC = 600
MAX_FEATURES = 5


def read_table(path: Path) -> pd.DataFrame:
    """Reads CSV with automatic separator detection."""
    assert path.exists(), f"File not found: {path}"
    return pd.read_csv(path, sep=None, engine="python")


def load_pyproject() -> dict:
    assert PYPROJECT_PATH.exists(), "Missing pyproject.toml"
    with PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


def get_project_dependencies(pyproject: dict) -> list[str]:
    project = pyproject.get("project", {})
    deps = project.get("dependencies", [])
    assert isinstance(deps, list), "project.dependencies in pyproject.toml must be a list"
    return [str(x).lower() for x in deps]


def parse_env_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            parsed[key] = value
    return parsed


def parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def ensure_env_file() -> None:
    assert ENV_PATH.exists(), "Missing .env"

    env_values = parse_env_file(ENV_PATH)
    if parse_bool(env_values.get("FEATURES_AGENT_DISABLE_LLM")):
        return

    provider = (env_values.get("LLM_PROVIDER") or "auto").strip().lower()
    mode = (env_values.get("FEATURES_AGENT_MODE") or "auto").strip().lower()

    if provider not in {"auto", "gigachat", "openrouter"}:
        raise AssertionError("Unsupported LLM_PROVIDER in .env. Expected: auto|gigachat|openrouter")

    if provider == "gigachat":
        assert env_values.get("GIGACHAT_CREDENTIALS", "").strip(), "Empty GIGACHAT_CREDENTIALS in .env"
        assert env_values.get("GIGACHAT_SCOPE", "").strip(), "Empty GIGACHAT_SCOPE in .env"
        return

    if provider == "openrouter":
        assert env_values.get("OPENROUTER_API_KEY", "").strip(), "Empty OPENROUTER_API_KEY in .env"
        return

    if mode == "llm":
        has_gigachat = bool(env_values.get("GIGACHAT_CREDENTIALS", "").strip()) and bool(
            env_values.get("GIGACHAT_SCOPE", "").strip()
        )
        has_openrouter = bool(env_values.get("OPENROUTER_API_KEY", "").strip())
        assert has_gigachat or has_openrouter, (
            "FEATURES_AGENT_MODE=llm requires at least one configured provider: "
            "GIGACHAT_CREDENTIALS+GIGACHAT_SCOPE or OPENROUTER_API_KEY"
        )


def ensure_required_files() -> None:
    assert RUN_PATH.exists(), "Missing run.py"
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), "Missing data/ directory"
    assert (DATA_DIR / "train.csv").exists(), "Missing data/train.csv"
    assert (DATA_DIR / "test.csv").exists(), "Missing data/test.csv"
    assert (DATA_DIR / "readme.txt").exists(), "Missing data/readme.txt"


def ensure_dependencies() -> None:
    pyproject = load_pyproject()
    deps = get_project_dependencies(pyproject)

    required_markers = [
        "catboost",
        "pandas",
        "numpy",
        "langchain-gigachat",
        "langchain-openai",
        "python-dotenv",
    ]
    missing = [dep for dep in required_markers if not any(dep in x for x in deps)]

    assert not missing, f"Missing required dependencies in pyproject.toml: {missing}"


def clean_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_solution() -> tuple[int, float, str, str]:
    env = os.environ.copy()

    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "run.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=MAX_RUNTIME_SEC,
        env=env,
    )
    elapsed = time.perf_counter() - start

    return proc.returncode, elapsed, proc.stdout, proc.stderr


def assert_output_files_exist() -> tuple[Path, Path]:
    train_out = OUTPUT_DIR / "train.csv"
    test_out = OUTPUT_DIR / "test.csv"

    assert train_out.exists(), "Missing output/train.csv after run"
    assert test_out.exists(), "Missing output/test.csv after run"

    assert train_out.stat().st_size > 0, "output/train.csv is empty"
    assert test_out.stat().st_size > 0, "output/test.csv is empty"

    return train_out, test_out


def assert_output_structure(
    input_train: pd.DataFrame,
    input_test: pd.DataFrame,
    output_train: pd.DataFrame,
    output_test: pd.DataFrame,
) -> None:
    # 1. Required columns
    for col in input_train.columns:
        assert col in output_train.columns, f"Missing required column in output/train.csv: {col}"
    for col in input_test.columns:
        assert col in output_test.columns, f"Missing required column in output/test.csv: {col}"

    # 2. Features consistency and limit
    reserved_train = set(input_train.columns)
    reserved_test = set(input_test.columns)

    feature_cols_train = [c for c in output_train.columns if c not in reserved_train]
    feature_cols_test = [c for c in output_test.columns if c not in reserved_test]

    assert feature_cols_train == feature_cols_test, (
        "Feature columns in output/train.csv and output/test.csv must match by names and order.\n"
        f"train features: {feature_cols_train}\n"
        f"test features: {feature_cols_test}"
    )

    assert 1 <= len(feature_cols_train) <= MAX_FEATURES, (
        f"Feature count must be in [1, {MAX_FEATURES}], got {len(feature_cols_train)}"
    )

    # 3. Non-empty feature values
    for col in feature_cols_train:
        assert not output_train[col].isna().all(), f"Feature {col} in train is fully NaN"
        assert not output_test[col].isna().all(), f"Feature {col} in test is fully NaN"

    # 4. Duplicate columns
    assert output_train.columns.is_unique, "Duplicate columns in output/train.csv"
    assert output_test.columns.is_unique, "Duplicate columns in output/test.csv"


def main() -> None:
    ensure_required_files()
    ensure_env_file()
    ensure_dependencies()

    input_train = read_table(DATA_DIR / "train.csv")
    input_test = read_table(DATA_DIR / "test.csv")

    clean_output_dir()

    try:
        returncode, elapsed, stdout, stderr = run_solution()
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"Solution exceeded time limit {MAX_RUNTIME_SEC} sec") from e

    assert returncode == 0, (
        "run.py failed.\n"
        f"Return code: {returncode}\n\n"
        f"STDOUT:\n{stdout[-5000:]}\n\n"
        f"STDERR:\n{stderr[-5000:]}"
    )

    assert elapsed <= MAX_RUNTIME_SEC, (
        f"Solution runtime is too long: {elapsed:.2f} sec. Limit: {MAX_RUNTIME_SEC} sec."
    )

    train_out_path, test_out_path = assert_output_files_exist()

    output_train = read_table(train_out_path)
    output_test = read_table(test_out_path)

    assert_output_structure(
        input_train=input_train,
        input_test=input_test,
        output_train=output_train,
        output_test=output_test,
    )

    print("OK: submit passed basic checks")
    print(f"Runtime: {elapsed:.2f} sec")
    print(f"Generated features: {len(output_test.columns) - len(input_test.columns)}")
    print(f"Output files: {train_out_path}, {test_out_path}")


if __name__ == "__main__":
    main()
