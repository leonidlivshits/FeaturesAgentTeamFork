from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = ROOT / "dist"
CHECK_SCRIPT = ROOT / "src" / "utils" / "check_submission.py"
ENV_FILE = ROOT / ".env"


WHITELIST_PATTERNS = [
    "run.py",
    "pyproject.toml",
    "README.md",
    ".env",
    ".env.example",
    "uv.lock",
    "src/**/*.py",
    "configs/**/*",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run checks and build submission zip.")
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip check_submission.py before zipping.",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="",
        help="Custom zip filename (example: submit.zip).",
    )
    return parser


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
    if not ENV_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {ENV_FILE}")

    env_values = parse_env_file(ENV_FILE)
    if parse_bool(env_values.get("FEATURES_AGENT_DISABLE_LLM")):
        return

    provider = (env_values.get("LLM_PROVIDER") or "auto").strip().lower()
    mode = (env_values.get("FEATURES_AGENT_MODE") or "auto").strip().lower()
    if provider not in {"auto", "gigachat", "openrouter"}:
        raise ValueError("Unsupported LLM_PROVIDER in .env. Expected: auto|gigachat|openrouter")

    if provider == "gigachat":
        if not env_values.get("GIGACHAT_CREDENTIALS", "").strip():
            raise ValueError("Empty GIGACHAT_CREDENTIALS in .env")
        if not env_values.get("GIGACHAT_SCOPE", "").strip():
            raise ValueError("Empty GIGACHAT_SCOPE in .env")
        return

    if provider == "openrouter":
        if not env_values.get("OPENROUTER_API_KEY", "").strip():
            raise ValueError("Empty OPENROUTER_API_KEY in .env")
        return

    if mode == "llm":
        has_gigachat = bool(env_values.get("GIGACHAT_CREDENTIALS", "").strip()) and bool(
            env_values.get("GIGACHAT_SCOPE", "").strip()
        )
        has_openrouter = bool(env_values.get("OPENROUTER_API_KEY", "").strip())
        if not (has_gigachat or has_openrouter):
            raise ValueError(
                "FEATURES_AGENT_MODE=llm requires at least one configured provider: "
                "GIGACHAT_CREDENTIALS+GIGACHAT_SCOPE or OPENROUTER_API_KEY"
            )


def run_local_checks() -> None:
    cmd = [sys.executable, str(CHECK_SCRIPT)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "check_submission.py failed.\n"
            f"STDOUT:\n{result.stdout[-5000:]}\n\n"
            f"STDERR:\n{result.stderr[-5000:]}"
        )


def collect_files() -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for pattern in WHITELIST_PATTERNS:
        for path in ROOT.glob(pattern):
            if not path.exists() or path.is_dir():
                continue
            rel = path.relative_to(ROOT)
            if rel.parts and rel.parts[0] in {".git", ".venv", "data", "output", "dist"}:
                continue
            if path not in seen:
                seen.add(path)
                files.append(path)

    if not files:
        raise RuntimeError("No files collected for submission zip.")
    return sorted(files)


def make_zip(files: list[Path], zip_name: str | None = None) -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if zip_name:
        final_name = zip_name if zip_name.endswith(".zip") else f"{zip_name}.zip"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name = f"submission_{timestamp}.zip"

    zip_path = DIST_DIR / final_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path, arcname=str(file_path.relative_to(ROOT)))

    return zip_path


def main() -> None:
    args = build_parser().parse_args()

    ensure_env_file()

    if not args.skip_check:
        run_local_checks()

    files = collect_files()
    zip_path = make_zip(files=files, zip_name=args.zip_name or None)

    print("OK: presubmit completed")
    print(f"Files packed: {len(files)}")
    print(f"Archive: {zip_path}")


if __name__ == "__main__":
    main()
