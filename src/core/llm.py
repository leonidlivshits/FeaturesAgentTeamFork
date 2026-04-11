from __future__ import annotations

import logging
import os
from typing import Any

from src.core.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _has_non_empty_env_value(name: str) -> bool:
    value = (os.getenv(name) or "").strip().strip("'\"")
    return bool(value)


def is_llm_available() -> bool:
    if (os.getenv("FEATURES_AGENT_DISABLE_LLM", "0").strip().lower()) in {"1", "true", "yes"}:
        return False

    if not _has_non_empty_env_value("GIGACHAT_CREDENTIALS"):
        return False
    if not _has_non_empty_env_value("GIGACHAT_SCOPE"):
        return False

    try:
        import langchain_gigachat.chat_models  # noqa: F401
    except Exception:
        return False

    return True


def get_gigachat_client(timeout: int = 25) -> Any | None:
    credentials = (os.getenv("GIGACHAT_CREDENTIALS") or "").strip().strip("'\"")
    scope = (os.getenv("GIGACHAT_SCOPE") or "").strip().strip("'\"")
    if not credentials or not scope:
        return None

    try:
        from langchain_gigachat.chat_models import GigaChat
    except Exception:
        logger.warning("langchain_gigachat is not available, llm-guided generator disabled")
        return None

    try:
        return GigaChat(
            credentials=credentials,
            scope=scope,
            model=DEFAULT_CONFIG.model_name,
            temperature=0.0,
            timeout=timeout,
            verify_ssl_certs=False,
        )
    except Exception as error:
        logger.warning("Failed to initialize GigaChat client: %s", error)
        return None
