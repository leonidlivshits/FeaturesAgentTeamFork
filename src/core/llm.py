from __future__ import annotations

import logging
import os
from typing import Any

from src.core.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    value = (os.getenv(name) or "").strip().strip("'\"")
    return value


def _has_non_empty_env_value(name: str) -> bool:
    value = _env(name)
    return bool(value)


def _llm_disabled() -> bool:
    return _env("FEATURES_AGENT_DISABLE_LLM").lower() in {"1", "true", "yes"}


def _can_use_gigachat() -> bool:
    if not _has_non_empty_env_value("GIGACHAT_CREDENTIALS"):
        return False
    if not _has_non_empty_env_value("GIGACHAT_SCOPE"):
        return False
    try:
        import langchain_gigachat.chat_models  # noqa: F401
    except Exception:
        return False
    return True


def _can_use_openrouter() -> bool:
    if not _has_non_empty_env_value("OPENROUTER_API_KEY"):
        return False
    try:
        import langchain_openai  # noqa: F401
    except Exception:
        return False
    return True


def get_effective_llm_provider() -> str:
    if _llm_disabled():
        return "none"

    requested = (_env("LLM_PROVIDER") or DEFAULT_CONFIG.default_llm_provider).lower()
    if requested not in {"auto", "gigachat", "openrouter"}:
        logger.warning("Unsupported LLM_PROVIDER='%s', using auto", requested)
        requested = "auto"

    if requested == "gigachat":
        return "gigachat" if _can_use_gigachat() else "none"
    if requested == "openrouter":
        return "openrouter" if _can_use_openrouter() else "none"

    if _can_use_gigachat():
        return "gigachat"
    if _can_use_openrouter():
        return "openrouter"
    return "none"


def is_llm_available() -> bool:
    return get_effective_llm_provider() != "none"


def validate_llm_configuration() -> None:
    if _llm_disabled():
        return

    requested = (_env("LLM_PROVIDER") or DEFAULT_CONFIG.default_llm_provider).lower()
    mode = (_env("FEATURES_AGENT_MODE") or "auto").lower()

    if requested not in {"auto", "gigachat", "openrouter"}:
        raise ValueError(f"Unsupported LLM_PROVIDER='{requested}'. Expected one of: auto, gigachat, openrouter.")

    if requested == "gigachat":
        if not _has_non_empty_env_value("GIGACHAT_CREDENTIALS"):
            raise ValueError("LLM_PROVIDER=gigachat but GIGACHAT_CREDENTIALS is empty.")
        if not _has_non_empty_env_value("GIGACHAT_SCOPE"):
            raise ValueError("LLM_PROVIDER=gigachat but GIGACHAT_SCOPE is empty.")
        if not _can_use_gigachat():
            raise ValueError("LLM_PROVIDER=gigachat but langchain_gigachat is unavailable.")
        return

    if requested == "openrouter":
        if not _has_non_empty_env_value("OPENROUTER_API_KEY"):
            raise ValueError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is empty.")
        if not _can_use_openrouter():
            raise ValueError("LLM_PROVIDER=openrouter but langchain_openai is unavailable. Run uv sync.")
        return

    if requested == "auto" and mode == "llm" and not is_llm_available():
        raise ValueError(
            "FEATURES_AGENT_MODE=llm but no LLM provider is available. "
            "Set gigachat/openrouter credentials or switch mode."
        )


def get_gigachat_client(timeout: int = 25) -> Any | None:
    credentials = _env("GIGACHAT_CREDENTIALS")
    scope = _env("GIGACHAT_SCOPE")
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


def get_openrouter_client(timeout: int = 25) -> Any | None:
    api_key = _env("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = _env("OPENROUTER_MODEL") or DEFAULT_CONFIG.openrouter_model
    base_url = _env("OPENROUTER_BASE_URL") or DEFAULT_CONFIG.openrouter_base_url
    referer = _env("OPENROUTER_HTTP_REFERER")
    app_title = _env("OPENROUTER_APP_TITLE")

    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        logger.warning("langchain_openai is not available, openrouter disabled")
        return None

    extra_headers: dict[str, str] = {}
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if app_title:
        extra_headers["X-Title"] = app_title

    try:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            timeout=timeout,
            default_headers=extra_headers or None,
        )
    except TypeError:
        # Backward-compatible kwargs for older langchain_openai versions.
        try:
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.0,
                timeout=timeout,
                default_headers=extra_headers or None,
            )
        except Exception as error:
            logger.warning("Failed to initialize OpenRouter client: %s", error)
            return None
    except Exception as error:
        logger.warning("Failed to initialize OpenRouter client: %s", error)
        return None


def get_llm_client(timeout: int = 25) -> Any | None:
    provider = get_effective_llm_provider()
    if provider == "gigachat":
        return get_gigachat_client(timeout=timeout)
    if provider == "openrouter":
        return get_openrouter_client(timeout=timeout)
    return None
