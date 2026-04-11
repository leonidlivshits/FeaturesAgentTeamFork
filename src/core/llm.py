from __future__ import annotations

import logging
import os
from typing import Any

from src.core.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def get_gigachat_client(timeout: int = 25) -> Any | None:
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    scope = os.getenv("GIGACHAT_SCOPE")
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

