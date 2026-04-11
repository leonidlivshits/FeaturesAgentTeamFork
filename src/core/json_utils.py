from __future__ import annotations

import json
import re
from typing import Any


def strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def safe_json_load(text: str) -> Any:
    payload = strip_json_fences(text)
    if not payload:
        return {}

    try:
        return json.loads(payload)
    except Exception:
        pass

    object_match = re.search(r"\{[\s\S]*\}", payload)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    array_match = re.search(r"\[[\s\S]*\]", payload)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    return {}

