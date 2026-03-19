"""Attempts to parse potentially incomplete JSON during streaming.

Always returns a valid object, even if the JSON is incomplete.

Upstream reference: ``packages/ai/src/utils/json-parse.ts``
"""

from __future__ import annotations

import json
from typing import Any

from partial_json_parser import loads as partial_parse  # type: ignore[import-untyped]


def parse_streaming_json(partial_json: str | None) -> Any:
    """Parse potentially incomplete JSON from a streaming response.

    Tries standard ``json.loads`` first (fastest for complete JSON), then
    falls back to ``partial-json`` for incomplete JSON.  Returns an empty
    dict if all parsing fails.

    Parameters
    ----------
    partial_json:
        The partial JSON string from streaming, or ``None`` / ``undefined``.

    Returns
    -------
    Parsed object or empty dict if parsing fails.
    """
    if not partial_json or not partial_json.strip():
        return {}

    # Try standard parsing first (fastest for complete JSON)
    try:
        return json.loads(partial_json)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try partial-json for incomplete JSON
    try:
        result = partial_parse(partial_json)
        return result if result is not None else {}
    except Exception:
        return {}
