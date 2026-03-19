"""Schema helpers for building tool parameter definitions.

Replaces upstream ``@sinclair/typebox`` ``StringEnum`` with a pydantic-based
approach.

Upstream reference: ``packages/ai/src/utils/typebox-helpers.ts``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def string_enum_json_schema(
    values: Sequence[str],
    *,
    description: str | None = None,
    default: str | None = None,
) -> dict[str, Any]:
    """Build a JSON Schema dict for a string enum type.

    This is the pydantic equivalent of upstream's ``StringEnum()``, which
    created a TypeBox ``TUnsafe`` schema.  The returned dict can be used
    directly in provider API calls or embedded in larger schemas.

    Parameters
    ----------
    values:
        The allowed string values.
    description:
        Optional description for the schema.
    default:
        Optional default value.

    Returns
    -------
    A JSON Schema dict with ``type: "string"`` and ``enum`` field.

    Example
    -------
    >>> schema = string_enum_json_schema(["add", "subtract"], description="Operation")
    >>> schema["type"]
    'string'
    >>> schema["enum"]
    ['add', 'subtract']
    """
    schema: dict[str, Any] = {
        "type": "string",
        "enum": list(values),
    }
    if description is not None:
        schema["description"] = description
    if default is not None:
        schema["default"] = default
    return schema
