"""Tool call argument validation using pydantic.

Validates tool call arguments against the tool's pydantic ``BaseModel``
parameter schema.  This replaces the upstream AJV + TypeBox approach with
pydantic's built-in validation.

Upstream reference: ``packages/ai/src/utils/validation.ts``
"""

from __future__ import annotations

import json

from otter_ai.exceptions import ValidationError as OtterValidationError
from otter_ai.types import Tool, ToolCall


def validate_tool_call(tools: list[Tool], tool_call: ToolCall) -> object:
    """Find a tool by name and validate its arguments.

    Parameters
    ----------
    tools:
        List of tool definitions to search.
    tool_call:
        The tool call from the LLM.

    Returns
    -------
    The validated (and potentially coerced) arguments dict.

    Raises
    ------
    RuntimeError
        If the tool is not found or validation fails.
    """
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        msg = f'Tool "{tool_call.name}" not found'
        raise OtterValidationError(msg)
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> object:
    """Validate tool call arguments against the tool's parameter schema.

    Uses ``pydantic.TypeAdapter`` to validate and coerce the raw arguments
    dict into the shape defined by the tool's ``parameters`` model class.

    Parameters
    ----------
    tool:
        The tool definition with a pydantic ``BaseModel`` subclass as
        ``parameters``.
    tool_call:
        The tool call from the LLM.

    Returns
    -------
    The validated arguments as a dict.

    Raises
    ------
    RuntimeError
        If validation fails, with a formatted error message listing
        all issues.
    """
    try:
        model_cls = tool.parameters
        result = model_cls.model_validate(tool_call.arguments)
        return result.model_dump()
    except Exception as exc:
        # Handle both pydantic ValidationError and other unexpected errors
        errors = ""
        if hasattr(exc, "errors"):
            errors = (
                "\n".join(
                    f"  - {err['loc'][-1] if err['loc'] else 'root'}: {err['msg']}"
                    for err in exc.errors()
                )
                or "Unknown validation error"
            )

        received = json.dumps(tool_call.arguments, indent=2)
        msg = (
            f'Validation failed for tool "{tool_call.name}":\n'
            f"{errors}\n\n"
            f"Received arguments:\n{received}"
        )
        raise OtterValidationError(msg) from exc
