"""Shared fixtures for otter-agent-core tests.

Mirrors upstream: ``packages/agent/test/`` infrastructure.
"""

from __future__ import annotations

from typing import Any

from otter_ai.models import get_model
from otter_ai import EventStream
from otter_ai.types import (
    AssistantMessage,
    Usage,
    UsageCost,
)

# ============================================================================
# Mock stream (mirrors packages/agent/test/agent-loop.test.ts MockAssistantStream)
# ============================================================================


class MockAssistantStream(EventStream[Any, Any]):
    """Mock event stream that mimics :class:`AssistantMessageEventStream` for testing."""

    def __init__(self) -> None:  # type: ignore[override]
        super().__init__(
            is_done=lambda event: event.type in ("done", "error"),  # type: ignore[union-attr]
            extract_result=lambda event: (  # type: ignore[union-attr]
                event.message
                if event.type == "done"  # type: ignore[union-attr]
                else event.error
                if event.type == "error"  # type: ignore[union-attr]
                else (_ for _ in ()).throw(RuntimeError(f"Unexpected event type: {event.type}"))  # type: ignore[union-attr]
            ),
        )


# ============================================================================
# Helper factories (mirrors upstream test helpers)
# ============================================================================


def create_usage() -> Usage:
    """Create a zeroed-out Usage object."""
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=UsageCost(),
    )


def create_model() -> Any:
    """Create a mock model for testing."""
    return get_model("openai-responses", "gpt-4o")


def create_assistant_message(text: str) -> AssistantMessage:
    """Create a simple assistant message with text content.

    Mirrors: ``packages/agent/test/agent.test.ts → createAssistantMessage()``
    """
    return AssistantMessage(
        role="assistant",
        content=[{"type": "text", "text": text}],
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stop_reason="stop",
        timestamp=0,
    )
