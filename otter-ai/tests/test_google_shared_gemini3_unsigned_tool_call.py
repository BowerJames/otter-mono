"""Tests for google-shared convertMessages — Gemini 3 unsigned tool calls.

Upstream reference: ``packages/ai/test/google-shared-gemini3-unsigned-tool-call.test.ts``
"""

from __future__ import annotations

from typing import Any

from otter_ai.providers.google_shared import convert_messages
from otter_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    ToolCall,
    Usage,
    UsageCost,
    UserMessage,
)

SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def _empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=UsageCost(),
    )


def _make_gemini3_model(model_id: str = "gemini-3-pro-preview") -> Model:
    return Model(
        id=model_id,
        name="Gemini 3 Pro Preview",
        api="google-generative-ai",  # type: ignore[assignment]
        provider="google",
        base_url="https://generativelanguage.googleapis.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=8192,
    )


class TestGoogleSharedGemini3UnsignedToolCalls:
    def test_uses_skip_thought_signature_validator_for_unsigned_tool_calls(self) -> None:
        model = _make_gemini3_model()
        now = 1000
        context = Context(
            messages=[
                UserMessage(role="user", content="Hi", timestamp=now),
                AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCall(
                            type="toolCall",
                            id="call_1",
                            name="bash",
                            arguments={"command": "ls -la"},
                            # No thoughtSignature: simulates Claude via Antigravity.
                        ),
                    ],
                    api="google-gemini-cli",
                    provider="google-antigravity",
                    model="claude-sonnet-4-20250514",
                    usage=_empty_usage(),
                    stop_reason="stop",
                    timestamp=now,
                ),
            ],
        )

        contents = convert_messages(model, context)

        model_turn: dict[str, Any] = next(c for c in contents if c.get("role") == "model")
        assert model_turn is not None

        # Should be a structured functionCall, NOT text fallback
        fc_part: dict[str, Any] | None = next(
            (p for p in model_turn["parts"] if p.get("functionCall") is not None), None
        )
        assert fc_part is not None
        assert fc_part["functionCall"]["name"] == "bash"
        assert fc_part["functionCall"]["args"] == {"command": "ls -la"}
        assert fc_part.get("thoughtSignature") == SKIP_THOUGHT_SIGNATURE

        # No text fallback should exist
        text_parts = [p for p in model_turn["parts"] if p.get("text") is not None]
        historical_text = [p for p in text_parts if "Historical context" in p.get("text", "")]
        assert len(historical_text) == 0

    def test_preserves_valid_thought_signature_when_present(self) -> None:
        model = _make_gemini3_model()
        now = 1000
        valid_sig = "AAAAAAAAAAAAAAAAAAAAAA=="
        context = Context(
            messages=[
                UserMessage(role="user", content="Hi", timestamp=now),
                AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCall(
                            type="toolCall",
                            id="call_1",
                            name="bash",
                            arguments={"command": "echo hi"},
                            thought_signature=valid_sig,
                        ),
                    ],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-3-pro-preview",
                    usage=_empty_usage(),
                    stop_reason="stop",
                    timestamp=now,
                ),
            ],
        )

        contents = convert_messages(model, context)
        model_turn: dict[str, Any] = next(c for c in contents if c.get("role") == "model")
        fc_part: dict[str, Any] | None = next(
            (p for p in model_turn["parts"] if p.get("functionCall") is not None), None
        )

        assert fc_part is not None
        assert fc_part.get("thoughtSignature") == valid_sig

    def test_does_not_add_sentinel_for_non_gemini_3_models(self) -> None:
        model = Model(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            api="google-generative-ai",  # type: ignore[assignment]
            provider="google",
            base_url="https://generativelanguage.googleapis.com",
            reasoning=True,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=128000,
            max_tokens=8192,
        )
        now = 1000
        context = Context(
            messages=[
                UserMessage(role="user", content="Hi", timestamp=now),
                AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCall(
                            type="toolCall",
                            id="call_1",
                            name="bash",
                            arguments={"command": "ls"},
                        ),
                    ],
                    api="google-gemini-cli",
                    provider="google-antigravity",
                    model="claude-sonnet-4-20250514",
                    usage=_empty_usage(),
                    stop_reason="stop",
                    timestamp=now,
                ),
            ],
        )

        contents = convert_messages(model, context)
        model_turn: dict[str, Any] = next(c for c in contents if c.get("role") == "model")
        fc_part: dict[str, Any] | None = next(
            (p for p in model_turn["parts"] if p.get("functionCall") is not None), None
        )

        assert fc_part is not None
        # No sentinel, no thoughtSignature at all
        assert fc_part.get("thoughtSignature") is None
