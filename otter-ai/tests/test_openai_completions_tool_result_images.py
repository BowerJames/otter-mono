"""Tests for openai-completions convertMessages tool-result image batching.

Upstream reference: ``packages/ai/test/openai-completions-tool-result-images.test.ts``
"""

from dataclasses import replace

from otter_ai.models import get_model
from otter_ai.providers.openai_completions import _ResolvedCompat  # type: ignore[import-private]
from otter_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)


def _empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=UsageCost(),
    )


def _default_compat() -> _ResolvedCompat:
    return _ResolvedCompat(
        supports_store=True,
        supports_developer_role=True,
        supports_reasoning_effort=True,
        reasoning_effort_map={},
        supports_usage_in_streaming=True,
        max_tokens_field="max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        thinking_format="openai",
        open_router_routing=None,
        vercel_gateway_routing=None,
        supports_strict_mode=True,
    )


class TestOpenAICompletionsConvertMessages:
    def test_batches_tool_result_images_after_consecutive_tool_results(self) -> None:
        """Tool-result images from consecutive tool calls are batched into a user message."""
        from otter_ai.providers.openai_completions import convert_messages

        base_model = get_model("openai", "gpt-4o-mini")
        model = replace(base_model, api="openai-completions", input=["text", "image"])

        now = 1000
        assistant_message = AssistantMessage(
            role="assistant",
            content=[
                ToolCall(
                    type="toolCall", id="tool-1", name="read", arguments={"path": "img-1.png"}
                ),
                ToolCall(
                    type="toolCall", id="tool-2", name="read", arguments={"path": "img-2.png"}
                ),
            ],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=_empty_usage(),
            stop_reason="toolUse",
            timestamp=now,
        )

        context = Context(
            messages=[
                UserMessage(role="user", content="Read the images", timestamp=now - 2),
                assistant_message,
                ToolResultMessage(
                    role="toolResult",
                    tool_call_id="tool-1",
                    tool_name="read",
                    content=[
                        TextContent(type="text", text="Read image file [image/png]"),
                        ImageContent(type="image", data="ZmFrZQ==", mime_type="image/png"),
                    ],
                    is_error=False,
                    timestamp=now + 1,
                ),
                ToolResultMessage(
                    role="toolResult",
                    tool_call_id="tool-2",
                    tool_name="read",
                    content=[
                        TextContent(type="text", text="Read image file [image/png]"),
                        ImageContent(type="image", data="ZmFrZQ==", mime_type="image/png"),
                    ],
                    is_error=False,
                    timestamp=now + 2,
                ),
            ],
        )

        messages = convert_messages(model, context, _default_compat())
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool", "tool", "user"]

        image_message = messages[-1]
        assert image_message["role"] == "user"
        assert isinstance(image_message["content"], list)

        content = image_message["content"]  # type: ignore[unknownVariableType]
        assert isinstance(content, list)
        image_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]  # type: ignore[unknownVariableType, unknownMemberType]
        assert len(list(image_parts)) == 2  # type: ignore[unknownArgumentType]
