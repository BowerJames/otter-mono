"""Tests for transform-messages OpenAI-to-Anthropic session migration.

Upstream reference: ``packages/ai/test/transform-messages-copilot-openai-to-anthropic.test.ts``
"""

from otter_ai.models import get_model
from otter_ai.providers.transform_messages import transform_messages
from otter_ai.types import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)


def anthropic_normalize_tool_call_id(
    tool_call_id: str, _model: Model, _msg: AssistantMessage
) -> str:
    """Normalize function matching what anthropic.ts uses."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in tool_call_id)[:64]


def make_copilot_claude_model():
    return get_model("anthropic", "claude-sonnet-4-5")


class TestCopilotClaudeMigration:
    def test_converts_thinking_blocks_to_plain_text_when_source_model_differs(self) -> None:
        model = make_copilot_claude_model()
        messages: list[Message] = [
            UserMessage(role="user", content="hello", timestamp=0),
            AssistantMessage(
                role="assistant",
                content=[
                    ThinkingContent(
                        type="thinking",
                        thinking="Let me think about this...",
                        thinking_signature="reasoning_content",
                    ),
                    TextContent(type="text", text="Hi there!"),
                ],
                api="openai-completions",
                provider="github-copilot",
                model="gpt-4o",
                usage=Usage(
                    input=0,
                    output=0,
                    cache_read=0,
                    cache_write=0,
                    total_tokens=0,
                    cost=UsageCost(),
                ),
                stop_reason="stop",
                timestamp=0,
            ),
        ]

        result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
        assistant_msg = next(m for m in result if m.role == "assistant")

        # Thinking block should be converted to text since models differ
        text_blocks = [b for b in assistant_msg.content if b.type == "text"]
        thinking_blocks = [b for b in assistant_msg.content if b.type == "thinking"]
        assert len(thinking_blocks) == 0
        assert len(text_blocks) >= 2

    def test_removes_thought_signature_from_tool_calls_when_migrating(self) -> None:
        model = make_copilot_claude_model()
        messages: list[Message] = [
            UserMessage(role="user", content="run a command", timestamp=0),
            AssistantMessage(
                role="assistant",
                content=[
                    ToolCall(
                        type="toolCall",
                        id="call_123",
                        name="bash",
                        arguments={"command": "ls"},
                        thought_signature=(
                            '{"type": "reasoning.encrypted", "id": "call_123", "data": "encrypted"}'
                        ),
                    ),
                ],
                api="openai-responses",
                provider="github-copilot",
                model="gpt-5",
                usage=Usage(
                    input=0,
                    output=0,
                    cache_read=0,
                    cache_write=0,
                    total_tokens=0,
                    cost=UsageCost(),
                ),
                stop_reason="toolUse",
                timestamp=0,
            ),
            ToolResultMessage(
                role="toolResult",
                tool_call_id="call_123",
                tool_name="bash",
                content=[TextContent(type="text", text="output")],
                is_error=False,
                timestamp=0,
            ),
        ]

        result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
        assistant_msg = next(m for m in result if m.role == "assistant")
        tool_call = next(b for b in assistant_msg.content if b.type == "toolCall")

        assert tool_call.thought_signature is None
