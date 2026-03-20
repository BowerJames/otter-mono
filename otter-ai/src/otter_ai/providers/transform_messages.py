"""Message transformation for cross-provider compatibility.

Handles:
- Thinking block sanitization (redacted blocks, empty blocks, signatures)
- Tool call ID normalization for API compatibility
- Synthetic tool results for orphaned tool calls
- Skipping errored/aborted assistant messages

Upstream reference: ``packages/ai/src/providers/transform-messages.ts``
"""

from __future__ import annotations

import time
from collections.abc import Callable

from otter_ai.types import (
    AssistantMessage,
    Content,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[Message]:
    """Transform messages for cross-provider compatibility.

    This function performs two passes over the message history:

    **First pass — content transformation:**
    - User messages pass through unchanged.
    - Tool-result messages get their ``toolCallId`` normalized if a mapping exists.
    - Assistant messages have their content blocks transformed:
      - Redacted thinking blocks are dropped for cross-model replay.
      - Empty thinking blocks are dropped; non-empty ones become text for cross-model.
      - Tool-call IDs are normalized and tracked for second-pass updates.
      - ``thoughtSignature`` is stripped for cross-model tool calls.

    **Second pass — orphan repair:**
    - Inserts synthetic error tool results for tool calls that have no
      corresponding ``toolResult`` message, ensuring API requirements are met.
    - Errored/aborted assistant messages are skipped entirely (incomplete turns).

    Parameters
    ----------
    messages:
        The conversation history.
    model:
        The target model (used to detect same-model vs cross-model).
    normalize_tool_call_id:
        Optional callback to normalize tool-call IDs (e.g. truncate
        OpenAI's 450+ char IDs for Anthropic's 64-char limit).
    """
    # Track original → normalized tool-call ID mappings
    tool_call_id_map: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # First pass: transform content within each message
    # ------------------------------------------------------------------ #
    transformed: list[Message] = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            transformed.append(msg)

        elif isinstance(msg, ToolResultMessage):
            normalized_id = tool_call_id_map.get(msg.tool_call_id)
            if normalized_id and normalized_id != msg.tool_call_id:
                transformed.append(_replace_tool_result_id(msg, normalized_id))
            else:
                transformed.append(msg)

        else:
            # Message is AssistantMessage after narrowing
            transformed.append(
                _transform_assistant_message(
                    msg,
                    model,
                    tool_call_id_map,
                    normalize_tool_call_id,
                ),
            )

    # ------------------------------------------------------------------ #
    # Second pass: insert synthetic results for orphaned tool calls
    # ------------------------------------------------------------------ #
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    for msg in transformed:
        if isinstance(msg, AssistantMessage):
            # Flush orphaned tool calls from previous assistant message
            if pending_tool_calls:
                _flush_orphans(pending_tool_calls, existing_tool_result_ids, result)
                pending_tool_calls = []
                existing_tool_result_ids = set()

            # Skip errored/aborted messages — incomplete turns that
            # shouldn't be replayed to avoid API errors.
            if msg.stop_reason in ("error", "aborted"):
                continue

            # Track this assistant message's tool calls
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()

            result.append(msg)

        elif isinstance(msg, ToolResultMessage):
            existing_tool_result_ids.add(msg.tool_call_id)
            result.append(msg)

        else:
            # Message is UserMessage after narrowing
            if pending_tool_calls:
                _flush_orphans(pending_tool_calls, existing_tool_result_ids, result)
                pending_tool_calls = []
                existing_tool_result_ids = set()
            result.append(msg)

    return result


# ============================================================================
# Helpers
# ============================================================================


def _is_same_model(msg: AssistantMessage, model: Model) -> bool:
    return msg.provider == model.provider and msg.api == model.api and msg.model == model.id


def _transform_assistant_message(
    msg: AssistantMessage,
    model: Model,
    tool_call_id_map: dict[str, str],
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None,
) -> AssistantMessage:
    same_model = _is_same_model(msg, model)
    new_content: list[Content] = []

    for block in msg.content:
        if isinstance(block, ThinkingContent):
            new_content.extend(_transform_thinking(block, same_model))
        elif isinstance(block, TextContent):
            new_content.append(_transform_text(block, same_model))
        else:
            # Content is ToolCall after narrowing
            new_tc = _transform_tool_call(
                block,
                same_model,
                model,
                msg,
                tool_call_id_map,
                normalize_tool_call_id,
            )
            new_content.append(new_tc)

    return AssistantMessage(
        role=msg.role,
        content=new_content,
        api=msg.api,
        provider=msg.provider,
        model=msg.model,
        response_id=msg.response_id,
        usage=msg.usage,
        stop_reason=msg.stop_reason,
        error_message=msg.error_message,
        timestamp=msg.timestamp,
    )


def _transform_thinking(
    block: ThinkingContent,
    same_model: bool,
) -> list[Content]:
    # Redacted thinking is opaque encrypted content — only valid for same model
    if block.redacted:
        return [block] if same_model else []

    # Same model: keep thinking blocks with signatures (needed for replay)
    if same_model and block.thinking_signature:
        return [block]

    # Skip empty thinking blocks
    if not block.thinking or not block.thinking.strip():
        return []

    # Same model: keep as-is
    if same_model:
        return [block]

    # Cross-model: convert to plain text
    return [TextContent(type="text", text=block.thinking)]


def _transform_text(block: TextContent, same_model: bool) -> TextContent:
    if same_model:
        return block
    return TextContent(type="text", text=block.text)


def _transform_tool_call(
    block: ToolCall,
    same_model: bool,
    model: Model,
    source_msg: AssistantMessage,
    tool_call_id_map: dict[str, str],
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None,
) -> ToolCall:
    result = block

    # Strip thoughtSignature for cross-model
    if not same_model and block.thought_signature:
        result = ToolCall(
            type=result.type,
            id=result.id,
            name=result.name,
            arguments=result.arguments,
        )

    # Normalize tool-call ID for cross-model
    if not same_model and normalize_tool_call_id:
        normalized_id = normalize_tool_call_id(block.id, model, source_msg)
        if normalized_id != block.id:
            tool_call_id_map[block.id] = normalized_id
            result = ToolCall(
                type=result.type,
                id=normalized_id,
                name=result.name,
                arguments=result.arguments,
                thought_signature=result.thought_signature,
            )

    return result


def _replace_tool_result_id(msg: ToolResultMessage, new_id: str) -> ToolResultMessage:
    return ToolResultMessage(
        role=msg.role,
        tool_call_id=new_id,
        tool_name=msg.tool_name,
        content=msg.content,
        details=msg.details,
        is_error=msg.is_error,
        timestamp=msg.timestamp,
    )


def _flush_orphans(
    pending: list[ToolCall],
    existing_ids: set[str],
    result: list[Message],
) -> None:
    """Insert synthetic error tool results for orphaned tool calls."""
    for tc in pending:
        if tc.id not in existing_ids:
            result.append(
                ToolResultMessage(
                    role="toolResult",
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=[TextContent(type="text", text="No result provided")],
                    is_error=True,
                    timestamp=int(time.time() * 1000),
                ),
            )
