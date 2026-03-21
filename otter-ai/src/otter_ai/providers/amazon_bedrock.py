"""Amazon Bedrock provider using the ConverseStream API.

Streams responses from Bedrock-hosted models (Claude, Mistral, etc.)
using ``boto3`` / ``aioboto3`` with the ``ConverseStream`` operation.

Upstream reference: ``packages/ai/src/providers/amazon-bedrock.ts``
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    pass  # type: ignore[import-untyped]

from otter_ai.models import calculate_cost
from otter_ai.types import (
    AssistantMessage,
    AssistantMessageEventDone,
    AssistantMessageEventError,
    AssistantMessageEventStart,
    AssistantMessageEventTextDelta,
    AssistantMessageEventTextEnd,
    AssistantMessageEventTextStart,
    AssistantMessageEventThinkingDelta,
    AssistantMessageEventThinkingEnd,
    AssistantMessageEventThinkingStart,
    AssistantMessageEventToolcallDelta,
    AssistantMessageEventToolcallEnd,
    AssistantMessageEventToolcallStart,
    CacheRetention,
    Context,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream
from otter_ai.utils.json_parse import parse_streaming_json
from otter_ai.utils.sanitize_unicode import sanitize_surrogates

from .simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_reasoning,
)
from .transform_messages import transform_messages

# ============================================================================
# Bedrock-specific options
# ============================================================================


@dataclass
class BedrockOptions(StreamOptions):
    """Options for the Amazon Bedrock provider.

    Extends :class:`StreamOptions` with Bedrock-specific settings.
    """

    region: str | None = None
    profile: str | None = None
    tool_choice: Literal["auto", "any", "none"] | dict[str, Any] | None = None
    reasoning: ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = None
    interleaved_thinking: bool | None = None


# ============================================================================
# Internal block type (carries transient index/partialJson fields)
# ============================================================================


@dataclass
class _Block:
    """Internal block with transient fields for streaming state."""

    type: str
    index: int | None = None
    partial_json: str = ""
    # TextContent fields
    text: str = ""
    text_signature: str | None = None
    # ThinkingContent fields
    thinking: str = ""
    thinking_signature: str | None = None
    # ToolCall fields
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]
    thought_signature: str | None = None

    def to_text_content(self) -> TextContent:
        return TextContent(type="text", text=self.text, text_signature=self.text_signature)

    def to_thinking_content(self) -> ThinkingContent:
        return ThinkingContent(
            type="thinking",
            thinking=self.thinking,
            thinking_signature=self.thinking_signature,
        )

    def to_tool_call(self) -> ToolCall:
        return ToolCall(
            type="toolCall",
            id=self.id,
            name=self.name,
            arguments=self.arguments,
            thought_signature=self.thought_signature,
        )


# ============================================================================
# Bedrock stop reason mapping
# ============================================================================

_BEDROCK_STOP_TO_REASON: dict[str, StopReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "toolUse",
}


def _map_stop_reason(reason: str | None) -> StopReason:
    if reason is None:
        return "error"
    return _BEDROCK_STOP_TO_REASON.get(reason, "error")


# ============================================================================
# Cache retention
# ============================================================================


def _resolve_cache_retention(cache_retention: CacheRetention | None) -> CacheRetention:
    """Resolve cache retention preference.

    Defaults to ``"short"`` and uses ``PI_CACHE_RETENTION`` env var for
    backward compatibility.
    """
    if cache_retention is not None:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


# ============================================================================
# Model capability checks
# ============================================================================


def _supports_prompt_caching(model: Model) -> bool:
    """Check if the model supports prompt caching.

    Supported: Claude 3.5 Haiku, Claude 3.7 Sonnet, Claude 4.x models.
    Application inference profiles (whose ARNs don't contain the model
    name) can force caching via ``AWS_BEDROCK_FORCE_CACHE=1``.
    """
    model_id = model.id.lower()
    if "claude" not in model_id:
        return os.environ.get("AWS_BEDROCK_FORCE_CACHE") == "1"
    if "-4-" in model_id or "-4." in model_id:
        return True
    if "claude-3-7-sonnet" in model_id:
        return True
    return "claude-3-5-haiku" in model_id


def _supports_thinking_signature(model: Model) -> bool:
    """Check if the model supports thinking signatures in reasoningContent.

    Only Anthropic Claude models support the signature field.  Other models
    reject it with: ``"This model doesn't support the
    reasoningContent.reasoningText.signature field"``.
    """
    model_id = model.id.lower()
    return "anthropic.claude" in model_id or "anthropic/claude" in model_id


def _supports_adaptive_thinking(model_id: str) -> bool:
    """Check if the model supports adaptive thinking (Opus 4.6, Sonnet 4.6)."""
    return (
        "opus-4-6" in model_id
        or "opus-4.6" in model_id
        or "sonnet-4-6" in model_id
        or "sonnet-4.6" in model_id
    )


# ============================================================================
# Thinking level mapping
# ============================================================================


def _map_thinking_level_to_effort(
    level: ThinkingLevel,
    model_id: str,
) -> str:
    """Map a thinking level to a Bedrock effort string."""
    match level:
        case "minimal" | "low":
            return "low"
        case "medium":
            return "medium"
        case "high":
            return "high"
        case "xhigh":
            return "max" if _supports_adaptive_thinking(model_id) else "high"
        case _:
            return "high"


# ============================================================================
# Tool call ID normalization
# ============================================================================


def _normalize_tool_call_id(tool_call_id: str) -> str:
    """Sanitize a tool-call ID for Bedrock's 64-char limit."""
    sanitized = "".join(c if c.isalnum() or c in "_-" else "_" for c in tool_call_id)
    return sanitized[:64]


# ============================================================================
# Image conversion
# ============================================================================


def _create_image_block(mime_type: str, data: str) -> dict[str, Any]:
    """Convert a base64 image to a Bedrock image block."""
    # Map MIME types to Bedrock format names
    format_map: dict[str, str] = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    fmt = format_map.get(mime_type)
    if fmt is None:
        raise RuntimeError(f"Unknown image type: {mime_type}")

    binary_data = base64.b64decode(data)
    return {"image": {"format": fmt, "source": {"bytes": binary_data}}}


# ============================================================================
# Message conversion
# ============================================================================


def _build_system_prompt(
    system_prompt: str | None,
    model: Model,
    cache_retention: CacheRetention,
) -> list[dict[str, Any]] | None:
    """Build the Bedrock system prompt blocks."""
    if not system_prompt:
        return None

    blocks: list[dict[str, Any]] = [{"text": sanitize_surrogates(system_prompt)}]

    if cache_retention != "none" and _supports_prompt_caching(model):
        cache_point: dict[str, Any] = {"cachePointType": "DEFAULT"}
        if cache_retention == "long":
            cache_point["ttlInSeconds"] = 3600
        blocks.append(cache_point)

    return blocks


def _convert_messages(
    context: Context,
    model: Model,
    cache_retention: CacheRetention,
) -> list[dict[str, Any]]:
    """Convert otter messages to Bedrock Converse API format.

    Handles:
    - User messages (text + images)
    - Assistant messages (text, thinking, tool calls)
    - Consecutive tool-result messages merged into a single user message
    - Cache points on the last user message
    - Skipping empty/errored assistant messages
    """
    result: list[dict[str, Any]] = []
    transformed = transform_messages(
        context.messages,
        model,
        lambda tool_id, _model, _msg: _normalize_tool_call_id(tool_id),
    )

    i = 0
    while i < len(transformed):
        msg = transformed[i]

        match msg.role:
            case "user":
                content = (
                    [{"text": sanitize_surrogates(msg.content)}]
                    if isinstance(msg.content, str)
                    else [
                        (
                            {"text": sanitize_surrogates(c.text)}
                            if c.type == "text"
                            else _create_image_block(c.mime_type, c.data)
                        )
                        for c in msg.content
                    ]
                )
                result.append({"role": "user", "content": content})

            case "assistant":
                # Skip empty content (e.g., from aborted requests)
                if len(msg.content) == 0:
                    i += 1
                    continue

                content_blocks: list[dict[str, Any]] = []
                for c in msg.content:
                    match c.type:
                        case "text":
                            if not c.text.strip():
                                continue
                            content_blocks.append({"text": sanitize_surrogates(c.text)})
                        case "toolCall":
                            content_blocks.append(
                                {
                                    "toolUse": {
                                        "toolUseId": c.id,
                                        "name": c.name,
                                        "input": c.arguments,
                                    },
                                },
                            )
                        case "thinking":
                            if not c.thinking.strip():
                                continue
                            if _supports_thinking_signature(model):
                                if not c.thinking_signature or not c.thinking_signature.strip():
                                    content_blocks.append(
                                        {"text": sanitize_surrogates(c.thinking)},
                                    )
                                else:
                                    content_blocks.append(
                                        {
                                            "reasoningContent": {
                                                "reasoningText": {
                                                    "text": sanitize_surrogates(c.thinking),
                                                    "signature": c.thinking_signature,
                                                },
                                            },
                                        },
                                    )
                            else:
                                content_blocks.append(
                                    {
                                        "reasoningContent": {
                                            "reasoningText": {
                                                "text": sanitize_surrogates(c.thinking),
                                            },
                                        },
                                    },
                                )
                        case _:
                            raise RuntimeError(f"Unknown assistant content type: {c.type}")

                if not content_blocks:
                    i += 1
                    continue

                result.append({"role": "assistant", "content": content_blocks})

            case "toolResult":
                # Collect consecutive tool-result messages into one user message
                tool_results: list[dict[str, Any]] = [
                    {
                        "toolResult": {
                            "toolUseId": msg.tool_call_id,
                            "content": [
                                (
                                    _create_image_block(c.mime_type, c.data)
                                    if c.type == "image"
                                    else {"text": sanitize_surrogates(c.text)}
                                )
                                for c in msg.content
                            ],
                            "status": "error" if msg.is_error else "success",
                        },
                    },
                ]

                j = i + 1
                while j < len(transformed) and transformed[j].role == "toolResult":
                    next_msg = transformed[j]
                    assert isinstance(next_msg, ToolResultMessage)
                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": next_msg.tool_call_id,
                                "content": [
                                    (
                                        _create_image_block(c.mime_type, c.data)
                                        if c.type == "image"
                                        else {"text": sanitize_surrogates(c.text)}
                                    )
                                    for c in next_msg.content
                                ],
                                "status": "error" if next_msg.is_error else "success",
                            },
                        },
                    )
                    j += 1

                i = j - 1
                result.append({"role": "user", "content": tool_results})

            case _:
                raise RuntimeError(f"Unknown message role: {msg.role}")

        i += 1

    # Add cache point to last user message
    if cache_retention != "none" and _supports_prompt_caching(model) and result:
        last = result[-1]
        if last["role"] == "user" and "content" in last:
            cache_point: dict[str, Any] = {"cachePointType": "DEFAULT"}
            if cache_retention == "long":
                cache_point["ttlInSeconds"] = 3600
            last["content"].append(cache_point)

    return result


# ============================================================================
# Tool configuration conversion
# ============================================================================


def _convert_tool_config(
    tools: list[Tool] | None,
    tool_choice: Any,
) -> dict[str, Any] | None:
    """Convert otter tools to Bedrock ToolConfiguration."""
    if not tools or tool_choice == "none":
        return None

    bedrock_tools = [
        {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.parameters.model_json_schema()},
            },
        }
        for tool in tools
    ]

    bedrock_tool_choice: dict[str, Any] | None = None
    match tool_choice:
        case "auto":
            bedrock_tool_choice = {"auto": {}}
        case "any":
            bedrock_tool_choice = {"any": {}}
        case _:
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":  # type: ignore[unknownMemberType]
                bedrock_tool_choice = {"tool": {"name": tool_choice["name"]}}

    return {"tools": bedrock_tools, "toolChoice": bedrock_tool_choice}


# ============================================================================
# Additional model request fields (thinking/reasoning)
# ============================================================================


def _build_additional_model_request_fields(
    model: Model,
    options: BedrockOptions,
) -> dict[str, Any] | None:
    """Build additionalModelRequestFields for thinking/reasoning support."""
    if not options.reasoning or not model.reasoning:
        return None

    model_id = model.id
    is_anthropic_claude = "anthropic.claude" in model_id or "anthropic/claude" in model_id

    if not is_anthropic_claude:
        return None

    if _supports_adaptive_thinking(model_id):
        result: dict[str, Any] = {
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": _map_thinking_level_to_effort(options.reasoning, model_id),
            },
        }
    else:
        default_budgets: dict[str, int] = {
            "minimal": 1024,
            "low": 2048,
            "medium": 8192,
            "high": 16384,
            "xhigh": 16384,
        }

        level = "high" if options.reasoning == "xhigh" else options.reasoning
        budget = (
            getattr(options.thinking_budgets, level, None)
            if options.thinking_budgets
            else None
        ) or default_budgets.get(options.reasoning, 16384)

        result = {
            "thinking": {
                "type": "enabled",
                "budgetTokens": budget,
            },
        }

    if (
        not _supports_adaptive_thinking(model_id)
        and (options.interleaved_thinking is None or options.interleaved_thinking)
    ):
        result["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]

    return result


# ============================================================================
# Client configuration
# ============================================================================


def _build_client_config(options: BedrockOptions) -> dict[str, Any]:
    """Build boto3 client configuration from options and environment."""
    config: dict[str, Any] = {}

    if options.profile:
        config["profile_name"] = options.profile

    # Region resolution: explicit option > env vars > default us-east-1
    explicit_region = (
        options.region
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if explicit_region:
        config["region_name"] = explicit_region
    elif not os.environ.get("AWS_PROFILE"):
        config["region_name"] = "us-east-1"

    # Support proxies / skip auth
    if os.environ.get("AWS_BEDROCK_SKIP_AUTH") == "1":
        config["aws_access_key_id"] = "dummy-access-key"
        config["aws_secret_access_key"] = "dummy-secret-key"

    return config


# ============================================================================
# Event handlers
# ============================================================================


def _handle_content_block_start(
    event: dict[str, Any],
    blocks: list[_Block],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> None:
    """Handle a contentBlockStart event."""
    content_block_index = event.get("contentBlockIndex", 0)
    start = event.get("start", {})

    if "toolUse" in start:
        block = _Block(
            type="toolCall",
            id=start["toolUse"].get("toolUseId", ""),
            name=start["toolUse"].get("name", ""),
            arguments={},
            partial_json="",
            index=content_block_index,
        )
        blocks.append(block)
        stream.push(
            AssistantMessageEventToolcallStart(
                type="toolcall_start",
                content_index=len(blocks) - 1,
                partial=output,
            ),
        )


def _handle_content_block_delta(
    event: dict[str, Any],
    blocks: list[_Block],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> None:
    """Handle a contentBlockDelta event."""
    content_block_index = event.get("contentBlockIndex", 0)
    delta = event.get("delta", {})

    # Find existing block by index
    block = next((b for b in blocks if b.index == content_block_index), None)
    index = next((i for i, b in enumerate(blocks) if b.index == content_block_index), -1)

    # Text delta
    if "text" in delta:
        if block is None:
            block = _Block(type="text", text="", index=content_block_index)
            blocks.append(block)
            index = len(blocks) - 1
            stream.push(
                AssistantMessageEventTextStart(
                    type="text_start",
                    content_index=index,
                    partial=output,
                ),
            )
        if block.type == "text":
            text = delta["text"]
            block.text += text
            stream.push(
                AssistantMessageEventTextDelta(
                    type="text_delta",
                    content_index=index,
                    delta=text,
                    partial=output,
                ),
            )

    # Tool use delta
    elif "toolUse" in delta and block is not None and block.type == "toolCall":
        input_json = delta["toolUse"].get("input", "")
        block.partial_json += input_json
        block.arguments = cast(dict[str, Any], parse_streaming_json(block.partial_json) or {})
        stream.push(
            AssistantMessageEventToolcallDelta(
                type="toolcall_delta",
                content_index=index,
                delta=input_json,
                partial=output,
            ),
        )

    # Reasoning/thinking delta
    elif "reasoningContent" in delta:
        rc = delta["reasoningContent"]
        thinking_block = block
        thinking_index = index

        if thinking_block is None:
            thinking_block = _Block(
                type="thinking",
                thinking="",
                thinking_signature="",
                index=content_block_index,
            )
            blocks.append(thinking_block)
            thinking_index = len(blocks) - 1
            stream.push(
                AssistantMessageEventThinkingStart(
                    type="thinking_start",
                    content_index=thinking_index,
                    partial=output,
                ),
            )

        if thinking_block.type == "thinking":
            text = rc.get("text")
            if text:
                thinking_block.thinking += text
                stream.push(
                    AssistantMessageEventThinkingDelta(
                        type="thinking_delta",
                        content_index=thinking_index,
                        delta=text,
                        partial=output,
                    ),
                )
            signature = rc.get("signature")
            if signature:
                thinking_block.thinking_signature = (
                    thinking_block.thinking_signature or ""
                ) + signature


def _handle_content_block_stop(
    event: dict[str, Any],
    blocks: list[_Block],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> None:
    """Handle a contentBlockStop event."""
    content_block_index = event.get("contentBlockIndex", 0)
    index = next((i for i, b in enumerate(blocks) if b.index == content_block_index), -1)
    if index < 0:
        return

    block = blocks[index]
    match block.type:
        case "text":
            stream.push(
                AssistantMessageEventTextEnd(
                    type="text_end",
                    content_index=index,
                    content=block.text,
                    partial=output,
                ),
            )
        case "thinking":
            stream.push(
                AssistantMessageEventThinkingEnd(
                    type="thinking_end",
                    content_index=index,
                    content=block.thinking,
                    partial=output,
                ),
            )
        case "toolCall":
            block.arguments = cast(dict[str, Any], parse_streaming_json(block.partial_json) or {})
            stream.push(
                AssistantMessageEventToolcallEnd(
                    type="toolcall_end",
                    content_index=index,
                    tool_call=block.to_tool_call(),
                    partial=output,
                ),
            )
        case _:  # pragma: no cover
            pass


def _handle_metadata(
    event: dict[str, Any],
    model: Model,
    output: AssistantMessage,
) -> None:
    """Handle a metadata event (usage data)."""
    usage_data = event.get("usage", {})
    if usage_data:
        output.usage.input = usage_data.get("inputTokens", 0)
        output.usage.output = usage_data.get("outputTokens", 0)
        output.usage.cache_read = usage_data.get("cacheReadInputTokens", 0)
        output.usage.cache_write = usage_data.get("cacheWriteInputTokens", 0)
        output.usage.total_tokens = usage_data.get(
            "totalTokens",
            output.usage.input + output.usage.output,
        )
        calculate_cost(model, output.usage)


# ============================================================================
# Stream function
# ============================================================================


def stream_bedrock(
    model: Model,
    context: Context,
    options: BedrockOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an LLM response from Amazon Bedrock.

    Parameters
    ----------
    model:
        The Bedrock model to use.
    context:
        The conversation context.
    options:
        Bedrock-specific stream options.
    """
    stream = AssistantMessageEventStream()
    options = options or BedrockOptions()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api="bedrock-converse-stream",
            provider=model.provider,
            model=model.id,
            usage=Usage(cost=UsageCost()),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        blocks: list[_Block] = []

        try:
            import boto3 as _boto3

            client_config = _build_client_config(options)
            client = cast(Any, _boto3.client("bedrock-runtime", **client_config))  # type: ignore[unknownMemberType]

            cache_retention = _resolve_cache_retention(options.cache_retention)

            command_input: dict[str, Any] = {
                "modelId": model.id,
                "messages": _convert_messages(context, model, cache_retention),
                "system": _build_system_prompt(context.system_prompt, model, cache_retention),
                "inferenceConfig": {
                    "maxTokens": options.max_tokens,
                    "temperature": options.temperature,
                },
                "toolConfig": _convert_tool_config(context.tools, options.tool_choice),
                "additionalModelRequestFields": _build_additional_model_request_fields(
                    model, options,
                ),
            }

            # Allow on_payload to inspect/modify the request
            if options.on_payload is not None:
                next_input = options.on_payload(command_input, model)
                if next_input is not None:
                    command_input = next_input

            # Call ConverseStream
            response = cast(dict[str, Any], client.converse_stream(**command_input))
            event_stream: Any = response.get("stream")

            if event_stream is None:
                raise RuntimeError("Bedrock response has no event stream")

            for raw_event in event_stream:
                event: dict[str, Any] = dict(raw_event)  # type: ignore[unknownVariableType, unknownArgumentType]
                if options.signal is not None and options.signal.is_set():
                    raise RuntimeError("Request was aborted")

                # Determine event type
                if "messageStart" in event:
                    role = event["messageStart"].get("role")
                    if role != "assistant":
                        raise RuntimeError(
                            f"Unexpected assistant message start but got role: {role}",
                        )
                    stream.push(AssistantMessageEventStart(type="start", partial=output))

                elif "contentBlockStart" in event:
                    _handle_content_block_start(event["contentBlockStart"], blocks, output, stream)

                elif "contentBlockDelta" in event:
                    _handle_content_block_delta(event["contentBlockDelta"], blocks, output, stream)

                elif "contentBlockStop" in event:
                    _handle_content_block_stop(event["contentBlockStop"], blocks, output, stream)

                elif "messageStop" in event:
                    output.stop_reason = _map_stop_reason(
                        event["messageStop"].get("stopReason"),
                    )

                elif "metadata" in event:
                    _handle_metadata(event["metadata"], model, output)

                elif "internalServerException" in event:
                    msg = event["internalServerException"].get("message", "")
                    raise RuntimeError(f"Internal server error: {msg}")
                elif "modelStreamErrorException" in event:
                    msg = event["modelStreamErrorException"].get("message", "")
                    raise RuntimeError(f"Model stream error: {msg}")
                elif "validationException" in event:
                    msg = event["validationException"].get("message", "")
                    raise RuntimeError(f"Validation error: {msg}")
                elif "throttlingException" in event:
                    msg = event["throttlingException"].get("message", "")
                    raise RuntimeError(f"Throttling error: {msg}")
                elif "serviceUnavailableException" in event:
                    msg = event["serviceUnavailableException"].get("message", "")
                    raise RuntimeError(f"Service unavailable: {msg}")

            if options.signal is not None and options.signal.is_set():
                raise RuntimeError("Request was aborted")

            if output.stop_reason in ("error", "aborted"):
                raise RuntimeError("An unknown error occurred")

            # Convert internal blocks to proper content types
            output.content = [
                b.to_text_content() if b.type == "text"
                else b.to_thinking_content() if b.type == "thinking"
                else b.to_tool_call()
                for b in blocks
            ]

            stream.push(
                AssistantMessageEventDone(
                    type="done",
                    reason=output.stop_reason,
                    message=output,
                ),
            )
            stream.end()

        except Exception as error:
            # Convert blocks to content (best effort)
            import contextlib

            with contextlib.suppress(Exception):
                output.content = [
                    b.to_text_content() if b.type == "text"
                    else b.to_thinking_content() if b.type == "thinking"
                    else b.to_tool_call()
                    for b in blocks
                ]

            is_aborted = options.signal is not None and options.signal.is_set()
            output.stop_reason = "aborted" if is_aborted else "error"
            output.error_message = str(error)
            stream.push(
                AssistantMessageEventError(
                    type="error",
                    reason=output.stop_reason,
                    error=output,
                ),
            )
            stream.end()

    asyncio.ensure_future(_run())
    return stream


# ============================================================================
# Simple stream function
# ============================================================================


def stream_simple_bedrock(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a Bedrock response using :class:`SimpleStreamOptions`.

    Handles reasoning/thinking level mapping and max-token budget adjustment.
    """
    base = build_base_options(model, options)

    if options is None or not options.reasoning:
        return stream_bedrock(model, context, BedrockOptions(**base.__dict__))

    model_id = model.id
    is_claude = "anthropic.claude" in model_id or "anthropic/claude" in model_id

    if is_claude:
        if _supports_adaptive_thinking(model_id):
            return stream_bedrock(
                model,
                context,
                BedrockOptions(
                    **base.__dict__,
                    reasoning=options.reasoning,
                    thinking_budgets=options.thinking_budgets,
                ),
            )

        adjusted = adjust_max_tokens_for_thinking(
            base.max_tokens or 0,
            model.max_tokens,
            options.reasoning,
            options.thinking_budgets,
        )
        clamped_level = clamp_reasoning(options.reasoning)
        assert clamped_level is not None

        return stream_bedrock(
            model,
            context,
            BedrockOptions(
                **base.__dict__,
                max_tokens=adjusted[0],
                reasoning=options.reasoning,
                thinking_budgets=options.thinking_budgets,
            ),
        )

    return stream_bedrock(
        model,
        context,
        BedrockOptions(
            **base.__dict__,
            reasoning=options.reasoning,
            thinking_budgets=options.thinking_budgets,
        ),
    )
