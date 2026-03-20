# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false

"""Anthropic Messages API provider.

Upstream: packages/ai/src/providers/anthropic.ts (~900 lines)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import anthropic

from otter_ai.env_api_keys import get_env_api_key
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
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream
from otter_ai.utils.json_parse import parse_streaming_json
from otter_ai.utils.sanitize_unicode import sanitize_surrogates

from .github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from .simple_options import adjust_max_tokens_for_thinking, build_base_options
from .transform_messages import transform_messages

# ============================================================================
# Cache control
# ============================================================================


def resolve_cache_retention(cache_retention: CacheRetention | None) -> CacheRetention:
    """Resolve cache retention preference.

    Defaults to "short" and uses PI_CACHE_RETENTION to match upstream.
    """
    if cache_retention:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


@dataclass
class _CacheControl:
    """Resolved cache control for Anthropic API."""

    retention: CacheRetention
    cache_control: dict[str, Any] | None = None


def _get_cache_control(
    base_url: str | None,
    cache_retention: CacheRetention | None,
) -> _CacheControl:
    """Get cache control settings for Anthropic requests."""
    retention = resolve_cache_retention(cache_retention)
    if retention == "none":
        return _CacheControl(retention=retention)
    ttl = "1h" if (retention == "long" and base_url and "api.anthropic.com" in base_url) else None
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    return _CacheControl(retention=retention, cache_control=cache_control)


# ============================================================================
# Claude Code stealth mode (OAuth token tool name mapping)
# ============================================================================

_CLAUDE_CODE_VERSION = "2.1.75"

# Claude Code 2.x tool names (canonical casing)
# Source: https://cchistory.mariozechner.at/data/prompts-2.1.11.md
_CLAUDE_CODE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
]

_CC_TOOL_LOOKUP = {t.lower(): t for t in _CLAUDE_CODE_TOOLS}


def _to_claude_code_name(name: str) -> str:
    """Convert tool name to Claude Code canonical casing if it matches."""
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def _from_claude_code_name(name: str, tools: list[Tool] | None) -> str:
    """Convert Claude Code name back to original tool name."""
    if tools:
        lower_name = name.lower()
        for tool in tools:
            if tool.name.lower() == lower_name:
                return tool.name
    return name


# ============================================================================
# Anthropic options
# ============================================================================

AnthropicEffort = Literal["low", "medium", "high", "max"]


@dataclass
class AnthropicOptions(StreamOptions):
    """Options for Anthropic streaming.

    Extends the base stream options with Anthropic-specific features.

    Upstream: ``interface AnthropicOptions extends StreamOptions``
    """

    thinking_enabled: bool = False
    thinking_budget_tokens: int | None = None
    effort: AnthropicEffort | None = None
    interleaved_thinking: bool = True
    tool_choice: str | dict[str, Any] | None = None
    client: anthropic.Anthropic | None = None


# ============================================================================
# Content conversion
# ============================================================================


def _convert_content_blocks(
    content: list[TextContent | ImageContent],
) -> str | list[dict[str, Any]]:
    """Convert content blocks to Anthropic API format."""
    has_images = any(c.type == "image" for c in content)
    if not has_images:
        return sanitize_surrogates("".join((c.text if c.type == "text" else "") for c in content))

    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append(
                {
                    "type": "text",
                    "text": sanitize_surrogates(block.text),
                }
            )
        elif block.type == "image":
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )

    has_text = any(b["type"] == "text" for b in blocks)
    if not has_text:
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})

    return blocks


# ============================================================================
# Adaptive thinking support
# ============================================================================


def _supports_adaptive_thinking(model_id: str) -> bool:
    """Check if a model supports adaptive thinking (Opus 4.6 and Sonnet 4.6)."""
    return (
        "opus-4-6" in model_id
        or "opus-4.6" in model_id
        or "sonnet-4-6" in model_id
        or "sonnet-4.6" in model_id
    )


def _map_thinking_level_to_effort(
    level: str | None,
    model_id: str,
) -> AnthropicEffort:
    """Map ThinkingLevel to Anthropic effort levels for adaptive thinking."""
    match level:
        case "minimal" | "low":
            return "low"
        case "medium":
            return "medium"
        case "high":
            return "high"
        case "xhigh":
            return "max" if ("opus-4-6" in model_id or "opus-4.6" in model_id) else "high"
        case _:
            return "high"


# ============================================================================
# Client creation
# ============================================================================


def _is_oauth_token(api_key: str) -> bool:
    """Check if an API key is an OAuth token."""
    return "sk-ant-oat" in api_key


def _merge_headers(*header_sources: dict[str, str] | None) -> dict[str, str]:
    """Merge multiple header dicts into one."""
    merged: dict[str, str] = {}
    for headers in header_sources:
        if headers:
            merged.update(headers)
    return merged


def _create_client(
    model: Model,
    api_key: str,
    interleaved_thinking: bool,
    options_headers: dict[str, str] | None = None,
    dynamic_headers: dict[str, str] | None = None,
) -> tuple[anthropic.Anthropic, bool]:
    """Create an Anthropic client appropriate for the model/provider.

    Returns (client, is_oauth_token).
    """
    # Adaptive thinking models have interleaved thinking built-in.
    needs_interleaved_beta = interleaved_thinking and not _supports_adaptive_thinking(model.id)

    # Copilot: Bearer auth, selective betas
    if model.provider == "github-copilot":
        beta_features: list[str] = []
        if needs_interleaved_beta:
            beta_features.append("interleaved-thinking-2025-05-14")

        client = anthropic.Anthropic(
            api_key=None,  # type: ignore[arg-type]
            auth_token=api_key,
            base_url=model.base_url,
            default_headers=_merge_headers(
                {
                    "accept": "application/json",
                    "anthropic-dangerous-direct-browser-access": "true",
                    **({"anthropic-beta": ",".join(beta_features)} if beta_features else {}),
                },
                model.headers,
                dynamic_headers,
                options_headers,
            ),
        )
        return client, False

    beta_features = ["fine-grained-tool-streaming-2025-05-14"]
    if needs_interleaved_beta:
        beta_features.append("interleaved-thinking-2025-05-14")

    # OAuth: Bearer auth, Claude Code identity headers
    if _is_oauth_token(api_key):
        client = anthropic.Anthropic(
            api_key=None,  # type: ignore[arg-type]
            auth_token=api_key,
            base_url=model.base_url,
            default_headers=_merge_headers(
                {
                    "accept": "application/json",
                    "anthropic-dangerous-direct-browser-access": "true",
                    "anthropic-beta": (
                        f"claude-code-20250219,oauth-2025-04-20,{','.join(beta_features)}"
                    ),
                    "user-agent": f"claude-cli/{_CLAUDE_CODE_VERSION}",
                    "x-app": "cli",
                },
                model.headers,
                options_headers,
            ),
        )
        return client, True

    # API key auth
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=_merge_headers(
            {
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                "anthropic-beta": ",".join(beta_features),
            },
            model.headers,
            options_headers,
        ),
    )
    return client, False


# ============================================================================
# Message/Tool conversion
# ============================================================================


def _normalize_tool_call_id(tool_call_id: str) -> str:
    """Normalize tool call IDs to match Anthropic's required pattern."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]


def convert_messages(
    messages: list[Message],
    model: Model,
    is_oauth_token: bool,
    cache_control: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert otter messages to Anthropic API message format.

    Upstream: anthropic.ts → convertMessages()
    """
    params: list[dict[str, Any]] = []

    # Transform messages for cross-provider compatibility
    transformed_messages = transform_messages(messages, model, _normalize_tool_call_id)

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]

        if msg.role == "user":
            if isinstance(msg.content, str):
                if msg.content.strip():
                    params.append(
                        {
                            "role": "user",
                            "content": sanitize_surrogates(msg.content),
                        }
                    )
            else:
                blocks: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        blocks.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(item.text),
                            }
                        )
                    elif item.type == "image":
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.mime_type,
                                    "data": item.data,
                                },
                            }
                        )

                # Filter images if model doesn't support them
                if "image" not in (model.input or []):
                    blocks = [b for b in blocks if b.get("type") != "image"]

                # Filter empty text blocks
                blocks = [
                    b
                    for b in blocks
                    if not (b.get("type") == "text" and not b.get("text", "").strip())
                ]

                if not blocks:
                    i += 1
                    continue
                params.append({"role": "user", "content": blocks})

        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []

            for block in msg.content:
                if block.type == "text":
                    if not block.text.strip():
                        continue
                    blocks.append(
                        {
                            "type": "text",
                            "text": sanitize_surrogates(block.text),
                        }
                    )
                elif block.type == "thinking":
                    # Redacted thinking: pass the opaque payload back
                    if getattr(block, "redacted", False):
                        blocks.append(
                            {
                                "type": "redacted_thinking",
                                "data": block.thinking_signature,
                            }
                        )
                        continue
                    if not block.thinking.strip():
                        continue
                    # If signature missing (aborted stream), convert to text
                    if not block.thinking_signature or not block.thinking_signature.strip():
                        blocks.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(block.thinking),
                            }
                        )
                    else:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(block.thinking),
                                "signature": block.thinking_signature,
                            }
                        )
                elif block.type == "toolCall":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": (
                                _to_claude_code_name(block.name) if is_oauth_token else block.name
                            ),
                            "input": block.arguments or {},
                        }
                    )

            if not blocks:
                i += 1
                continue
            params.append({"role": "assistant", "content": blocks})

        elif msg.role == "toolResult":
            # Collect all consecutive toolResult messages
            tool_results: list[dict[str, Any]] = []

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": _convert_content_blocks(msg.content),
                    "is_error": msg.is_error,
                }
            )

            # Look ahead for consecutive toolResult messages
            j = i + 1
            while j < len(transformed_messages) and transformed_messages[j].role == "toolResult":
                next_msg = transformed_messages[j]
                assert isinstance(next_msg, ToolResultMessage)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": next_msg.tool_call_id,
                        "content": _convert_content_blocks(next_msg.content),
                        "is_error": next_msg.is_error,
                    }
                )
                j += 1

            i = j - 1  # Will be incremented at end of loop

            params.append({"role": "user", "content": tool_results})

        i += 1

    # Add cache_control to the last user message
    if cache_control and params:
        last_msg = params[-1]
        if last_msg.get("role") == "user":
            content = last_msg.get("content")
            if isinstance(content, list):
                last_block = content[-1] if content else None
                if last_block and last_block.get("type") in ("text", "image", "tool_result"):
                    last_block["cache_control"] = cache_control
            elif isinstance(content, str):
                last_msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": cache_control,
                    },
                ]

    return params


def _convert_tools(tools: list[Tool], is_oauth_token: bool) -> list[dict[str, Any]]:
    """Convert otter tools to Anthropic API tool format."""
    if not tools:
        return []

    result: list[dict[str, Any]] = []
    for tool in tools:
        json_schema = tool.parameters if isinstance(tool.parameters, dict) else {}
        result.append(
            {
                "name": _to_claude_code_name(tool.name) if is_oauth_token else tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", []),
                },
            }
        )
    return result


def _map_stop_reason(reason: str) -> StopReason:
    """Map Anthropic stop reason to otter StopReason."""
    match reason:
        case "end_turn":
            return "stop"
        case "max_tokens":
            return "length"
        case "tool_use":
            return "toolUse"
        case "refusal":
            return "error"
        case "pause_turn":
            return "stop"
        case "stop_sequence":
            return "stop"
        case "sensitive":
            return "error"
        case _:
            raise RuntimeError(f"Unhandled Anthropic stop reason: {reason}")


# ============================================================================
# Build API params
# ============================================================================


def _build_params(
    model: Model,
    context: Context,
    is_oauth_token: bool,
    options: AnthropicOptions | None = None,
) -> dict[str, Any]:
    """Build Anthropic API request parameters."""
    cache_ctrl = _get_cache_control(model.base_url, options.cache_retention if options else None)
    cache_control = cache_ctrl.cache_control

    params: dict[str, Any] = {
        "model": model.id,
        "messages": convert_messages(context.messages, model, is_oauth_token, cache_control),
        "max_tokens": (
            options.max_tokens
            if options and options.max_tokens is not None
            else (model.max_tokens // 3)
        ),
        "stream": True,
    }

    # System prompt with cache control
    if is_oauth_token:
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                **({"cache_control": cache_control} if cache_control else {}),
            },
        ]
        if context.system_prompt:
            system_blocks.append(
                {
                    "type": "text",
                    "text": sanitize_surrogates(context.system_prompt),
                    **({"cache_control": cache_control} if cache_control else {}),
                }
            )
        params["system"] = system_blocks
    elif context.system_prompt:
        params["system"] = [
            {
                "type": "text",
                "text": sanitize_surrogates(context.system_prompt),
                **({"cache_control": cache_control} if cache_control else {}),
            },
        ]

    # Temperature (incompatible with thinking)
    if options and options.temperature is not None and not options.thinking_enabled:
        params["temperature"] = options.temperature

    # Tools
    if context.tools:
        params["tools"] = _convert_tools(context.tools, is_oauth_token)

    # Thinking mode
    if options and options.thinking_enabled and model.reasoning:
        if _supports_adaptive_thinking(model.id):
            # Adaptive thinking: Claude decides when/how much to think
            params["thinking"] = {"type": "adaptive"}
            if options.effort:
                params["output_config"] = {"effort": options.effort}
        else:
            # Budget-based thinking for older models
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": options.thinking_budget_tokens or 1024,
            }

    # Metadata
    if options and options.metadata:
        user_id = options.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    # Tool choice
    if options and options.tool_choice:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    return params


# ============================================================================
# Streaming
# ============================================================================


def stream_anthropic(
    model: Model,
    context: Context,
    options: AnthropicOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream completions from the Anthropic Messages API.

    Upstream: anthropic.ts → streamAnthropic()
    """
    stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api="anthropic-messages",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=0,
        )

        try:
            # Create client
            if options and options.client:
                client = options.client
                is_oauth = False
            else:
                api_key = (
                    (options.api_key if options else None) or get_env_api_key(model.provider) or ""
                )

                copilot_dynamic_headers: dict[str, str] | None = None
                if model.provider == "github-copilot":
                    has_images = has_copilot_vision_input(context.messages)
                    copilot_dynamic_headers = build_copilot_dynamic_headers(
                        messages=context.messages,
                        has_images=has_images,
                    )

                client, is_oauth = _create_client(
                    model,
                    api_key,
                    (options.interleaved_thinking if options else True),
                    (options.headers if options else None),
                    copilot_dynamic_headers,
                )

            params = _build_params(model, context, is_oauth, options)

            # Allow payload inspection/replacement
            if options and options.on_payload:
                next_params = await options.on_payload(params, model)
                if next_params is not None:
                    params = next_params

            # Create streaming request
            import time

            output.timestamp = int(time.time() * 1000)

            with client.messages.stream(
                params, signal=options.signal if options else None
            ) as anthropic_stream:
                stream.push(AssistantMessageEventStart(partial=output))

                # Track content blocks with their Anthropic indices
                block_indices: dict[int, int] = {}  # anthropic index -> output index
                partial_jsons: dict[int, str] = {}  # anthropic index -> accumulated JSON

                for event in anthropic_stream:
                    event_type = event.type

                    if event_type == "message_start":
                        output.response_id = event.message.id
                        output.usage.input = event.message.usage.input_tokens or 0
                        output.usage.output = event.message.usage.output_tokens or 0
                        output.usage.cache_read = (
                            getattr(event.message.usage, "cache_read_input_tokens", 0) or 0
                        )
                        output.usage.cache_write = (
                            getattr(event.message.usage, "cache_creation_input_tokens", 0) or 0
                        )
                        output.usage.total_tokens = (
                            output.usage.input
                            + output.usage.output
                            + output.usage.cache_read
                            + output.usage.cache_write
                        )
                        calculate_cost(model, output.usage)

                    elif event_type == "content_block_start":
                        cb = event.content_block
                        if cb.type == "text":
                            block_indices[event.index] = len(output.content)
                            output.content.append(TextContent(type="text", text=""))
                            stream.push(
                                AssistantMessageEventTextStart(
                                    content_index=len(output.content) - 1,
                                    partial=output,
                                )
                            )
                        elif cb.type == "thinking":
                            block_indices[event.index] = len(output.content)
                            output.content.append(
                                ThinkingContent(
                                    type="thinking",
                                    thinking="",
                                    thinking_signature="",
                                )
                            )
                            stream.push(
                                AssistantMessageEventThinkingStart(
                                    content_index=len(output.content) - 1,
                                    partial=output,
                                )
                            )
                        elif cb.type == "redacted_thinking":
                            block_indices[event.index] = len(output.content)
                            output.content.append(
                                ThinkingContent(
                                    type="thinking",
                                    thinking="[Reasoning redacted]",
                                    thinking_signature=cb.data,
                                    redacted=True,
                                )
                            )
                            stream.push(
                                AssistantMessageEventThinkingStart(
                                    content_index=len(output.content) - 1,
                                    partial=output,
                                )
                            )
                        elif cb.type == "tool_use":
                            block_indices[event.index] = len(output.content)
                            tool_name = (
                                _from_claude_code_name(cb.name, context.tools)
                                if is_oauth
                                else cb.name
                            )
                            output.content.append(
                                ToolCall(
                                    type="toolCall",
                                    id=cb.id,
                                    name=tool_name,
                                    arguments=(cb.input if isinstance(cb.input, dict) else {})
                                    or {},
                                )
                            )
                            partial_jsons[event.index] = ""
                            stream.push(
                                AssistantMessageEventToolcallStart(
                                    content_index=len(output.content) - 1,
                                    partial=output,
                                )
                            )

                    elif event_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            idx = block_indices.get(event.index)
                            if idx is not None:
                                block = output.content[idx]
                                if isinstance(block, TextContent):
                                    block.text += delta.text
                                    stream.push(
                                        AssistantMessageEventTextDelta(
                                            content_index=idx,
                                            delta=delta.text,
                                            partial=output,
                                        )
                                    )
                        elif delta.type == "thinking_delta":
                            idx = block_indices.get(event.index)
                            if idx is not None:
                                block = output.content[idx]
                                if isinstance(block, ThinkingContent):
                                    block.thinking += delta.thinking
                                    stream.push(
                                        AssistantMessageEventThinkingDelta(
                                            content_index=idx,
                                            delta=delta.thinking,
                                            partial=output,
                                        )
                                    )
                        elif delta.type == "input_json_delta":
                            idx = block_indices.get(event.index)
                            if idx is not None:
                                block = output.content[idx]
                                if block.type == "toolCall":
                                    partial_jsons[event.index] += delta.partial_json
                                    block.arguments = (
                                        parse_streaming_json(partial_jsons[event.index]) or {}
                                    )
                                    stream.push(
                                        AssistantMessageEventToolcallDelta(
                                            content_index=idx,
                                            delta=delta.partial_json,
                                            partial=output,
                                        )
                                    )
                        elif delta.type == "signature_delta":
                            idx = block_indices.get(event.index)
                            if idx is not None:
                                block = output.content[idx]
                                if isinstance(block, ThinkingContent):
                                    sig = block.thinking_signature or ""
                                    sig += delta.signature
                                    block.thinking_signature = sig

                    elif event_type == "content_block_stop":
                        idx = block_indices.get(event.index)
                        if idx is not None:
                            block = output.content[idx]
                            if isinstance(block, TextContent):
                                stream.push(
                                    AssistantMessageEventTextEnd(
                                        content_index=idx,
                                        content=block.text,
                                        partial=output,
                                    )
                                )
                            elif isinstance(block, ThinkingContent):
                                stream.push(
                                    AssistantMessageEventThinkingEnd(
                                        content_index=idx,
                                        content=block.thinking,
                                        partial=output,
                                    )
                                )
                            elif block.type == "toolCall":
                                block.arguments = (
                                    parse_streaming_json(partial_jsons.get(event.index, "")) or {}
                                )
                                stream.push(
                                    AssistantMessageEventToolcallEnd(
                                        content_index=idx,
                                        tool_call=block,
                                        partial=output,
                                    )
                                )

                    elif event_type == "message_delta":
                        if event.delta.stop_reason:
                            output.stop_reason = _map_stop_reason(event.delta.stop_reason)
                        # Update usage fields if present
                        if event.usage.input_tokens is not None:
                            output.usage.input = event.usage.input_tokens
                        if event.usage.output_tokens is not None:
                            output.usage.output = event.usage.output_tokens
                        cache_read = getattr(event.usage, "cache_read_input_tokens", None)
                        if cache_read is not None:
                            output.usage.cache_read = cache_read
                        cache_write = getattr(event.usage, "cache_creation_input_tokens", None)
                        if cache_write is not None:
                            output.usage.cache_write = cache_write
                        output.usage.total_tokens = (
                            output.usage.input
                            + output.usage.output
                            + output.usage.cache_read
                            + output.usage.cache_write
                        )
                        calculate_cost(model, output.usage)

            # Check abort
            if options and options.signal and options.signal.is_set():
                raise RuntimeError("Request was aborted")

            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError("An unknown error occurred")

            stream.push(AssistantMessageEventDone(reason=output.stop_reason, message=output))
            stream.end()

        except Exception as error:
            output.stop_reason = (
                "aborted" if (options and options.signal and options.signal.is_set()) else "error"
            )
            output.error_message = error.args[0] if error.args else str(error)
            stream.push(AssistantMessageEventError(reason=output.stop_reason, error=output))
            stream.end()

    import asyncio

    asyncio.create_task(_run())
    return stream


def stream_simple_anthropic(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simplified Anthropic streaming with automatic thinking configuration.

    Upstream: anthropic.ts → streamSimpleAnthropic()
    """
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)

    if not (options and options.reasoning):
        return stream_anthropic(
            model,
            context,
            AnthropicOptions(
                api_key=base.api_key,
                max_tokens=base.max_tokens,
                temperature=base.temperature,
                signal=base.signal,
                headers=base.headers,
                on_payload=base.on_payload,
                metadata=base.metadata,
                thinking_enabled=False,
            ),
        )

    # Adaptive thinking for Opus 4.6 / Sonnet 4.6
    if _supports_adaptive_thinking(model.id):
        effort = _map_thinking_level_to_effort(options.reasoning, model.id)
        return stream_anthropic(
            model,
            context,
            AnthropicOptions(
                api_key=base.api_key,
                max_tokens=base.max_tokens,
                temperature=base.temperature,
                signal=base.signal,
                headers=base.headers,
                on_payload=base.on_payload,
                metadata=base.metadata,
                thinking_enabled=True,
                effort=effort,
            ),
        )

    # Budget-based thinking for older models
    adjusted = adjust_max_tokens_for_thinking(
        base.max_tokens or 0,
        model.max_tokens,
        options.reasoning,
        options.thinking_budgets,
    )
    return stream_anthropic(
        model,
        context,
        AnthropicOptions(
            api_key=base.api_key,
            max_tokens=adjusted[0],
            temperature=base.temperature,
            signal=base.signal,
            headers=base.headers,
            on_payload=base.on_payload,
            metadata=base.metadata,
            thinking_enabled=True,
            thinking_budget_tokens=adjusted[1],
        ),
    )
