# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false
"""OpenAI Chat Completions API provider with full streaming support.

Supports the standard OpenAI Chat Completions API shape used by OpenAI,
and dozens of compatible providers (xai, groq, cerebras, openrouter,
vercel-ai-gateway, zai, minimax, huggingface, opencode, kimi-coding, etc.)
via ``baseUrl`` configuration.

Upstream reference: ``packages/ai/src/providers/openai-completions.ts``
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI

from otter_ai.models import calculate_cost, supports_xhigh
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
    Context,
    Message,
    Model,
    OpenAICompletionsCompat,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    Usage,
    UsageCost,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream
from otter_ai.utils.json_parse import parse_streaming_json
from otter_ai.utils.sanitize_unicode import sanitize_surrogates

from .github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from .simple_options import build_base_options, clamp_reasoning
from .transform_messages import transform_messages

# ============================================================================
# Provider-specific options
# ============================================================================


@dataclass
class OpenAICompletionsOptions(StreamOptions):
    """Extended stream options for OpenAI Completions providers."""

    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None


# ============================================================================
# Resolved compatibility (all fields required)
# ============================================================================


@dataclass
class _ResolvedCompat:
    """Fully resolved compatibility settings with all fields set."""

    supports_store: bool
    supports_developer_role: bool
    supports_reasoning_effort: bool
    reasoning_effort_map: dict[str, str] | None
    supports_usage_in_streaming: bool
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"]
    requires_tool_result_name: bool
    requires_assistant_after_tool_result: bool
    requires_thinking_as_text: bool
    thinking_format: Literal["openai", "openrouter", "zai", "qwen", "qwen-chat-template"]
    open_router_routing: Any | None
    vercel_gateway_routing: Any | None
    supports_strict_mode: bool


# ============================================================================
# Helpers
# ============================================================================


def _has_tool_history(messages: list[Message]) -> bool:
    """Check if conversation messages contain tool calls or tool results."""
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant" and any(block.type == "toolCall" for block in msg.content):
            return True
    return False


def _make_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=UsageCost(),
    )


def _make_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_make_usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


# ============================================================================
# Compat detection
# ============================================================================


def _detect_compat(model: Model) -> _ResolvedCompat:
    """Auto-detect compatibility settings from provider and baseUrl."""
    provider = model.provider
    base_url = model.base_url

    is_zai = provider == "zai" or "api.z.ai" in base_url

    is_non_standard = (
        provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_zai
        or provider == "opencode"
        or "opencode.ai" in base_url
    )

    use_max_tokens = "chutes.ai" in base_url
    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_groq = provider == "groq" or "groq.com" in base_url

    reasoning_effort_map: dict[str, str] = (
        {
            "minimal": "default",
            "low": "default",
            "medium": "default",
            "high": "default",
            "xhigh": "default",
        }
        if is_groq and model.id == "qwen/qwen3-32b"
        else {}
    )

    thinking_format: Literal["openai", "openrouter", "zai", "qwen", "qwen-chat-template"] = (
        "zai"
        if is_zai
        else "openrouter"
        if provider == "openrouter" or "openrouter.ai" in base_url
        else "openai"
    )

    return _ResolvedCompat(
        supports_store=not is_non_standard,
        supports_developer_role=not is_non_standard,
        supports_reasoning_effort=not is_grok and not is_zai,
        reasoning_effort_map=reasoning_effort_map,
        supports_usage_in_streaming=True,
        max_tokens_field="max_tokens" if use_max_tokens else "max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        thinking_format=thinking_format,
        open_router_routing=None,
        vercel_gateway_routing=None,
        supports_strict_mode=True,
    )


def _get_compat(model: Model) -> _ResolvedCompat:
    """Get resolved compatibility: explicit model.compat overrides auto-detection."""
    detected = _detect_compat(model)
    compat = model.compat
    if compat is None or not isinstance(compat, OpenAICompletionsCompat):
        return detected

    return _ResolvedCompat(
        supports_store=compat.supports_store
        if compat.supports_store is not None
        else detected.supports_store,
        supports_developer_role=(
            compat.supports_developer_role
            if compat.supports_developer_role is not None
            else detected.supports_developer_role
        ),
        supports_reasoning_effort=(
            compat.supports_reasoning_effort
            if compat.supports_reasoning_effort is not None
            else detected.supports_reasoning_effort
        ),
        reasoning_effort_map=(  # type: ignore[arg-type]
            compat.reasoning_effort_map
            if compat.reasoning_effort_map is not None
            else detected.reasoning_effort_map
        ),
        supports_usage_in_streaming=(
            compat.supports_usage_in_streaming
            if compat.supports_usage_in_streaming is not None
            else detected.supports_usage_in_streaming
        ),
        max_tokens_field=compat.max_tokens_field
        if compat.max_tokens_field is not None
        else detected.max_tokens_field,
        requires_tool_result_name=(
            compat.requires_tool_result_name
            if compat.requires_tool_result_name is not None
            else detected.requires_tool_result_name
        ),
        requires_assistant_after_tool_result=(
            compat.requires_assistant_after_tool_result
            if compat.requires_assistant_after_tool_result is not None
            else detected.requires_assistant_after_tool_result
        ),
        requires_thinking_as_text=(
            compat.requires_thinking_as_text
            if compat.requires_thinking_as_text is not None
            else detected.requires_thinking_as_text
        ),
        thinking_format=compat.thinking_format
        if compat.thinking_format is not None
        else detected.thinking_format,
        open_router_routing=compat.open_router_routing,
        vercel_gateway_routing=(
            compat.vercel_gateway_routing
            if compat.vercel_gateway_routing is not None
            else detected.vercel_gateway_routing
        ),
        supports_strict_mode=(
            compat.supports_strict_mode
            if compat.supports_strict_mode is not None
            else detected.supports_strict_mode
        ),
    )


# ============================================================================
# Stop reason mapping
# ============================================================================


def _map_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    """Map an OpenAI finish_reason to our StopReason."""
    if reason is None:
        return ("stop", None)
    if reason in ("stop", "end"):
        return ("stop", None)
    if reason == "length":
        return ("length", None)
    if reason in ("function_call", "tool_calls"):
        return ("toolUse", None)
    if reason == "content_filter":
        return ("error", "Provider finish_reason: content_filter")
    if reason == "network_error":
        return ("error", "Provider finish_reason: network_error")
    return ("error", f"Provider finish_reason: {reason}")


# ============================================================================
# Usage parsing
# ============================================================================


def _parse_chunk_usage(raw: Any, model: Model) -> Usage:
    """Parse token usage from an OpenAI chunk."""
    prompt_details: Any = raw.get("prompt_tokens_details") or {}
    completion_details: Any = raw.get("completion_tokens_details") or {}
    cached_tokens: int = prompt_details.get("cached_tokens", 0) or 0
    reasoning_tokens: int = completion_details.get("reasoning_tokens", 0) or 0
    input_tokens: int = (raw.get("prompt_tokens", 0) or 0) - cached_tokens
    output_tokens: int = (raw.get("completion_tokens", 0) or 0) + reasoning_tokens

    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cached_tokens,
        cache_write=0,
        total_tokens=input_tokens + output_tokens + cached_tokens,
        cost=UsageCost(),
    )
    calculate_cost(model, usage)
    return usage


# ============================================================================
# Reasoning effort mapping
# ============================================================================


def _map_reasoning_effort(
    effort: str,
    reasoning_effort_map: dict[str, str] | None,
) -> str:
    """Map reasoning effort level, using provider-specific overrides if available."""
    if reasoning_effort_map:
        return reasoning_effort_map.get(effort, effort)
    return effort


# ============================================================================
# Client creation
# ============================================================================


def _create_client(
    model: Model,
    context: Context,
    api_key: str,
    options_headers: dict[str, str] | None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client configured for the given model."""
    headers: dict[str, str] = dict(model.headers) if model.headers else {}

    if model.provider == "github-copilot":
        has_images = has_copilot_vision_input(context.messages)
        copilot_headers = build_copilot_dynamic_headers(context.messages, has_images)
        headers.update(copilot_headers)

    if options_headers:
        headers.update(options_headers)

    return AsyncOpenAI(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=headers,
    )


# ============================================================================
# Message conversion
# ============================================================================


def _maybe_add_openrouter_anthropic_cache_control(
    model: Model,
    messages: list[dict[str, Any]],
) -> None:
    """Add cache_control markers for Anthropic models via OpenRouter."""
    if model.provider != "openrouter" or not model.id.startswith("anthropic/"):
        return

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") not in ("user", "assistant"):
            continue

        content: Any = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
            ]
            return

        if not isinstance(content, list):
            continue

        for j in range(len(content) - 1, -1, -1):
            part: Any = content[j]
            if isinstance(part, dict) and part.get("type") == "text":
                part["cache_control"] = {"type": "ephemeral"}
                return


def _normalize_tool_call_id(tool_call_id: str, provider: str) -> str:
    """Normalize tool call IDs for cross-provider compatibility."""
    if "|" in tool_call_id:
        call_id = tool_call_id.split("|")[0]
        return call_id.replace("[^a-zA-Z0-9_-]", "_")[:40]  # noqa: S603

    if provider == "openai":
        return tool_call_id[:40] if len(tool_call_id) > 40 else tool_call_id
    return tool_call_id


def convert_messages(
    model: Model,
    context: Context,
    compat: _ResolvedCompat,
) -> list[dict[str, Any]]:
    """Convert otter-ai messages to OpenAI Chat Completions message format.

    This is exported for use by tests and other providers.
    """
    params: list[dict[str, Any]] = []

    transformed_messages = transform_messages(
        context.messages,
        model,
        lambda tc_id, _model, _msg: _normalize_tool_call_id(tc_id, model.provider),
    )

    # System prompt
    if context.system_prompt:
        use_developer_role = model.reasoning and compat.supports_developer_role
        role = "developer" if use_developer_role else "system"
        params.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    last_role: str | None = None

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]

        # Some providers need an assistant bridge between tool results and user messages
        if (
            compat.requires_assistant_after_tool_result
            and last_role == "toolResult"
            and msg.role == "user"
        ):
            params.append(
                {
                    "role": "assistant",
                    "content": "I have processed the tool results.",
                }
            )

        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append(
                    {
                        "role": "user",
                        "content": sanitize_surrogates(msg.content),
                    }
                )
            else:
                user_content: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        user_content.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(item.text),
                            }
                        )
                    elif item.type == "image":
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{item.mime_type};base64,{item.data}",
                                },
                            }
                        )

                if "image" not in model.input:
                    user_content = [c for c in user_content if c.get("type") != "image_url"]

                if user_content:
                    params.append({"role": "user", "content": user_content})

        elif msg.role == "assistant":
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "" if compat.requires_assistant_after_tool_result else None,
            }

            # Text blocks
            text_blocks = [b for b in msg.content if b.type == "text"]
            non_empty_text = [b for b in text_blocks if b.text and b.text.strip()]
            if non_empty_text:
                assistant_msg["content"] = "".join(
                    sanitize_surrogates(b.text) for b in non_empty_text
                )

            # Thinking blocks
            thinking_blocks = [b for b in msg.content if b.type == "thinking"]
            non_empty_thinking = [b for b in thinking_blocks if b.thinking and b.thinking.strip()]
            if non_empty_thinking:
                if compat.requires_thinking_as_text:
                    thinking_text = "\n\n".join(b.thinking for b in non_empty_thinking)
                    existing_content = assistant_msg.get("content")
                    if isinstance(existing_content, str) and existing_content:
                        assistant_msg["content"] = f"{thinking_text}\n\n{existing_content}"
                    else:
                        assistant_msg["content"] = thinking_text
                else:
                    # Use signature from first thinking block if available
                    signature = non_empty_thinking[0].thinking_signature
                    if signature and len(signature) > 0:
                        assistant_msg[signature] = "\n".join(b.thinking for b in non_empty_thinking)

            # Tool calls
            tool_calls = [b for b in msg.content if b.type == "toolCall"]
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
                reasoning_details = []
                for tc in tool_calls:
                    if tc.thought_signature:
                        try:
                            parsed = json.loads(tc.thought_signature)
                            reasoning_details.append(parsed)
                        except (json.JSONDecodeError, TypeError):
                            pass
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details

            # Skip empty assistant messages (no content, no tool calls)
            content: Any = assistant_msg.get("content")
            has_content = content is not None and (
                (isinstance(content, str) and len(content) > 0)
                or (isinstance(content, list) and len(content) > 0)
            )
            if not has_content and not assistant_msg.get("tool_calls"):
                i += 1
                last_role = msg.role
                continue

            params.append(assistant_msg)

        elif msg.role == "toolResult":
            # Batch consecutive tool results
            image_blocks: list[dict[str, Any]] = []
            j = i

            while j < len(transformed_messages) and transformed_messages[j].role == "toolResult":
                tool_msg = transformed_messages[j]

                text_result = "".join(c.text for c in tool_msg.content if c.type == "text")
                has_images = any(c.type == "image" for c in tool_msg.content)

                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": sanitize_surrogates(
                        text_result if text_result else "(see attached image)"
                    ),
                    "tool_call_id": tool_msg.tool_call_id,
                }
                if compat.requires_tool_result_name and tool_msg.tool_name:
                    tool_result_msg["name"] = tool_msg.tool_name
                params.append(tool_result_msg)

                if has_images and "image" in model.input:
                    for block in tool_msg.content:
                        if block.type == "image":
                            image_blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{block.mime_type};base64,{block.data}",
                                    },
                                }
                            )

                j += 1

            i = j - 1

            if image_blocks:
                if compat.requires_assistant_after_tool_result:
                    params.append(
                        {
                            "role": "assistant",
                            "content": "I have processed the tool results.",
                        }
                    )
                params.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Attached image(s) from tool result:"},
                            *image_blocks,
                        ],
                    }
                )
                last_role = "user"
            else:
                last_role = "toolResult"

            i += 1
            continue

        last_role = msg.role
        i += 1

    return params


# ============================================================================
# Tool conversion
# ============================================================================


def _convert_tools(tools: list[Tool], compat: _ResolvedCompat) -> list[dict[str, Any]]:
    """Convert otter-ai Tool definitions to OpenAI function tool format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        tool_def: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_json_schema(),
            },
        }
        if compat.supports_strict_mode:
            tool_def["function"]["strict"] = False
        result.append(tool_def)
    return result


# ============================================================================
# Build params
# ============================================================================


def _build_params(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None,
    compat: _ResolvedCompat,
) -> dict[str, Any]:
    """Build the OpenAI API request parameters."""
    messages = convert_messages(model, context, compat)
    _maybe_add_openrouter_anthropic_cache_control(model, messages)

    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = False

    if options and options.max_tokens:
        if compat.max_tokens_field == "max_tokens":
            params["max_tokens"] = options.max_tokens
        else:
            params["max_completion_tokens"] = options.max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = _convert_tools(context.tools, compat)
    elif _has_tool_history(context.messages):
        # Some providers require tools param when conversation has tool history
        params["tools"] = []

    if options and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    # Thinking / reasoning effort
    if options and options.reasoning_effort and model.reasoning:
        if compat.thinking_format == "zai" or compat.thinking_format == "qwen":
            params["enable_thinking"] = True
        elif compat.thinking_format == "qwen-chat-template":
            params["chat_template_kwargs"] = {"enable_thinking": True}
        elif compat.thinking_format == "openrouter":
            params["reasoning"] = {
                "effort": _map_reasoning_effort(
                    options.reasoning_effort, compat.reasoning_effort_map
                ),
            }
        elif compat.supports_reasoning_effort:
            params["reasoning_effort"] = _map_reasoning_effort(
                options.reasoning_effort,
                compat.reasoning_effort_map,
            )

    # OpenRouter provider routing
    if "openrouter.ai" in model.base_url and compat.open_router_routing:
        params["provider"] = compat.open_router_routing

    # Vercel AI Gateway routing
    if "ai-gateway.vercel.sh" in model.base_url and compat.vercel_gateway_routing:
        routing = compat.vercel_gateway_routing
        gateway_options: dict[str, Any] = {}
        if routing.get("only"):
            gateway_options["only"] = routing["only"]
        if routing.get("order"):
            gateway_options["order"] = routing["order"]
        if gateway_options:
            params["providerOptions"] = {"gateway": gateway_options}

    return params


# ============================================================================
# Stream implementation
# ============================================================================


def stream_openai_completions(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an OpenAI Completions API response.

    This is the full-featured stream function that accepts provider-specific
    options (``toolChoice``, ``reasoningEffort``, etc.).

    Returns an :class:`AssistantMessageEventStream` immediately.  The async
    work runs in a background task, pushing events to the stream.
    Failures are encoded as ``"error"`` events — **not** raised.
    """
    import asyncio

    from otter_ai.env_api_keys import get_env_api_key

    stream = AssistantMessageEventStream()
    output = _make_output(model)

    async def _run() -> None:
        try:
            api_key = (
                (options.api_key if options else None) or get_env_api_key(model.provider) or ""
            )
            client = _create_client(model, context, api_key, options.headers if options else None)
            compat = _get_compat(model)
            params = _build_params(model, context, options, compat)

            # Allow payload inspection/replacement
            if options and options.on_payload:
                next_params = await options.on_payload(params, model)
                if next_params is not None:
                    params = next_params

            openai_response = await client.chat.completions.create(**params)
            stream.push(AssistantMessageEventStart(partial=output))

            # Track current content block
            current_block: dict[str, Any] | None = None
            partial_args: str = ""

            def _block_index() -> int:
                return len(output.content) - 1

            def _finish_current_block() -> None:
                nonlocal current_block, partial_args
                if current_block is None:
                    return

                idx = _block_index()
                if current_block["type"] == "text":
                    stream.push(
                        AssistantMessageEventTextEnd(
                            content_index=idx,
                            content=current_block["text"],
                            partial=output,
                        )
                    )
                elif current_block["type"] == "thinking":
                    stream.push(
                        AssistantMessageEventThinkingEnd(
                            content_index=idx,
                            content=current_block["thinking"],
                            partial=output,
                        )
                    )
                elif current_block["type"] == "toolCall":
                    final_tc = ToolCall(
                        type="toolCall",
                        id=current_block["id"],
                        name=current_block["name"],
                        arguments=parse_streaming_json(partial_args) or {},
                    )
                    output.content[idx] = final_tc
                    stream.push(
                        AssistantMessageEventToolcallEnd(
                            content_index=idx,
                            tool_call=final_tc,
                            partial=output,
                        )
                    )

                current_block = None
                partial_args = ""

            async for chunk in openai_response:
                # Check abort signal between chunks
                if options and options.signal and options.signal.is_set():
                    raise RuntimeError("Request was aborted")

                # Capture response ID from first chunk
                if not output.response_id and hasattr(chunk, "id") and chunk.id:
                    output.response_id = chunk.id

                # Parse usage
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = (
                        chunk.usage.model_dump()
                        if hasattr(chunk.usage, "model_dump")
                        else dict(chunk.usage)
                    )
                    output.usage = _parse_chunk_usage(usage_data, model)

                chunk_data = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                choices = chunk_data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]

                # Fallback: some providers return usage in choice.usage
                if not (hasattr(chunk, "usage") and chunk.usage) and "usage" in choice:
                    output.usage = _parse_chunk_usage(choice["usage"], model)

                # Finish reason
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    stop_reason, error_message = _map_stop_reason(finish_reason)
                    output.stop_reason = stop_reason
                    if error_message:
                        output.error_message = error_message

                delta = choice.get("delta", {})
                if not delta:
                    continue

                # --- Text content ---
                content = delta.get("content")
                if content is not None and content:
                    if current_block is None or current_block["type"] != "text":
                        _finish_current_block()
                        current_block = {"type": "text", "text": ""}
                        output.content.append(TextContent(type="text", text=""))
                        stream.push(
                            AssistantMessageEventTextStart(
                                content_index=_block_index(),
                                partial=output,
                            )
                        )

                    if current_block["type"] == "text":
                        current_block["text"] += content
                        output.content[_block_index()] = TextContent(
                            type="text",
                            text=current_block["text"],
                        )
                        stream.push(
                            AssistantMessageEventTextDelta(
                                content_index=_block_index(),
                                delta=content,
                                partial=output,
                            )
                        )

                # --- Reasoning content (llama.cpp, other compatible endpoints) ---
                reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"]
                found_reasoning_field: str | None = None
                for field_name in reasoning_fields:
                    field_val = delta.get(field_name)
                    if field_val is not None and field_val:
                        found_reasoning_field = field_name
                        break

                if found_reasoning_field:
                    if current_block is None or current_block["type"] != "thinking":
                        _finish_current_block()
                        current_block = {
                            "type": "thinking",
                            "thinking": "",
                            "thinking_signature": found_reasoning_field,
                        }
                        output.content.append(ThinkingContent(type="thinking", thinking=""))
                        stream.push(
                            AssistantMessageEventThinkingStart(
                                content_index=_block_index(),
                                partial=output,
                            )
                        )

                    if current_block["type"] == "thinking":
                        reasoning_delta = str(delta[found_reasoning_field])
                        current_block["thinking"] += reasoning_delta
                        output.content[_block_index()] = ThinkingContent(
                            type="thinking",
                            thinking=current_block["thinking"],
                            thinking_signature=current_block.get("thinking_signature"),
                        )
                        stream.push(
                            AssistantMessageEventThinkingDelta(
                                content_index=_block_index(),
                                delta=reasoning_delta,
                                partial=output,
                            )
                        )

                # --- Tool calls ---
                tool_calls_delta = delta.get("tool_calls")
                if tool_calls_delta:
                    for tc_delta in tool_calls_delta:
                        tc_id = tc_delta.get("id")
                        tc_function = tc_delta.get("function", {})
                        tc_name = tc_function.get("name") if tc_function else None
                        tc_args = tc_function.get("arguments") if tc_function else None

                        if (
                            current_block is None
                            or current_block["type"] != "toolCall"
                            or (tc_id and current_block["id"] != tc_id)
                        ):
                            _finish_current_block()
                            current_block = {
                                "type": "toolCall",
                                "id": tc_id or "",
                                "name": tc_name or "",
                            }
                            output.content.append(
                                ToolCall(
                                    type="toolCall",
                                    id=current_block["id"],
                                    name=current_block["name"],
                                    arguments={},
                                )
                            )
                            stream.push(
                                AssistantMessageEventToolcallStart(
                                    content_index=_block_index(),
                                    partial=output,
                                )
                            )

                        if current_block["type"] == "toolCall":
                            if tc_id:
                                current_block["id"] = tc_id
                            if tc_name:
                                current_block["name"] = tc_name

                            delta_str = ""
                            if tc_args:
                                delta_str = tc_args
                                partial_args += tc_args
                                parsed = parse_streaming_json(partial_args) or {}
                                output.content[_block_index()] = ToolCall(
                                    type="toolCall",
                                    id=current_block["id"],
                                    name=current_block["name"],
                                    arguments=parsed,
                                )

                            stream.push(
                                AssistantMessageEventToolcallDelta(
                                    content_index=_block_index(),
                                    delta=delta_str,
                                    partial=output,
                                )
                            )

                # --- Reasoning details (encrypted thought signatures) ---
                reasoning_details = delta.get("reasoning_details")
                if reasoning_details and isinstance(reasoning_details, list):
                    for detail in reasoning_details:
                        if (
                            isinstance(detail, dict)
                            and detail.get("type") == "reasoning.encrypted"
                            and detail.get("id")
                            and detail.get("data")
                        ):
                            for block in output.content:
                                if isinstance(block, ToolCall) and block.id == detail["id"]:
                                    block.thought_signature = json.dumps(detail)

            # Finish last block
            _finish_current_block()

            if output.stop_reason == "aborted":
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "error":
                raise RuntimeError(output.error_message or "Provider returned an error stop reason")

            # Ensure stopReason is valid for "done" event
            final_reason: Literal["stop", "length", "toolUse"] = (
                output.stop_reason
                if output.stop_reason in ("stop", "length", "toolUse")
                else "stop"
            )
            stream.push(AssistantMessageEventDone(reason=final_reason, message=output))
            stream.end()

        except Exception as error:
            output.stop_reason = (
                "aborted" if (options and options.signal and options.signal.is_set()) else "error"
            )
            error_msg = error.args[0] if error.args else str(error)
            # Some providers via OpenRouter give additional information
            raw_metadata = getattr(getattr(error, "error", None), "metadata", None)
            if raw_metadata and hasattr(raw_metadata, "raw"):
                error_msg += f"\n{raw_metadata.raw}"
            output.error_message = error_msg
            stream.push(AssistantMessageEventError(reason=output.stop_reason, error=output))
            stream.end()

    asyncio.create_task(_run())
    return stream


# ============================================================================
# Simple stream (with reasoning level support)
# ============================================================================


def stream_simple_openai_completions(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream with simplified options (includes reasoning level support).

    Wraps :func:`stream_openai_completions` with automatic reasoning effort
    clamping and base option building.
    """
    from otter_ai.env_api_keys import get_env_api_key

    api_key = get_env_api_key(model.provider)
    if not api_key:
        api_key = options.api_key if options else None
    if not api_key:
        msg = f"No API key for provider: {model.provider}"
        raise RuntimeError(msg)

    base = build_base_options(model, options, api_key)
    reasoning_effort = (
        options.reasoning
        if options and supports_xhigh(model)
        else clamp_reasoning(options.reasoning if options else None)
    )
    tool_choice = None

    return stream_openai_completions(
        model,
        context,
        OpenAICompletionsOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            signal=base.signal,
            api_key=base.api_key,
            transport=base.transport,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            on_payload=base.on_payload,
            headers=base.headers,
            max_retry_delay_ms=base.max_retry_delay_ms,
            metadata=base.metadata,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
        ),
    )
