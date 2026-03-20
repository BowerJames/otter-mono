# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Mistral AI provider.

Upstream: packages/ai/src/providers/mistral.ts (~585 lines)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from mistralai.client import Mistral

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
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    Usage,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream
from otter_ai.utils.hash import short_hash
from otter_ai.utils.json_parse import parse_streaming_json
from otter_ai.utils.sanitize_unicode import sanitize_surrogates

from .simple_options import build_base_options, clamp_reasoning
from .transform_messages import transform_messages

_MISTRAL_TOOL_CALL_ID_LENGTH = 9
_MAX_MISTRAL_ERROR_BODY_CHARS = 4000


# ============================================================================
# Options
# ============================================================================


@dataclass
class MistralOptions(StreamOptions):
    """Options for Mistral streaming.

    Upstream: ``interface MistralOptions extends StreamOptions``
    """

    tool_choice: str | dict[str, Any] | None = None
    prompt_mode: str | None = None


# ============================================================================
# Tool call ID normalization
# ============================================================================


def _create_mistral_tool_call_id_normalizer() -> Any:
    """Create a stateful normalizer for Mistral tool call IDs (9-char alphanumeric)."""
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def _normalize(tool_call_id: str) -> str:
        existing = id_map.get(tool_call_id)
        if existing:
            return existing

        attempt = 0
        while True:
            candidate = _derive_mistral_tool_call_id(tool_call_id, attempt)
            owner = reverse_map.get(candidate)
            if not owner or owner == tool_call_id:
                id_map[tool_call_id] = candidate
                reverse_map[candidate] = tool_call_id
                return candidate
            attempt += 1

    return _normalize


def _derive_mistral_tool_call_id(tool_call_id: str, attempt: int) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]", "", tool_call_id) if tool_call_id else ""
    if attempt == 0 and len(normalized) == _MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or tool_call_id
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return re.sub(r"[^a-zA-Z0-9]", "", short_hash(seed))[:_MISTRAL_TOOL_CALL_ID_LENGTH]


# ============================================================================
# Error formatting
# ============================================================================


def _format_mistral_error(error: Any) -> str:
    if isinstance(error, Exception):
        status_code = getattr(error, "status_code", None)
        body = getattr(error, "body", None)
        body_text = body.strip() if isinstance(body, str) else None
        if isinstance(status_code, int) and body_text:
            return (
                f"Mistral API error ({status_code}): "
                f"{_truncate_error_text(body_text, _MAX_MISTRAL_ERROR_BODY_CHARS)}"
            )
        if isinstance(status_code, int):
            return f"Mistral API error ({status_code}): {error}"
        return str(error)
    return _safe_json_stringify(error)


def _truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value) or str(value)
    except (TypeError, ValueError):
        return str(value)


# ============================================================================
# Message conversion
# ============================================================================


def _to_chat_messages(messages: list[Any], supports_images: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": sanitize_surrogates(msg.content)})
                continue

            had_images = any(item.type == "image" for item in msg.content)
            content: list[dict[str, Any]] = []
            for item in msg.content:
                if item.type == "text":
                    content.append({"type": "text", "text": sanitize_surrogates(item.text)})
                elif supports_images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": f"data:{item.mime_type};base64,{item.data}",
                        }
                    )

            if content:
                result.append({"role": "user", "content": content})
            elif had_images and not supports_images:
                result.append(
                    {"role": "user", "content": "(image omitted: model does not support images)"}
                )

        elif msg.role == "assistant":
            content_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in msg.content:
                if block.type == "text":
                    if block.text.strip():
                        content_parts.append(
                            {"type": "text", "text": sanitize_surrogates(block.text)}
                        )
                elif block.type == "thinking":
                    if block.thinking.strip():
                        content_parts.append(
                            {
                                "type": "thinking",
                                "thinking": [
                                    {"type": "text", "text": sanitize_surrogates(block.thinking)}
                                ],
                            }
                        )
                elif block.type == "toolCall":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments or {}),
                            },
                        }
                    )

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if content_parts:
                assistant_msg["content"] = content_parts
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            if content_parts or tool_calls:
                result.append(assistant_msg)

        elif msg.role == "toolResult":
            text_parts = [c for c in msg.content if c.type == "text"]
            text_result = "\n".join(sanitize_surrogates(c.text) for c in text_parts)
            has_images = any(c.type == "image" for c in msg.content)

            tool_text = _build_tool_result_text(
                text_result, has_images, supports_images, msg.is_error
            )
            tool_content: list[dict[str, Any]] = [{"type": "text", "text": tool_text}]

            if supports_images:
                for part in msg.content:
                    if part.type == "image":
                        tool_content.append(
                            {
                                "type": "image_url",
                                "image_url": f"data:{part.mime_type};base64,{part.data}",
                            }
                        )

            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.tool_name,
                    "content": tool_content,
                }
            )

    return result


def _build_tool_result_text(
    text: str,
    has_images: bool,
    supports_images: bool,
    is_error: bool,
) -> str:
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""

    if trimmed:
        suffix = (
            "\n[tool image omitted: model does not support images]"
            if (has_images and not supports_images)
            else ""
        )
        return f"{error_prefix}{trimmed}{suffix}"

    if has_images:
        if supports_images:
            return f"{error_prefix}(see attached image)"
        return f"{error_prefix}(image omitted: model does not support images)"

    return f"{error_prefix}(no tool output)"


def _to_function_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": False,
            },
        }
        for tool in tools
    ]


def _map_tool_choice(choice: Any) -> Any:
    if not choice:
        return None
    if choice in ("auto", "none", "any", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        return choice
    return choice


def _map_chat_stop_reason(reason: str | None) -> str:
    if not reason:
        return "stop"
    match reason:
        case "stop":
            return "stop"
        case "length" | "model_length":
            return "length"
        case "tool_calls":
            return "toolUse"
        case "error":
            return "error"
        case _:
            return "stop"


# ============================================================================
# Streaming
# ============================================================================


def stream_mistral(
    model: Model,
    context: Context,
    options: MistralOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream completions from Mistral AI.

    Upstream: mistral.ts → streamMistral()
    """
    stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api="mistral-conversations",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")

            mistral_client = Mistral(
                api_key=api_key,
                server_url=model.base_url,
            )

            normalize_id = _create_mistral_tool_call_id_normalizer()
            transformed = transform_messages(context.messages, model, normalize_id)

            payload = _build_chat_payload(model, context, transformed, options)
            if options and options.on_payload:
                next_payload = await options.on_payload(payload, model)
                if next_payload is not None:
                    payload = next_payload

            request_options: dict[str, Any] = {}
            if options and options.signal:
                request_options["signal"] = options.signal

            merged_headers: dict[str, str] = {}
            if model.headers:
                merged_headers.update(model.headers)
            if options and options.headers:
                merged_headers.update(options.headers)
            if options and options.session_id and "x-affinity" not in merged_headers:
                merged_headers["x-affinity"] = options.session_id
            if merged_headers:
                request_options["headers"] = merged_headers

            mistral_stream = await mistral_client.chat.stream_async(payload)
            stream.push(AssistantMessageEventStart(partial=output))
            await _consume_chat_stream(model, output, stream, mistral_stream)

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
            output.error_message = _format_mistral_error(error)
            stream.push(AssistantMessageEventError(reason=output.stop_reason, error=output))
            stream.end()

    import asyncio

    asyncio.create_task(_run())
    return stream


async def _consume_chat_stream(
    model: Model,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    mistral_stream: Any,
) -> None:
    """Process Mistral chat stream events."""
    current_block: TextContent | ThinkingContent | None = None
    blocks = output.content
    tool_blocks_by_key: dict[str, int] = {}

    def block_index() -> int:
        return len(blocks) - 1

    def finish_block(block: TextContent | ThinkingContent | None) -> None:
        nonlocal current_block
        if not block:
            return
        idx = len(blocks) - 1
        if block.type == "text":
            stream.push(
                AssistantMessageEventTextEnd(
                    content_index=idx,
                    content=block.text,
                    partial=output,
                )
            )
        elif block.type == "thinking":
            stream.push(
                AssistantMessageEventThinkingEnd(
                    content_index=idx,
                    content=block.thinking,
                    partial=output,
                )
            )

    async for event in mistral_stream:
        chunk = event.data

        if not output.response_id and hasattr(chunk, "id") and chunk.id:
            output.response_id = chunk.id

        if hasattr(chunk, "usage") and chunk.usage:
            output.usage.input = getattr(chunk.usage, "prompt_tokens", 0) or 0
            output.usage.output = getattr(chunk.usage, "completion_tokens", 0) or 0
            output.usage.cache_read = 0
            output.usage.cache_write = 0
            output.usage.total_tokens = (
                getattr(chunk.usage, "total_tokens", 0) or output.usage.input + output.usage.output
            )
            calculate_cost(model, output.usage)

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        choice = choices[0]
        if not choice:
            continue

        if hasattr(choice, "finish_reason") and choice.finish_reason:
            output.stop_reason = _map_chat_stop_reason(choice.finish_reason)

        delta = choice.delta

        # Text/thinking content
        content = getattr(delta, "content", None)
        if content is not None:
            items = [content] if isinstance(content, str) else content
            for item in items:
                if isinstance(item, str):
                    text_delta = sanitize_surrogates(item)
                    if not current_block or current_block.type != "text":
                        finish_block(current_block)
                        current_block = TextContent(type="text", text="")
                        output.content.append(current_block)
                        stream.push(
                            AssistantMessageEventTextStart(
                                content_index=block_index(), partial=output
                            )
                        )
                    current_block.text += text_delta
                    stream.push(
                        AssistantMessageEventTextDelta(
                            content_index=block_index(), delta=text_delta, partial=output
                        )
                    )
                    continue

                item_type = getattr(item, "type", None)
                if item_type == "thinking":
                    parts = getattr(item, "thinking", [])
                    delta_text = "".join(
                        p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                        for p in parts
                    )
                    thinking_delta = sanitize_surrogates(delta_text)
                    if not thinking_delta:
                        continue
                    if not current_block or current_block.type != "thinking":
                        finish_block(current_block)
                        current_block = ThinkingContent(type="thinking", thinking="")
                        output.content.append(current_block)
                        stream.push(
                            AssistantMessageEventThinkingStart(
                                content_index=block_index(), partial=output
                            )
                        )
                    current_block.thinking += thinking_delta
                    stream.push(
                        AssistantMessageEventThinkingDelta(
                            content_index=block_index(), delta=thinking_delta, partial=output
                        )
                    )
                    continue

                if item_type == "text":
                    text_delta = sanitize_surrogates(getattr(item, "text", ""))
                    if not current_block or current_block.type != "text":
                        finish_block(current_block)
                        current_block = TextContent(type="text", text="")
                        output.content.append(current_block)
                        stream.push(
                            AssistantMessageEventTextStart(
                                content_index=block_index(), partial=output
                            )
                        )
                    current_block.text += text_delta
                    stream.push(
                        AssistantMessageEventTextDelta(
                            content_index=block_index(), delta=text_delta, partial=output
                        )
                    )

        # Tool calls
        tool_calls = getattr(delta, "tool_calls", None) or []
        for tc in tool_calls:
            if current_block:
                finish_block(current_block)
                current_block = None

            tc_id = getattr(tc, "id", None)
            if not tc_id or tc_id == "null":
                tc_id = _derive_mistral_tool_call_id(f"toolcall:{getattr(tc, 'index', 0)}", 0)

            key = f"{tc_id}:{getattr(tc, 'index', 0)}"
            existing_idx = tool_blocks_by_key.get(key)

            if existing_idx is not None:
                existing = output.content[existing_idx]
                block = existing if existing.type == "toolCall" else None
            else:
                block = None

            if block is None:
                func = tc.function
                block = ToolCall(
                    type="toolCall",
                    id=tc_id,
                    name=func.name,
                    arguments={},
                )
                block._partial_args = ""  # type: ignore[attr-defined]
                output.content.append(block)
                tool_blocks_by_key[key] = len(output.content) - 1
                stream.push(
                    AssistantMessageEventToolcallStart(
                        content_index=len(output.content) - 1,
                        partial=output,
                    )
                )

            func = tc.function
            args_delta = (
                func.arguments
                if isinstance(func.arguments, str)
                else json.dumps(func.arguments or {})
            )
            partial = getattr(block, "_partial_args", "") or ""
            partial += args_delta
            block._partial_args = partial  # type: ignore[attr-defined]
            block.arguments = parse_streaming_json(partial) or {}
            stream.push(
                AssistantMessageEventToolcallDelta(
                    content_index=tool_blocks_by_key[key],
                    delta=args_delta,
                    partial=output,
                )
            )

    finish_block(current_block)

    for idx in tool_blocks_by_key.values():
        block = output.content[idx]
        if block.type != "toolCall":
            continue
        partial = getattr(block, "_partial_args", "") or ""
        block.arguments = parse_streaming_json(partial) or {}
        if hasattr(block, "_partial_args"):
            del block._partial_args  # type: ignore[attr-defined]
        stream.push(
            AssistantMessageEventToolcallEnd(content_index=idx, tool_call=block, partial=output)
        )


def _build_chat_payload(
    model: Model,
    context: Context,
    messages: list[Any],
    options: MistralOptions | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "messages": _to_chat_messages(messages, "image" in (model.input or [])),
    }

    if context.tools:
        payload["tools"] = _to_function_tools(context.tools)
    if options and options.temperature is not None:
        payload["temperature"] = options.temperature
    if options and options.max_tokens is not None:
        payload["max_tokens"] = options.max_tokens
    if options and options.tool_choice:
        payload["tool_choice"] = _map_tool_choice(options.tool_choice)
    if options and options.prompt_mode:
        payload["prompt_mode"] = options.prompt_mode

    if context.system_prompt:
        payload["messages"].insert(
            0,
            {
                "role": "system",
                "content": sanitize_surrogates(context.system_prompt),
            },
        )

    return payload


def stream_simple_mistral(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simplified Mistral streaming.

    Upstream: mistral.ts → streamSimpleMistral()
    """
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)
    reasoning = clamp_reasoning(options.reasoning) if options else None

    return stream_mistral(
        model,
        context,
        MistralOptions(
            api_key=base.api_key,
            max_tokens=base.max_tokens,
            temperature=base.temperature,
            signal=base.signal,
            headers=base.headers,
            on_payload=base.on_payload,
            metadata=base.metadata,
            prompt_mode="reasoning" if (model.reasoning and reasoning) else None,
        ),
    )
