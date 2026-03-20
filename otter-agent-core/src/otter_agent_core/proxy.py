"""Proxy stream function for apps that route LLM calls through a server.

The server manages auth and proxies requests to LLM providers.  The server
strips the ``partial`` field from delta events to reduce bandwidth; this
module reconstructs the partial :class:`AssistantMessage` client-side.

Use this as the ``stream_fn`` option when creating an :class:`Agent` that
needs to go through a proxy.

Upstream reference: ``packages/agent/src/proxy.ts``
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from otter_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
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
    Content,
    Context,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultContent,
    Usage,
    UsageCost,
)
from otter_ai.utils.event_stream import EventStream
from otter_ai.utils.json_parse import parse_streaming_json


# ============================================================================
# ProxyMessageEventStream
# ============================================================================


class ProxyMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Concrete ``EventStream`` for proxy-streamed assistant message events.

    Terminal events:
    * ``"done"``  — extracts ``event.message``
    * ``"error"`` — extracts ``event.error``
    """

    def __init__(self) -> None:
        super().__init__(
            is_complete=lambda e: e.type in ("done", "error"),
            extract_result=self._extract,
        )

    @staticmethod
    def _extract(event: AssistantMessageEvent) -> AssistantMessage:
        if event.type == "done":
            return event.message
        if event.type == "error":
            return event.error
        raise RuntimeError("Unexpected event type for final result")


# ============================================================================
# Proxy Event Types
#
# Bandwidth-optimized event types sent by the server with the ``partial``
# field stripped.
# ============================================================================


@dataclass
class ProxyEventStart:
    type: Literal["start"]


@dataclass
class ProxyEventTextStart:
    type: Literal["text_start"]
    content_index: int


@dataclass
class ProxyEventTextDelta:
    type: Literal["text_delta"]
    content_index: int
    delta: str


@dataclass
class ProxyEventTextEnd:
    type: Literal["text_end"]
    content_index: int
    content_signature: str | None = None


@dataclass
class ProxyEventThinkingStart:
    type: Literal["thinking_start"]
    content_index: int


@dataclass
class ProxyEventThinkingDelta:
    type: Literal["thinking_delta"]
    content_index: int
    delta: str


@dataclass
class ProxyEventThinkingEnd:
    type: Literal["thinking_end"]
    content_index: int
    content_signature: str | None = None


@dataclass
class ProxyEventToolcallStart:
    type: Literal["toolcall_start"]
    content_index: int
    id: str
    tool_name: str


@dataclass
class ProxyEventToolcallDelta:
    type: Literal["toolcall_delta"]
    content_index: int
    delta: str


@dataclass
class ProxyEventToolcallEnd:
    type: Literal["toolcall_end"]
    content_index: int


@dataclass
class ProxyEventDone:
    type: Literal["done"]
    reason: Literal["stop", "length", "toolUse"]
    usage: Usage


@dataclass
class ProxyEventError:
    type: Literal["error"]
    reason: Literal["aborted", "error"]
    error_message: str | None = None
    usage: Usage = field(default_factory=Usage)


type ProxyAssistantMessageEvent = (
    ProxyEventStart
    | ProxyEventTextStart
    | ProxyEventTextDelta
    | ProxyEventTextEnd
    | ProxyEventThinkingStart
    | ProxyEventThinkingDelta
    | ProxyEventThinkingEnd
    | ProxyEventToolcallStart
    | ProxyEventToolcallDelta
    | ProxyEventToolcallEnd
    | ProxyEventDone
    | ProxyEventError
)


# ============================================================================
# Proxy Stream Options
# ============================================================================


@dataclass(kw_only=True)
class ProxyStreamOptions(SimpleStreamOptions):
    """Options for :func:`stream_proxy`.

    Extends :class:`SimpleStreamOptions` with proxy-specific settings.
    """

    auth_token: str
    """Auth token for the proxy server."""

    proxy_url: str
    """Proxy server URL (e.g. ``"https://genai.example.com"``)."""


# ============================================================================
# Event processing
# ============================================================================


def _process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    partial: AssistantMessage,
) -> AssistantMessageEvent | None:
    """Process a proxy event and update the partial message.

    Returns the corresponding :class:`AssistantMessageEvent`, or ``None``
    for events that should be silently consumed (e.g. ``toolcall_end``).
    """
    match proxy_event.type:
        case "start":
            return AssistantMessageEventStart(type="start", partial=partial)

        case "text_start":
            partial.content[proxy_event.content_index] = TextContent(type="text", text="")
            return AssistantMessageEventTextStart(
                type="text_start",
                content_index=proxy_event.content_index,
                partial=partial,
            )

        case "text_delta":
            content = partial.content[proxy_event.content_index]
            if not isinstance(content, TextContent):
                raise RuntimeError("Received text_delta for non-text content")
            content.text += proxy_event.delta
            return AssistantMessageEventTextDelta(
                type="text_delta",
                content_index=proxy_event.content_index,
                delta=proxy_event.delta,
                partial=partial,
            )

        case "text_end":
            content = partial.content[proxy_event.content_index]
            if not isinstance(content, TextContent):
                raise RuntimeError("Received text_end for non-text content")
            content.text_signature = proxy_event.content_signature
            return AssistantMessageEventTextEnd(
                type="text_end",
                content_index=proxy_event.content_index,
                content=content.text,
                partial=partial,
            )

        case "thinking_start":
            partial.content[proxy_event.content_index] = ThinkingContent(
                type="thinking", thinking="",
            )
            return AssistantMessageEventThinkingStart(
                type="thinking_start",
                content_index=proxy_event.content_index,
                partial=partial,
            )

        case "thinking_delta":
            content = partial.content[proxy_event.content_index]
            if not isinstance(content, ThinkingContent):
                raise RuntimeError("Received thinking_delta for non-thinking content")
            content.thinking += proxy_event.delta
            return AssistantMessageEventThinkingDelta(
                type="thinking_delta",
                content_index=proxy_event.content_index,
                delta=proxy_event.delta,
                partial=partial,
            )

        case "thinking_end":
            content = partial.content[proxy_event.content_index]
            if not isinstance(content, ThinkingContent):
                raise RuntimeError("Received thinking_end for non-thinking content")
            content.thinking_signature = proxy_event.content_signature
            return AssistantMessageEventThinkingEnd(
                type="thinking_end",
                content_index=proxy_event.content_index,
                content=content.thinking,
                partial=partial,
            )

        case "toolcall_start":
            tc = ToolCall(
                type="toolCall",
                id=proxy_event.id,
                name=proxy_event.tool_name,
                arguments={},
            )
            # Store partial JSON accumulator as an extra attribute.
            tc._partial_json = ""  # type: ignore[attr-defined]
            partial.content[proxy_event.content_index] = tc
            return AssistantMessageEventToolcallStart(
                type="toolcall_start",
                content_index=proxy_event.content_index,
                partial=partial,
            )

        case "toolcall_delta":
            content = partial.content[proxy_event.content_index]
            if not isinstance(content, ToolCall):
                raise RuntimeError("Received toolcall_delta for non-toolCall content")
            content._partial_json += proxy_event.delta  # type: ignore[attr-defined]
            content.arguments = parse_streaming_json(content._partial_json) or {}  # type: ignore[attr-defined]
            # Trigger reactivity — replace the object in the list so that
            # the partial reference sees the updated arguments.
            partial.content[proxy_event.content_index] = content
            return AssistantMessageEventToolcallDelta(
                type="toolcall_delta",
                content_index=proxy_event.content_index,
                delta=proxy_event.delta,
                partial=partial,
            )

        case "toolcall_end":
            content = partial.content[proxy_event.content_index]
            if isinstance(content, ToolCall):
                # Remove the transient partial JSON accumulator.
                if hasattr(content, "_partial_json"):
                    del content._partial_json  # type: ignore[attr-defined]
                return AssistantMessageEventToolcallEnd(
                    type="toolcall_end",
                    content_index=proxy_event.content_index,
                    tool_call=content,
                    partial=partial,
                )
            return None

        case "done":
            partial.stop_reason = proxy_event.reason
            partial.usage = proxy_event.usage
            return AssistantMessageEventDone(
                type="done",
                reason=proxy_event.reason,
                message=partial,
            )

        case "error":
            partial.stop_reason = proxy_event.reason
            partial.error_message = proxy_event.error_message
            partial.usage = proxy_event.usage
            return AssistantMessageEventError(
                type="error",
                reason=proxy_event.reason,
                error=partial,
            )

        case _:
            # Exhaustive check — should be unreachable.
            raise RuntimeError(f"Unhandled proxy event type: {proxy_event.type!r}")


# ============================================================================
# Proxy event deserialization
# ============================================================================


def _parse_proxy_event(raw: dict[str, Any]) -> ProxyAssistantMessageEvent:
    """Deserialize a JSON dict into the appropriate proxy event dataclass."""
    event_type = raw.get("type")
    usage_data = raw.get("usage")

    match event_type:
        case "start":
            return ProxyEventStart(type="start")
        case "text_start":
            return ProxyEventTextStart(type="text_start", content_index=raw["contentIndex"])
        case "text_delta":
            return ProxyEventTextDelta(
                type="text_delta",
                content_index=raw["contentIndex"],
                delta=raw["delta"],
            )
        case "text_end":
            return ProxyEventTextEnd(
                type="text_end",
                content_index=raw["contentIndex"],
                content_signature=raw.get("contentSignature"),
            )
        case "thinking_start":
            return ProxyEventThinkingStart(type="thinking_start", content_index=raw["contentIndex"])
        case "thinking_delta":
            return ProxyEventThinkingDelta(
                type="thinking_delta",
                content_index=raw["contentIndex"],
                delta=raw["delta"],
            )
        case "thinking_end":
            return ProxyEventThinkingEnd(
                type="thinking_end",
                content_index=raw["contentIndex"],
                content_signature=raw.get("contentSignature"),
            )
        case "toolcall_start":
            return ProxyEventToolcallStart(
                type="toolcall_start",
                content_index=raw["contentIndex"],
                id=raw["id"],
                tool_name=raw["toolName"],
            )
        case "toolcall_delta":
            return ProxyEventToolcallDelta(
                type="toolcall_delta",
                content_index=raw["contentIndex"],
                delta=raw["delta"],
            )
        case "toolcall_end":
            return ProxyEventToolcallEnd(type="toolcall_end", content_index=raw["contentIndex"])
        case "done":
            return ProxyEventDone(
                type="done",
                reason=raw["reason"],
                usage=_parse_usage(usage_data),
            )
        case "error":
            return ProxyEventError(
                type="error",
                reason=raw["reason"],
                error_message=raw.get("errorMessage"),
                usage=_parse_usage(usage_data),
            )
        case _:
            raise RuntimeError(f"Unknown proxy event type: {event_type!r}")


def _parse_usage(data: Any) -> Usage:
    """Deserialize a usage dict from a proxy event into a :class:`Usage`."""
    if not isinstance(data, dict):
        return Usage()
    d: dict[str, Any] = dict(data)  # type: ignore[arg-type]
    return Usage(
        input=d.get("input", 0),
        output=d.get("output", 0),
        cache_read=d.get("cacheRead", 0),
        cache_write=d.get("cacheWrite", 0),
        total_tokens=d.get("totalTokens", 0),
        cost=_parse_usage_cost(d.get("cost")),
    )


def _parse_usage_cost(data: Any) -> UsageCost:
    """Deserialize a usage cost dict from a proxy event."""
    if not isinstance(data, dict):
        return UsageCost()
    d: dict[str, Any] = dict(data)  # type: ignore[arg-type]
    return UsageCost(
        input=d.get("input", 0.0),
        output=d.get("output", 0.0),
        cache_read=d.get("cacheRead", 0.0),
        cache_write=d.get("cacheWrite", 0.0),
        total=d.get("total", 0.0),
    )


# ============================================================================
# stream_proxy
# ============================================================================


def stream_proxy(
    model: Model,
    context: Context,
    options: ProxyStreamOptions,
) -> ProxyMessageEventStream:
    """Stream an LLM response through a proxy server.

    POSTs to ``{proxy_url}/api/stream``, reads the SSE response, and
    reconstructs a partial :class:`AssistantMessage` client-side from
    the compact proxy events.

    Parameters
    ----------
    model:
        The model to use.
    context:
        The conversation context.
    options:
        Proxy stream options (includes ``auth_token``, ``proxy_url``,
        and all :class:`SimpleStreamOptions` fields).

    Returns
    -------
    A :class:`ProxyMessageEventStream` yielding :class:`AssistantMessageEvent`\\ s.

    Example
    -------
    ::

        from otter_agent_core.proxy import stream_proxy, ProxyStreamOptions

        stream = stream_proxy(
            model,
            context,
            ProxyStreamOptions(
                auth_token="...",
                proxy_url="https://genai.example.com",
                temperature=0.7,
            ),
        )
    """
    stream = ProxyMessageEventStream()

    async def _run() -> None:
        # Initialize the partial message that we'll build up from events.
        partial = AssistantMessage(
            role="assistant",
            stop_reason="stop",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=UsageCost(),
            ),
            timestamp=int(time.time() * 1000),
        )

        # Set up abort signal handling.
        abort_event: asyncio.Event | None = None
        if options.signal is not None:
            abort_event = options.signal

        async with httpx.AsyncClient() as client:
            try:
                # Build the request body (camelCase to match upstream wire format).
                body: dict[str, Any] = {
                    "model": {
                        "id": model.id,
                        "name": model.name,
                        "api": model.api,
                        "provider": model.provider,
                        "baseUrl": model.base_url,
                        "reasoning": model.reasoning,
                        "input": model.input,
                        "cost": {
                            "input": model.cost.input,
                            "output": model.cost.output,
                            "cacheRead": model.cost.cache_read,
                            "cacheWrite": model.cost.cache_write,
                        },
                        "contextWindow": model.context_window,
                        "maxTokens": model.max_tokens,
                    },
                    "context": {
                        "systemPrompt": context.system_prompt,
                        "messages": _serialize_messages(context.messages),
                        "tools": _serialize_tools(context.tools),
                    } if context.tools is not None else {
                        "systemPrompt": context.system_prompt,
                        "messages": _serialize_messages(context.messages),
                    },
                    "options": {
                        "temperature": options.temperature,
                        "maxTokens": options.max_tokens,
                        "reasoning": options.reasoning,
                    },
                }

                # Send request.
                response = await client.post(
                    f"{options.proxy_url}/api/stream",
                    headers={
                        "Authorization": f"Bearer {options.auth_token}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )

                if response.status_code != 200:
                    error_message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                    try:
                        raw_error: Any = response.json()
                        if isinstance(raw_error, dict):
                            err_dict: dict[str, Any] = dict(raw_error)  # type: ignore[arg-type]
                            if err_dict.get("error"):
                                error_message = f"Proxy error: {err_dict['error']}"
                    except Exception:
                        pass
                    raise RuntimeError(error_message)

                # Read SSE stream.
                buffer = ""
                async for chunk in response.aiter_text():
                    if abort_event is not None and abort_event.is_set():
                        raise RuntimeError("Request aborted by user")

                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop() or ""

                    for line in lines:
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data:
                                try:
                                    raw_event = json.loads(data)
                                    proxy_event = _parse_proxy_event(raw_event)
                                    event = _process_proxy_event(proxy_event, partial)
                                    if event is not None:
                                        stream.push(event)
                                except json.JSONDecodeError:
                                    pass

                if abort_event is not None and abort_event.is_set():
                    raise RuntimeError("Request aborted by user")

                stream.end()

            except Exception as exc:
                error_msg = exc.args[0] if exc.args else str(exc)
                reason: Literal["aborted", "error"] = (
                    "aborted"
                    if abort_event is not None and abort_event.is_set()
                    else "error"
                )
                partial.stop_reason = reason
                partial.error_message = error_msg
                stream.push(
                    AssistantMessageEventError(
                        type="error",
                        reason=reason,
                        error=partial,
                    ),
                )
                stream.end()

    asyncio.get_event_loop().create_task(_run())
    return stream


# ============================================================================
# Message / Tool serialization helpers
#
# These convert otter-ai Python dataclasses to the camelCase JSON format
# expected by the upstream proxy server wire protocol.
# ============================================================================


def _serialize_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Serialize messages to the upstream camelCase JSON format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        match msg.role:
            case "user":
                result.append({
                    "role": "user",
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                })
            case "assistant":
                content = _serialize_assistant_content(msg.content)
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                    "api": msg.api,
                    "provider": msg.provider,
                    "model": msg.model,
                    "usage": _serialize_usage(msg.usage),
                    "stopReason": msg.stop_reason,
                    "timestamp": msg.timestamp,
                }
                if msg.response_id is not None:
                    entry["responseId"] = msg.response_id
                if msg.error_message is not None:
                    entry["errorMessage"] = msg.error_message
                result.append(entry)
            case "toolResult":
                entry = {
                    "role": "toolResult",
                    "toolCallId": msg.tool_call_id,
                    "toolName": msg.tool_name,
                    "content": _serialize_tool_result_content(msg.content),
                    "isError": msg.is_error,
                    "timestamp": msg.timestamp,
                }
                if msg.details is not None:
                    entry["details"] = msg.details
                result.append(entry)
    return result


def _serialize_assistant_content(content: list[Content]) -> list[dict[str, Any]]:
    """Serialize assistant message content blocks to camelCase."""
    result: list[dict[str, Any]] = []
    for block in content:
        match block.type:
            case "text":
                entry: dict[str, Any] = {"type": "text", "text": block.text}
                if block.text_signature is not None:
                    entry["textSignature"] = block.text_signature
                result.append(entry)
            case "thinking":
                entry = {"type": "thinking", "thinking": block.thinking}
                if block.thinking_signature is not None:
                    entry["thinkingSignature"] = block.thinking_signature
                if block.redacted is not None:
                    entry["redacted"] = block.redacted
                result.append(entry)
            case "toolCall":
                entry = {
                    "type": "toolCall",
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.arguments,
                }
                if block.thought_signature is not None:
                    entry["thoughtSignature"] = block.thought_signature
                result.append(entry)
    return result


def _serialize_tool_result_content(content: list[ToolResultContent]) -> list[dict[str, Any]]:
    """Serialize tool result content blocks to camelCase."""
    result: list[dict[str, Any]] = []
    for block in content:
        match block.type:
            case "text":
                result.append({"type": "text", "text": block.text})
            case "image":
                result.append({"type": "image", "data": block.data, "mimeType": block.mime_type})
    return result


def _serialize_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """Serialize tool definitions to the upstream JSON format."""
    if tools is None:
        return None
    result: list[dict[str, Any]] = []
    for tool in tools:
        result.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        })
    return result


def _serialize_usage(usage: Usage) -> dict[str, Any]:
    """Serialize a :class:`Usage` to the upstream camelCase JSON format."""
    return {
        "input": usage.input,
        "output": usage.output,
        "cacheRead": usage.cache_read,
        "cacheWrite": usage.cache_write,
        "totalTokens": usage.total_tokens,
        "cost": {
            "input": usage.cost.input,
            "output": usage.cost.output,
            "cacheRead": usage.cost.cache_read,
            "cacheWrite": usage.cost.cache_write,
            "total": usage.cost.total,
        },
    }
