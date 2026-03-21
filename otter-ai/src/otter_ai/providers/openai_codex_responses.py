# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""OpenAI Codex Responses provider.

Implements the ``openai-codex-responses`` provider which uses the ChatGPT
backend API (``/codex/responses``) with SSE and WebSocket transports.

Upstream reference: ``packages/ai/src/providers/openai-codex-responses.ts``
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import platform
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from otter_ai.env_api_keys import get_env_api_key
from otter_ai.models import supports_xhigh
from otter_ai.types import (
    AssistantMessage,
    AssistantMessageEventDone,
    AssistantMessageEventError,
    AssistantMessageEventStart,
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    Usage,
    UsageCost,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream

from .openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import build_base_options, clamp_reasoning

# ============================================================================
# Configuration
# ============================================================================

_DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
_JWT_CLAIM_PATH = "https://api.openai.com/auth"
_MAX_RETRIES = 3
_BASE_DELAY_MS = 1000
_CODEX_TOOL_CALL_PROVIDERS = frozenset(["openai", "openai-codex", "opencode"])

_CODEX_RESPONSE_STATUSES = frozenset(
    [
        "completed",
        "incomplete",
        "failed",
        "cancelled",
        "queued",
        "in_progress",
    ]
)

_OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
_SESSION_WEBSOCKET_CACHE_TTL_MS = 5 * 60 * 1000


# ============================================================================
# Options
# ============================================================================


@dataclass
class OpenAICodexResponsesOptions(StreamOptions):
    """Options for the OpenAI Codex Responses provider.

    Upstream: ``interface OpenAICodexResponsesOptions extends StreamOptions``
    """

    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    text_verbosity: str | None = None


# ============================================================================
# Types
# ============================================================================

CodexResponseStatus = str


# ============================================================================
# Retry Helpers
# ============================================================================


def _is_retryable_error(status: int, error_text: str) -> bool:
    """Check if an HTTP error is retryable."""
    if status in (429, 500, 502, 503, 504):
        return True
    return bool(
        re.search(
            r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
            error_text,
            re.IGNORECASE,
        )
    )


async def _sleep(ms: int, signal: asyncio.Event | None = None) -> None:
    """Sleep for *ms* milliseconds, aborting early if *signal* is set."""
    if signal is not None and signal.is_set():
        raise RuntimeError("Request was aborted")
    await asyncio.sleep(ms / 1000)
    if signal is not None and signal.is_set():
        raise RuntimeError("Request was aborted")


def _is_aborted(options: OpenAICodexResponsesOptions | None) -> bool:
    """Check if the request has been aborted via signal."""
    return options is not None and options.signal is not None and options.signal.is_set()


# ============================================================================
# Main Stream Function
# ============================================================================


def stream_openai_codex_responses(
    model: Model,
    context: Context,
    options: OpenAICodexResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the OpenAI Codex Responses API.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → streamOpenAICodexResponses()``
    """
    stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api="openai-codex-responses",
            provider=model.provider,
            model=model.id,
            usage=Usage(cost=UsageCost()),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = (
                (options.api_key if options else None) or get_env_api_key(model.provider) or ""
            )
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")

            account_id = _extract_account_id(api_key)
            body = _build_request_body(model, context, options)
            if options is not None and options.on_payload is not None:
                next_body = await options.on_payload(body, model)
                if next_body is not None:
                    body = next_body

            websocket_request_id = (
                options.session_id if options else None
            ) or _create_codex_request_id()
            sse_headers = _build_sse_headers(
                model.headers,  # type: ignore[arg-type]
                options.headers if options else None,
                account_id,
                api_key,
                options.session_id if options else None,
            )
            websocket_headers = _build_websocket_headers(
                model.headers,  # type: ignore[arg-type]
                options.headers if options else None,
                account_id,
                api_key,
                websocket_request_id,
            )
            body_json = json.dumps(body)
            transport = options.transport if options else "sse"

            if transport != "sse":
                websocket_started = False
                try:
                    await _process_websocket_stream(
                        _resolve_codex_websocket_url(model.base_url),
                        body,
                        websocket_headers,
                        output,
                        stream,
                        model,
                        lambda: None,
                        options,
                    )

                    if _is_aborted(options):
                        raise RuntimeError("Request was aborted")

                    stream.push(
                        AssistantMessageEventDone(
                            type="done",
                            reason=output.stop_reason,
                            message=output,
                        ),
                    )
                    stream.end()
                    return
                except Exception:
                    if transport == "websocket" or websocket_started:
                        raise
                    # Fall through to SSE

            # Fetch with retry logic
            response: Any = None
            last_error: Exception | None = None

            for attempt in range(_MAX_RETRIES + 1):
                if _is_aborted(options):
                    raise RuntimeError("Request was aborted")

                try:
                    import aiohttp

                    url = _resolve_codex_url(model.base_url)
                    timeout = aiohttp.ClientTimeout(total=300)

                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        resp = await session.post(
                            url,
                            headers=sse_headers,
                            data=body_json,
                        )

                    if resp.status == 200:
                        response = resp
                        break

                    error_text = await resp.text()

                    if attempt < _MAX_RETRIES and _is_retryable_error(resp.status, error_text):
                        delay_ms = _BASE_DELAY_MS * 2**attempt
                        await _sleep(delay_ms, options.signal if options else None)
                        continue

                    # Parse error for friendly message on final attempt or non-retryable
                    info = await _parse_error_response(error_text, resp.status)
                    err_msg = info.get("friendly_message") or info.get("message", "Request failed")
                    raise RuntimeError(err_msg)

                except RuntimeError:
                    raise
                except Exception as exc:
                    if _is_aborted(options):
                        raise RuntimeError("Request was aborted") from exc

                    last_error = exc
                    error_msg = str(exc)
                    # Network errors are retryable
                    if attempt < _MAX_RETRIES and "usage limit" not in error_msg:
                        delay_ms = _BASE_DELAY_MS * 2**attempt
                        await _sleep(delay_ms, options.signal if options else None)
                        continue
                    raise

            if response is None or response.status != 200:
                raise last_error or RuntimeError("Failed after retries")

            if not hasattr(response, "read"):
                raise RuntimeError("No response body")

            stream.push(AssistantMessageEventStart(type="start", partial=output))
            await _process_stream(response, output, stream, model)

            if _is_aborted(options):
                raise RuntimeError("Request was aborted")

            stream.push(
                AssistantMessageEventDone(
                    type="done",
                    reason=output.stop_reason,
                    message=output,
                ),
            )
            stream.end()

        except Exception as error:
            output.stop_reason = (
                "aborted"
                if options is not None and options.signal is not None and options.signal.is_set()
                else "error"
            )
            output.error_message = str(error)
            stream.push(
                AssistantMessageEventError(
                    type="error",
                    reason=output.stop_reason,
                    error=output,
                ),
            )
            stream.end()

    asyncio.create_task(_run())
    return stream


def stream_simple_openai_codex_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a Codex Responses response using :class:`SimpleStreamOptions`.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → streamSimpleOpenAICodexResponses()``
    """
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)
    reasoning_effort = (
        options.reasoning
        if options and supports_xhigh(model)
        else clamp_reasoning(options.reasoning if options else None)
    )

    base_dict = {
        "temperature": base.temperature,
        "max_tokens": base.max_tokens,
        "signal": base.signal,
        "api_key": base.api_key,
        "transport": base.transport,
        "cache_retention": base.cache_retention,
        "session_id": base.session_id,
        "on_payload": base.on_payload,
        "headers": base.headers,
        "max_retry_delay_ms": base.max_retry_delay_ms,
        "metadata": base.metadata,
    }

    return stream_openai_codex_responses(
        model,
        context,
        OpenAICodexResponsesOptions(
            **base_dict,
            reasoning_effort=reasoning_effort,
        ),
    )


# ============================================================================
# Request Building
# ============================================================================


def _build_request_body(
    model: Model,
    context: Context,
    options: OpenAICodexResponsesOptions | None = None,
) -> dict[str, Any]:
    """Build the request body for the Codex Responses API.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → buildRequestBody()``
    """
    messages = convert_responses_messages(
        model,
        context,
        _CODEX_TOOL_CALL_PROVIDERS,
        _ConvertResponsesMessagesOptions(include_system_prompt=False),
    )

    body: dict[str, Any] = {
        "model": model.id,
        "store": False,
        "stream": True,
        "instructions": context.system_prompt,
        "input": messages,
        "text": {"verbosity": options.text_verbosity or "medium" if options else "medium"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": options.session_id if options else None,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    if options is not None and options.temperature is not None:
        body["temperature"] = options.temperature

    if context.tools:
        body["tools"] = convert_responses_tools(
            context.tools,
            _ConvertResponsesToolsOptions(strict=None),
        )

    if options is not None and options.reasoning_effort is not None:
        body["reasoning"] = {
            "effort": _clamp_reasoning_effort(model.id, options.reasoning_effort),
            "summary": options.reasoning_summary or "auto",
        }

    return body


class _ConvertResponsesMessagesOptions:
    """Inline options for convert_responses_messages."""

    def __init__(self, *, include_system_prompt: bool) -> None:
        self.include_system_prompt = include_system_prompt


class _ConvertResponsesToolsOptions:
    """Inline options for convert_responses_tools."""

    def __init__(self, *, strict: bool | None = None) -> None:
        self.strict = strict


def _clamp_reasoning_effort(model_id: str, effort: str) -> str:
    """Clamp reasoning effort based on model-specific constraints.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → clampReasoningEffort()``
    """
    model_id_part = model_id.split("/")[-1] if "/" in model_id else model_id

    # GPT-5.2/5.3/5.4 don't support "minimal" — bump to "low"
    if (
        model_id_part.startswith("gpt-5.2")
        or model_id_part.startswith("gpt-5.3")
        or model_id_part.startswith("gpt-5.4")
    ) and effort == "minimal":
        return "low"

    # GPT-5.1 caps "xhigh" at "high"
    if model_id_part == "gpt-5.1" and effort == "xhigh":
        return "high"

    # GPT-5.1-codex-mini: high/xhigh → high, else → medium
    if model_id_part == "gpt-5.1-codex-mini":
        return "high" if effort in ("high", "xhigh") else "medium"

    return effort


def _resolve_codex_url(base_url: str | None = None) -> str:
    """Resolve the Codex Responses API URL.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → resolveCodexUrl()``
    """
    raw = base_url.strip() if base_url else _DEFAULT_CODEX_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _resolve_codex_websocket_url(base_url: str | None = None) -> str:
    """Resolve the Codex WebSocket URL (https → wss, http → ws).

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → resolveCodexWebSocketUrl()``
    """
    url = _resolve_codex_url(base_url)
    parsed = urlparse(url)
    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    else:
        scheme = parsed.scheme
    return parsed._replace(scheme=scheme).geturl()


# ============================================================================
# SSE Parsing
# ============================================================================


async def _parse_sse(response: Any) -> AsyncGenerator[dict[str, Any], None]:
    """Parse SSE events from an aiohttp response body.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → parseSSE()``
    """
    data = await response.read()
    text = data.decode("utf-8", errors="replace")

    for chunk in text.split("\n\n"):
        if not chunk:
            continue
        data_lines = [line[5:].strip() for line in chunk.split("\n") if line.startswith("data:")]
        if not data_lines:
            continue
        payload = "\n".join(data_lines).strip()
        if payload and payload != "[DONE]":
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                yield json.loads(payload)


async def _map_codex_events(
    events: AsyncGenerator[dict[str, Any], None],
) -> AsyncGenerator[Any, None]:
    """Map raw Codex SSE events to normalized ResponseStreamEvent-like objects.

    Handles error events, normalizes completion status, and yields events
    for ``process_responses_stream`` to consume.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → mapCodexEvents()``
    """

    class _Event:
        """Minimal event proxy that exposes ``.type`` and nested attributes."""

        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data
            self.type: str = data.get("type", "")

        def __getattr__(self, name: str) -> Any:
            return self._data.get(name)

    async for event in events:
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            continue

        if event_type == "error":
            code = event.get("code", "")
            message = event.get("message", "")
            raise RuntimeError(f"Codex error: {message or code or json.dumps(event)}")

        if event_type == "response.failed":
            msg = (
                event.get("response", {}).get("error", {}).get("message")
                if isinstance(event.get("response"), dict)
                else None
            )
            raise RuntimeError(msg or "Codex response failed")

        if event_type in ("response.done", "response.completed", "response.incomplete"):
            response = event.get("response")
            if isinstance(response, dict):
                normalized_status = _normalize_codex_status(response.get("status"))
                if normalized_status is not None:
                    response = {**response, "status": normalized_status}
            yield _Event({**event, "type": "response.completed", "response": response})
            return

        yield _Event(event)


def _normalize_codex_status(status: Any) -> CodexResponseStatus | None:
    """Normalize a Codex response status to a known value."""
    if not isinstance(status, str):
        return None
    return status if status in _CODEX_RESPONSE_STATUSES else None


# ============================================================================
# Response Processing
# ============================================================================


async def _process_stream(
    response: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
) -> None:
    """Process the SSE response body into stream events.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → processStream()``
    """
    await process_responses_stream(
        _map_codex_events(_parse_sse(response)),
        output,
        stream,
        model,
    )


# ============================================================================
# WebSocket Session Caching
# ============================================================================

_websocket_session_cache: dict[str, _CachedWebSocketConnection] = {}


async def _decode_websocket_data(data: Any) -> str | None:
    """Decode WebSocket message data to string.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → decodeWebSocketData()``
    """
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray)):
        result = bytes(data)  # type: ignore[arg-type]
        return result.decode("utf-8", errors="replace")
    if hasattr(data, "read"):
        return data.read().decode("utf-8", errors="replace") if callable(data.read) else None
    return None


class _CachedWebSocketConnection:
    """A cached WebSocket connection with busy/idle state tracking."""

    def __init__(self, socket: Any) -> None:
        self.socket = socket
        self.busy = True
        self.idle_timer: asyncio.TimerHandle | None = None


def _get_websocket_ready_state(socket: Any) -> int | None:
    """Get the WebSocket readyState, or None if unavailable."""
    state = getattr(socket, "state", None)
    if isinstance(state, int):
        return state
    # websockets library uses state constants: 0=CONNECTING, 1=OPEN, etc.
    # The OPEN state mapping depends on the library.
    if hasattr(socket, "OPEN"):
        return 1 if state == socket.OPEN else None
    return None


def _is_websocket_reusable(socket: Any) -> bool:
    """Check if a WebSocket connection is still open and reusable."""
    ready_state = _get_websocket_ready_state(socket)
    if ready_state is None:
        return True  # Assume reusable if state is unavailable
    return ready_state == 1


def _close_websocket_silently(socket: Any, code: int = 1000, reason: str = "done") -> None:
    """Close a WebSocket, ignoring any errors."""
    try:
        close_method = getattr(socket, "close", None)
        if callable(close_method):
            close_method(code, reason)
    except Exception:
        pass


def _schedule_session_websocket_expiry(
    session_id: str,
    entry: _CachedWebSocketConnection,
) -> None:
    """Schedule expiry of a cached WebSocket session connection."""

    def _expire() -> None:
        if entry.busy:
            return
        _close_websocket_silently(entry.socket, 1000, "idle_timeout")
        _websocket_session_cache.pop(session_id, None)

    if entry.idle_timer is not None:
        entry.idle_timer.cancel()
    try:
        loop = asyncio.get_running_loop()
        entry.idle_timer = loop.call_later(
            _SESSION_WEBSOCKET_CACHE_TTL_MS / 1000,
            _expire,
        )
    except RuntimeError:
        pass


async def _connect_websocket(
    url: str,
    headers: dict[str, str],
    signal: asyncio.Event | None = None,
) -> Any:
    """Connect to a WebSocket and wait for it to open.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → connectWebSocket()``
    """
    import websockets

    ws_headers = {k: v for k, v in headers.items() if k.lower() != "openai-beta"}

    socket = await websockets.connect(url, additional_headers=ws_headers)
    return socket


async def _acquire_websocket(
    url: str,
    headers: dict[str, str],
    session_id: str | None,
    signal: asyncio.Event | None = None,
) -> tuple[Any, Any]:
    """Acquire a WebSocket connection, reusing cached sessions when possible.

    Returns ``(socket, release_callback)`` where *release_callback* accepts
    ``keep: bool``.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → acquireWebSocket()``
    """
    if not session_id:
        socket = await _connect_websocket(url, headers, signal)

        def release(keep: bool = True) -> None:  # noqa: ARG001
            _close_websocket_silently(socket)

        return socket, release

    cached = _websocket_session_cache.get(session_id)
    if cached is not None:
        if cached.idle_timer is not None:
            cached.idle_timer.cancel()
            cached.idle_timer = None

        if not cached.busy and _is_websocket_reusable(cached.socket):
            cached.busy = True

            def release_reused(keep: bool = True) -> None:
                if not keep or not _is_websocket_reusable(cached.socket):
                    _close_websocket_silently(cached.socket)
                    _websocket_session_cache.pop(session_id, None)
                    return
                cached.busy = False
                _schedule_session_websocket_expiry(session_id, cached)

            return cached.socket, release_reused

        if cached.busy:
            socket = await _connect_websocket(url, headers, signal)

            def release_busy() -> None:
                _close_websocket_silently(socket)

            return socket, release_busy

        if not _is_websocket_reusable(cached.socket):
            _close_websocket_silently(cached.socket)
            _websocket_session_cache.pop(session_id, None)

    socket = await _connect_websocket(url, headers, signal)
    entry = _CachedWebSocketConnection(socket)
    _websocket_session_cache[session_id] = entry

    def release_new(keep: bool = True) -> None:
        if not keep or not _is_websocket_reusable(entry.socket):
            _close_websocket_silently(entry.socket)
            if entry.idle_timer is not None:
                entry.idle_timer.cancel()
            _websocket_session_cache.pop(session_id, None)
            return
        entry.busy = False
        _schedule_session_websocket_expiry(session_id, entry)

    return socket, release_new


# ============================================================================
# WebSocket Parsing
# ============================================================================


async def _parse_websocket(
    socket: Any,
    signal: asyncio.Event | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Parse WebSocket messages into event dicts.

    Yields parsed JSON events until a completion event is received.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → parseWebSocket()``
    """
    saw_completion = False

    try:
        async for raw_message in socket:
            if signal is not None and signal.is_set():
                raise RuntimeError("Request was aborted")

            text = await _decode_websocket_data(raw_message)
            if not text:
                continue

            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                continue

            event_type = parsed.get("type")
            if not isinstance(event_type, str):
                continue

            if event_type in ("response.completed", "response.done", "response.incomplete"):
                saw_completion = True
                yield parsed
                break

            yield parsed

    finally:
        pass

    if not saw_completion:
        raise RuntimeError("WebSocket stream closed before response.completed")


async def _process_websocket_stream(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    on_start: Any,
    options: OpenAICodexResponsesOptions | None = None,
) -> None:
    """Process a WebSocket stream for Codex Responses.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → processWebSocketStream()``
    """
    socket, release = await _acquire_websocket(
        url,
        headers,
        options.session_id if options else None,
        options.signal if options else None,
    )
    keep_connection = True
    try:
        await socket.send(json.dumps({"type": "response.create", **body}))
        on_start()
        stream.push(AssistantMessageEventStart(type="start", partial=output))

        await process_responses_stream(
            _map_codex_events(_parse_websocket(socket, options.signal if options else None)),
            output,
            stream,
            model,
        )

        if _is_aborted(options):
            keep_connection = False
    except Exception:
        keep_connection = False
        raise
    finally:
        release(keep=keep_connection)


# ============================================================================
# Error Handling
# ============================================================================


async def _parse_error_response(
    raw: str,
    status: int,
) -> dict[str, str]:
    """Parse an error response body into a friendly message.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → parseErrorResponse()``
    """
    message = raw or "Request failed"
    friendly_message: str | None = None

    try:
        parsed = json.loads(raw)
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, dict):
            code = err.get("code") or err.get("type") or ""
            is_usage_limit = bool(
                re.search(
                    r"usage_limit_reached|usage_not_included|rate_limit_exceeded",
                    code,
                    re.IGNORECASE,
                )
            )
            if is_usage_limit or status == 429:
                plan = err.get("plan_type", "")
                plan_text = f" ({plan.lower()} plan)" if plan else ""
                resets_at = err.get("resets_at")
                if resets_at is not None:
                    mins = max(0, round((resets_at * 1000 - time.time() * 1000) / 60000))
                    when = f" Try again in ~{mins} min."
                else:
                    when = ""
                friendly_message = (
                    f"You have hit your ChatGPT usage limit{plan_text}.{when}"
                ).strip()

            message = err.get("message") or friendly_message or message
    except (json.JSONDecodeError, TypeError):
        pass

    result: dict[str, str] = {"message": message}
    if friendly_message:
        result["friendly_message"] = friendly_message
    return result


# ============================================================================
# Auth & Headers
# ============================================================================


def _extract_account_id(token: str) -> str:
    """Extract the ChatGPT account ID from a JWT token.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → extractAccountId()``
    """
    import base64

    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise RuntimeError("Invalid token")
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
        account_id = payload.get(_JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not account_id:
            raise RuntimeError("No account ID in token")
        return account_id
    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as exc:
        raise RuntimeError("Failed to extract accountId from token") from exc


def _create_codex_request_id() -> str:
    """Create a unique Codex request ID.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → createCodexRequestId()``
    """
    return str(uuid.uuid4())


def _build_base_codex_headers(
    init_headers: dict[str, str] | None,
    additional_headers: dict[str, str] | None,
    account_id: str,
    token: str,
) -> dict[str, str]:
    """Build the common Codex API headers.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → buildBaseCodexHeaders()``
    """
    headers: dict[str, str] = {}
    if init_headers:
        headers.update(init_headers)
    if additional_headers:
        headers.update(additional_headers)

    headers["Authorization"] = f"Bearer {token}"
    headers["chatgpt-account-id"] = account_id
    headers["originator"] = "pi"
    user_agent = f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"
    headers["User-Agent"] = user_agent
    return headers


def _build_sse_headers(
    init_headers: dict[str, str] | None,
    additional_headers: dict[str, str] | None,
    account_id: str,
    token: str,
    session_id: str | None = None,
) -> dict[str, str]:
    """Build headers for SSE transport.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → buildSSEHeaders()``
    """
    headers = _build_base_codex_headers(init_headers, additional_headers, account_id, token)
    headers["OpenAI-Beta"] = "responses=experimental"
    headers["accept"] = "text/event-stream"
    headers["content-type"] = "application/json"

    if session_id:
        headers["session_id"] = session_id

    return headers


def _build_websocket_headers(
    init_headers: dict[str, str] | None,
    additional_headers: dict[str, str] | None,
    account_id: str,
    token: str,
    request_id: str,
) -> dict[str, str]:
    """Build headers for WebSocket transport.

    Upstream reference:
    ``packages/ai/src/providers/openai-codex-responses.ts → buildWebSocketHeaders()``
    """
    headers = _build_base_codex_headers(init_headers, additional_headers, account_id, token)
    headers.pop("accept", None)
    headers.pop("content-type", None)
    headers.pop("OpenAI-Beta", None)
    headers.pop("openai-beta", None)
    headers["OpenAI-Beta"] = _OPENAI_BETA_RESPONSES_WEBSOCKETS
    headers["x-client-request-id"] = request_id
    headers["session_id"] = request_id
    return headers
