"""Google Gemini CLI / Antigravity provider.

Shared implementation for both ``google-gemini-cli`` and ``google-antigravity``
providers.  Uses the Cloud Code Assist API endpoint to access Gemini and
Claude models via Server-Sent Events (SSE).

Upstream reference: ``packages/ai/src/providers/google-gemini-cli.ts``
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

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
    ThinkingBudgets,
    ThinkingContent,
    ToolCall,
    Usage,
    UsageCost,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream
from otter_ai.utils.sanitize_unicode import sanitize_surrogates

from .google_shared import (
    convert_messages,
    convert_tools,
    is_thinking_part,
    map_stop_reason_string,
    map_tool_choice,
    retain_thought_signature,
)
from .simple_options import build_base_options, clamp_reasoning

# ============================================================================
# Provider options
# ============================================================================

GoogleThinkingLevel = Literal[
    "THINKING_LEVEL_UNSPECIFIED",
    "MINIMAL",
    "LOW",
    "MEDIUM",
    "HIGH",
]


@dataclass
class GoogleGeminiCliOptions(StreamOptions):
    """Options for the Google Gemini CLI / Antigravity provider."""

    tool_choice: Literal["auto", "none", "any"] | None = None
    thinking: dict[str, Any] | None = None
    project_id: str | None = None


# ============================================================================
# Constants
# ============================================================================

_DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
_ANTIGRAVITY_DAILY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
_ANTIGRAVITY_AUTOPUSH_ENDPOINT = "https://autopush-cloudcode-pa.sandbox.googleapis.com"
_ANTIGRAVITY_ENDPOINT_FALLBACKS = [
    _ANTIGRAVITY_DAILY_ENDPOINT,
    _ANTIGRAVITY_AUTOPUSH_ENDPOINT,
    _DEFAULT_ENDPOINT,
]

_GEMINI_CLI_HEADERS: dict[str, str] = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": json.dumps(
        {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    ),
}

_DEFAULT_ANTIGRAVITY_VERSION = "1.18.4"
_MAX_RETRIES = 3
_BASE_DELAY_MS = 1000
_MAX_EMPTY_STREAM_RETRIES = 2
_EMPTY_STREAM_BASE_DELAY_MS = 500
_CLAUDE_THINKING_BETA_HEADER = "interleaved-thinking-2025-05-14"

_ANTIGRAVITY_SYSTEM_INSTRUCTION = (
    "You are Antigravity, a powerful agentic AI coding assistant designed by the "
    "Google Deepmind team working on Advanced Agentic Coding."
    "You are pair programming with a USER to solve their coding task. The task may "
    "require creating a new codebase, modifying or debugging an existing codebase, "
    "or simply answering a question."
    "**Absolute paths only**"
    "**Proactiveness**"
)

_tool_call_counter = 0


# ============================================================================
# Helpers
# ============================================================================


def _get_antigravity_headers() -> dict[str, str]:
    version = os.environ.get("PI_AI_ANTIGRAVITY_VERSION", _DEFAULT_ANTIGRAVITY_VERSION)
    return {"User-Agent": f"antigravity/{version} darwin/arm64"}


def _needs_claude_thinking_beta_header(model: Model) -> bool:
    return (
        model.provider == "google-antigravity"
        and model.id.startswith("claude-")
        and model.reasoning
    )


def _is_gemini_3_pro_model(model_id: str) -> bool:
    return bool(re.search(r"gemini-3(?:\.1)?-pro", model_id.lower()))


def _is_gemini_3_flash_model(model_id: str) -> bool:
    return bool(re.search(r"gemini-3(?:\.1)?-flash", model_id.lower()))


def _is_gemini_3_model(model_id: str) -> bool:
    return _is_gemini_3_pro_model(model_id) or _is_gemini_3_flash_model(model_id)


# ============================================================================
# Retry delay extraction
# ============================================================================


def extract_retry_delay(error_text: str, response: Any = None) -> int | None:
    """Extract retry delay from Gemini error response (in milliseconds).

    Checks headers first (``Retry-After``, ``x-ratelimit-reset``,
    ``x-ratelimit-reset-after``), then parses body patterns like:

    - ``"Your quota will reset after 39s"``
    - ``"Your quota will reset after 18h31m10s"``
    - ``"Please retry in Xs"`` or ``"Please retry in Xms"``
    - ``"retryDelay": "34.074824224s"`` (JSON field)

    Upstream reference: ``packages/ai/src/providers/google-gemini-cli.ts → extractRetryDelay()``
    """

    def _normalize(ms: float) -> int | None:
        return math.ceil(ms + 1000) if ms > 0 else None

    headers = response
    if headers is not None and hasattr(headers, "get"):
        retry_after = headers.get("retry-after")
        if retry_after:
            retry_after_s = float(retry_after)
            if math.isfinite(retry_after_s):
                delay = _normalize(retry_after_s * 1000)
                if delay is not None:
                    return delay
            # Try HTTP date
            from email.utils import parsedate_to_datetime

            try:
                dt: Any = parsedate_to_datetime(retry_after)
                delay = _normalize(dt.timestamp() * 1000 - time.time() * 1000)  # type: ignore[unknownMemberType, unknownVariableType]
                if delay is not None:
                    return delay
            except (ValueError, TypeError):
                pass

        rate_limit_reset = headers.get("x-ratelimit-reset")
        if rate_limit_reset:
            reset_s = int(rate_limit_reset)
            delay = _normalize(reset_s * 1000 - time.time() * 1000)
            if delay is not None:
                return delay

        rate_limit_reset_after = headers.get("x-ratelimit-reset-after")
        if rate_limit_reset_after:
            reset_after_s = float(rate_limit_reset_after)
            if math.isfinite(reset_after_s):
                delay = _normalize(reset_after_s * 1000)
                if delay is not None:
                    return delay

    # Pattern 1: "Your quota will reset after 18h31m10s"
    m = re.search(r"reset after (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", error_text, re.IGNORECASE)
    if m:
        hours = int(m.group(1)) if m.group(1) else 0
        minutes = int(m.group(2)) if m.group(2) else 0
        seconds = float(m.group(3))
        if not math.isnan(seconds):
            total_ms = ((hours * 60 + minutes) * 60 + seconds) * 1000
            delay = _normalize(total_ms)
            if delay is not None:
                return delay

    # Pattern 2: "Please retry in X[ms|s]"
    m = re.search(r"Please retry in ([0-9.]+)(ms|s)", error_text, re.IGNORECASE)
    if m and m.group(1):
        value = float(m.group(1))
        if not math.isnan(value) and value > 0:
            ms = value if m.group(2).lower() == "ms" else value * 1000
            delay = _normalize(ms)
            if delay is not None:
                return delay

    # Pattern 3: "retryDelay": "34.074824224s"
    m = re.search(r'"retryDelay":\s*"([0-9.]+)(ms|s)"', error_text, re.IGNORECASE)
    if m and m.group(1):
        value = float(m.group(1))
        if not math.isnan(value) and value > 0:
            ms = value if m.group(2).lower() == "ms" else value * 1000
            delay = _normalize(ms)
            if delay is not None:
                return delay

    return None


# ============================================================================
# Error helpers
# ============================================================================


def _is_retryable_error(status: int, error_text: str) -> bool:
    if status in (429, 500, 502, 503, 504):
        return True
    return bool(
        re.search(
            r"resource.?exhausted|rate.?limit|overloaded|service.?unavailable|other.?side.?closed",
            error_text,
            re.IGNORECASE,
        )
    )


def _extract_error_message(error_text: str) -> str:
    try:
        parsed = json.loads(error_text)
        msg = parsed.get("error", {}).get("message")
        if msg:
            return msg
    except (json.JSONDecodeError, AttributeError):
        pass
    return error_text


# ============================================================================
# Request builder
# ============================================================================


def build_request(
    model: Model,
    context: Context,
    project_id: str,
    options: GoogleGeminiCliOptions | None = None,
    is_antigravity: bool = False,
) -> dict[str, Any]:
    """Build the Cloud Code Assist API request body.

    Upstream reference: ``packages/ai/src/providers/google-gemini-cli.ts → buildRequest()``
    """
    options = options or GoogleGeminiCliOptions()
    contents = convert_messages(model, context)

    generation_config: dict[str, Any] = {}
    if options.temperature is not None:
        generation_config["temperature"] = options.temperature
    if options.max_tokens is not None:
        generation_config["maxOutputTokens"] = options.max_tokens

    # Thinking config
    if options.thinking and options.thinking.get("enabled") and model.reasoning:
        generation_config["thinkingConfig"] = {"includeThoughts": True}
        level = options.thinking.get("level")
        budget = options.thinking.get("budgetTokens")
        if level is not None:
            generation_config["thinkingConfig"]["thinkingLevel"] = level
        elif budget is not None:
            generation_config["thinkingConfig"]["thinkingBudget"] = budget

    request: dict[str, Any] = {"contents": contents}
    request["sessionId"] = options.session_id

    if context.system_prompt:
        request["systemInstruction"] = {
            "parts": [{"text": sanitize_surrogates(context.system_prompt)}],
        }

    if generation_config:
        request["generationConfig"] = generation_config

    if context.tools and len(context.tools) > 0:
        use_parameters = model.id.startswith("claude-")
        request["tools"] = convert_tools(context.tools, use_parameters)
        if options.tool_choice:
            request["toolConfig"] = {
                "functionCallingConfig": {"mode": map_tool_choice(options.tool_choice)},
            }

    if is_antigravity:
        existing_parts = request.get("systemInstruction", {}).get("parts", [])
        request["systemInstruction"] = {
            "role": "user",
            "parts": [
                {"text": _ANTIGRAVITY_SYSTEM_INSTRUCTION},
                {
                    "text": (
                        "Please ignore following "
                        f"[ignore]{_ANTIGRAVITY_SYSTEM_INSTRUCTION}[/ignore]"
                    ),
                },
                *existing_parts,
            ],
        }

    result: dict[str, Any] = {
        "project": project_id,
        "model": model.id,
        "request": request,
        "userAgent": "antigravity" if is_antigravity else "pi-coding-agent",
        "requestId": (
            f"{'agent' if is_antigravity else 'pi'}-{int(time.time() * 1000)}"
            f"-{__import__('random').random():.10f}"[:30]
        ),
    }
    if is_antigravity:
        result["requestType"] = "agent"

    return result


# ============================================================================
# Thinking level mapping
# ============================================================================

ClampedThinkingLevel = Literal["minimal", "low", "medium", "high"]


def _get_gemini_cli_thinking_level(
    effort: ClampedThinkingLevel,
    model_id: str,
) -> GoogleThinkingLevel:
    if _is_gemini_3_pro_model(model_id):
        match effort:
            case "minimal" | "low":
                return "LOW"
            case "medium" | "high":
                return "HIGH"
    match effort:
        case "minimal":
            return "MINIMAL"
        case "low":
            return "LOW"
        case "medium":
            return "MEDIUM"
        case "high":
            return "HIGH"


# ============================================================================
# SSE stream parsing
# ============================================================================


async def _stream_response(
    response: Any,
    model: Model,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    options: GoogleGeminiCliOptions | None,
    started: list[bool],
) -> bool:
    """Parse SSE events from the response body.

    Returns True if any content was received.
    """
    global _tool_call_counter  # noqa: PLW0603

    async def _ensure_started() -> None:
        if not started[0]:
            stream.push(AssistantMessageEventStart(type="start", partial=output))
            started[0] = True

    has_content = False
    current_block: TextContent | ThinkingContent | None = None
    blocks = output.content

    def block_index() -> int:
        return len(blocks) - 1

    # Read SSE stream
    raw: bytes = await response.read()  # type: ignore[unknownVariableType]
    text = raw.decode("utf-8", errors="replace")

    for line in text.split("\n"):
        if not line.startswith("data:"):
            continue

        json_str = line[5:].strip()
        if not json_str:
            continue

        try:
            chunk: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        response_data: dict[str, Any] = chunk.get("response") or {}
        if not response_data:
            continue

        # Keep first non-empty responseId
        if not output.response_id:
            output.response_id = response_data.get("responseId")

        candidates: list[dict[str, Any]] = response_data.get("candidates") or []
        if not candidates:
            continue
        candidate = candidates[0]

        parts: list[dict[str, Any]] = (candidate.get("content") or {}).get("parts") or []
        for part in parts:
            part_text: Any = part.get("text")
            if part_text is not None:
                has_content = True
                is_thinking = is_thinking_part(type("P", (), part)())

                if (
                    current_block is None
                    or (is_thinking and current_block.type != "thinking")
                    or (not is_thinking and current_block.type != "text")
                ):
                    if current_block is not None:
                        if current_block.type == "text":
                            stream.push(
                                AssistantMessageEventTextEnd(
                                    type="text_end",
                                    content_index=len(blocks) - 1,
                                    content=current_block.text,
                                    partial=output,
                                ),
                            )
                        else:
                            stream.push(
                                AssistantMessageEventThinkingEnd(
                                    type="thinking_end",
                                    content_index=block_index(),
                                    content=current_block.thinking,
                                    partial=output,
                                ),
                            )
                    if is_thinking:
                        current_block = ThinkingContent(
                            type="thinking",
                            thinking="",
                            thinking_signature=None,
                        )
                        blocks.append(current_block)
                        await _ensure_started()
                        stream.push(
                            AssistantMessageEventThinkingStart(
                                type="thinking_start",
                                content_index=block_index(),
                                partial=output,
                            ),
                        )
                    else:
                        current_block = TextContent(type="text", text="")
                        blocks.append(current_block)
                        await _ensure_started()
                        stream.push(
                            AssistantMessageEventTextStart(
                                type="text_start",
                                content_index=block_index(),
                                partial=output,
                            ),
                        )

                if current_block.type == "thinking":
                    current_block.thinking += part_text
                    current_block.thinking_signature = retain_thought_signature(
                        current_block.thinking_signature,
                        part.get("thoughtSignature"),
                    )
                    stream.push(
                        AssistantMessageEventThinkingDelta(
                            type="thinking_delta",
                            content_index=block_index(),
                            delta=part_text,
                            partial=output,
                        ),
                    )
                else:
                    current_block.text += part_text
                    current_block.text_signature = retain_thought_signature(
                        current_block.text_signature,
                        part.get("thoughtSignature"),
                    )
                    stream.push(
                        AssistantMessageEventTextDelta(
                            type="text_delta",
                            content_index=block_index(),
                            delta=part_text,
                            partial=output,
                        ),
                    )

            func_call = part.get("functionCall")
            if func_call:
                has_content = True
                if current_block is not None:
                    if current_block.type == "text":
                        stream.push(
                            AssistantMessageEventTextEnd(
                                type="text_end",
                                content_index=block_index(),
                                content=current_block.text,
                                partial=output,
                            ),
                        )
                    else:
                        stream.push(
                            AssistantMessageEventThinkingEnd(
                                type="thinking_end",
                                content_index=block_index(),
                                content=current_block.thinking,
                                partial=output,
                            ),
                        )
                    current_block = None

                provided_id = func_call.get("id")
                needs_new_id = not provided_id or any(
                    b.type == "toolCall" and b.id == provided_id for b in blocks
                )
                if needs_new_id:
                    _tool_call_counter += 1
                    ts = int(time.time() * 1000)
                    tool_call_id = (
                        f"{func_call.get('name', '')}_{ts}_{_tool_call_counter}"
                    )
                else:
                    tool_call_id = provided_id

                tool_call = ToolCall(
                    type="toolCall",
                    id=tool_call_id,
                    name=func_call.get("name", ""),
                    arguments=func_call.get("args") or {},
                )
                thought_sig = part.get("thoughtSignature")
                if thought_sig:
                    tool_call.thought_signature = thought_sig

                blocks.append(tool_call)
                await _ensure_started()
                stream.push(
                    AssistantMessageEventToolcallStart(
                        type="toolcall_start",
                        content_index=block_index(),
                        partial=output,
                    ),
                )
                stream.push(
                    AssistantMessageEventToolcallDelta(
                        type="toolcall_delta",
                        content_index=block_index(),
                        delta=json.dumps(tool_call.arguments),
                        partial=output,
                    ),
                )
                stream.push(
                    AssistantMessageEventToolcallEnd(
                        type="toolcall_end",
                        content_index=block_index(),
                        tool_call=tool_call,
                        partial=output,
                    ),
                )

        finish_reason = candidate.get("finishReason")
        if finish_reason:
            output.stop_reason = map_stop_reason_string(finish_reason)
            if any(b.type == "toolCall" for b in blocks):
                output.stop_reason = "toolUse"

        usage_meta = response_data.get("usageMetadata")
        if usage_meta:
            prompt_tokens = usage_meta.get("promptTokenCount", 0) or 0
            cache_read_tokens = usage_meta.get("cachedContentTokenCount", 0) or 0
            candidates_tokens = usage_meta.get("candidatesTokenCount", 0) or 0
            thoughts_tokens = usage_meta.get("thoughtsTokenCount", 0) or 0
            total_tokens = usage_meta.get("totalTokenCount", 0) or 0

            output.usage = Usage(
                input=prompt_tokens - cache_read_tokens,
                output=candidates_tokens + thoughts_tokens,
                cache_read=cache_read_tokens,
                cache_write=0,
                total_tokens=total_tokens,
                cost=UsageCost(),
            )
            calculate_cost(model, output.usage)

    # Close final block
    if current_block is not None:
        if current_block.type == "text":
            stream.push(
                AssistantMessageEventTextEnd(
                    type="text_end",
                    content_index=block_index(),
                    content=current_block.text,
                    partial=output,
                ),
            )
        else:
            stream.push(
                AssistantMessageEventThinkingEnd(
                    type="thinking_end",
                    content_index=block_index(),
                    content=current_block.thinking,
                    partial=output,
                ),
            )

    return has_content


# ============================================================================
# Main stream function
# ============================================================================


def stream_google_gemini_cli(
    model: Model,
    context: Context,
    options: GoogleGeminiCliOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the Google Cloud Code Assist API.

    Upstream reference: ``packages/ai/src/providers/google-gemini-cli.ts → streamGoogleGeminiCli()``
    """
    stream = AssistantMessageEventStream()
    options = options or GoogleGeminiCliOptions()

    async def _run() -> None:
        global _tool_call_counter

        output = AssistantMessage(
            role="assistant",
            content=[],
            api="google-gemini-cli",
            provider=model.provider,
            model=model.id,
            usage=Usage(cost=UsageCost()),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            import aiohttp

            api_key_raw = options.api_key
            if not api_key_raw:
                raise RuntimeError(
                    "Google Cloud Code Assist requires OAuth authentication. "
                    "Use /login to authenticate.",
                )

            try:
                parsed = json.loads(api_key_raw)
                access_token: str = parsed["token"]
                project_id: str = parsed["projectId"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    "Invalid Google Cloud Code Assist credentials. "
                    "Use /login to re-authenticate.",
                ) from exc

            if not access_token or not project_id:
                raise RuntimeError(
                    "Missing token or projectId in Google Cloud credentials. "
                    "Use /login to re-authenticate.",
                )

            is_antigravity = model.provider == "google-antigravity"
            base_url = (model.base_url or "").strip()
            endpoints = (
                [base_url]
                if base_url
                else list(_ANTIGRAVITY_ENDPOINT_FALLBACKS)
                if is_antigravity
                else [_DEFAULT_ENDPOINT]
            )

            request_body = build_request(model, context, project_id, options, is_antigravity)
            if options.on_payload is not None:
                next_body = options.on_payload(request_body, model)
                if next_body is not None:
                    request_body = next_body

            headers = _get_antigravity_headers() if is_antigravity else _GEMINI_CLI_HEADERS
            request_headers: dict[str, str] = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **headers,
            }
            if _needs_claude_thinking_beta_header(model):
                request_headers["anthropic-beta"] = _CLAUDE_THINKING_BETA_HEADER
            if options.headers:
                request_headers.update(options.headers)

            request_body_json = json.dumps(request_body)

            # Fetch with retry logic
            response: aiohttp.ClientResponse | None = None
            last_error: Exception | None = None
            request_url: str | None = None
            endpoint_index = 0

            for attempt in range(_MAX_RETRIES + 1):
                if options.signal is not None and options.signal.is_set():
                    raise RuntimeError("Request was aborted")

                try:
                    endpoint = endpoints[endpoint_index]
                    request_url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"

                    timeout = aiohttp.ClientTimeout(total=300)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        resp = await session.post(
                            request_url,
                            headers=request_headers,
                            data=request_body_json,
                        )

                    if resp.status == 200:
                        response = resp
                        break

                    error_text = await resp.text()

                    # On 403/404, cascade to next endpoint immediately
                    if resp.status in (403, 404) and endpoint_index < len(endpoints) - 1:
                        endpoint_index += 1
                        continue

                    if attempt < _MAX_RETRIES and _is_retryable_error(resp.status, error_text):
                        if endpoint_index < len(endpoints) - 1:
                            endpoint_index += 1
                        server_delay = extract_retry_delay(error_text, resp.headers)
                        delay_ms = (
                            server_delay
                            if server_delay is not None
                            else _BASE_DELAY_MS * 2**attempt
                        )

                        max_delay_ms = (
                            options.max_retry_delay_ms
                            if options.max_retry_delay_ms is not None
                            else 60000
                        )
                        if (
                            max_delay_ms > 0
                            and server_delay is not None
                            and server_delay > max_delay_ms
                        ):
                            delay_s = math.ceil(server_delay / 1000)
                            max_s = math.ceil(max_delay_ms / 1000)
                            raise RuntimeError(
                                f"Server requested {delay_s}s retry delay (max: {max_s}s). "
                                f"{_extract_error_message(error_text)}",
                            )

                        await asyncio.sleep(delay_ms / 1000)
                        continue

                    raise RuntimeError(
                        f"Cloud Code Assist API error ({resp.status}): "
                        f"{_extract_error_message(error_text)}",
                    )

                except RuntimeError:
                    raise
                except Exception as exc:
                    if options.signal is not None and options.signal.is_set():
                        raise RuntimeError("Request was aborted") from exc

                    last_error = exc
                    if attempt < _MAX_RETRIES:
                        delay_ms = _BASE_DELAY_MS * 2**attempt
                        await asyncio.sleep(delay_ms / 1000)
                        continue
                    raise

            if response is None or response.status != 200:
                raise last_error or RuntimeError("Failed to get response after retries")

            started = [False]

            async def reset_output() -> None:
                output.content = []
                output.usage = Usage(cost=UsageCost())
                output.stop_reason = "stop"
                output.error_message = None
                output.timestamp = int(time.time() * 1000)
                started[0] = False

            received_content = False

            for empty_attempt in range(_MAX_EMPTY_STREAM_RETRIES + 1):
                if options.signal is not None and options.signal.is_set():
                    raise RuntimeError("Request was aborted")

                if empty_attempt > 0:
                    backoff_ms = _EMPTY_STREAM_BASE_DELAY_MS * 2 ** (empty_attempt - 1)
                    await asyncio.sleep(backoff_ms / 1000)

                    if not request_url:
                        raise RuntimeError("Missing request URL")

                    timeout = aiohttp.ClientTimeout(total=300)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        current_resp = await session.post(
                            request_url,
                            headers=request_headers,
                            data=request_body_json,
                        )
                    if current_resp.status != 200:
                        retry_error_text = await current_resp.text()
                        raise RuntimeError(
                            f"Cloud Code Assist API error ({current_resp.status}): "
                            f"{retry_error_text}",
                        )
                    streamed = await _stream_response(
                        current_resp,
                        model,
                        output,
                        stream,
                        options,
                        started,
                    )
                else:
                    streamed = await _stream_response(
                        response,
                        model,
                        output,
                        stream,
                        options,
                        started,
                    )

                if streamed:
                    received_content = True
                    break

                if empty_attempt < _MAX_EMPTY_STREAM_RETRIES:
                    await reset_output()

            if not received_content:
                raise RuntimeError("Cloud Code Assist API returned an empty response")

            if options.signal is not None and options.signal.is_set():
                raise RuntimeError("Request was aborted")

            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError("An unknown error occurred")

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
                "aborted" if options.signal is not None and options.signal.is_set() else "error"
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


# ============================================================================
# Simple stream function
# ============================================================================


def stream_simple_google_gemini_cli(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a Gemini CLI response using :class:`SimpleStreamOptions`.

    Upstream reference:
    ``packages/ai/src/providers/google-gemini-cli.ts → streamSimpleGoogleGeminiCli()``
    """
    api_key = options.api_key if options else None
    if not api_key:
        raise RuntimeError(
            "Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate.",
        )

    base = build_base_options(model, options, api_key)

    if options is None or not options.reasoning:
        return stream_google_gemini_cli(
            model,
            context,
            GoogleGeminiCliOptions(
                **{k: v for k, v in base.__dict__.items() if not k.startswith("_")},
                thinking={"enabled": False},
            ),
        )

    effort = clamp_reasoning(options.reasoning)
    assert effort is not None
    clamped_effort: ClampedThinkingLevel = effort  # type: ignore[assignment]

    base_dict = {k: v for k, v in base.__dict__.items() if not k.startswith("_")}  # type: ignore[dict-item]

    if _is_gemini_3_model(model.id):
        return stream_google_gemini_cli(
            model,
            context,
            GoogleGeminiCliOptions(
                **base_dict,
                thinking={
                    "enabled": True,
                    "level": _get_gemini_cli_thinking_level(clamped_effort, model.id),
                },
            ),
        )

    default_budgets = ThinkingBudgets(
        minimal=1024,
        low=2048,
        medium=8192,
        high=16384,
    )
    budgets = ThinkingBudgets(
        minimal=default_budgets.get("minimal"),
        low=default_budgets.get("low"),
        medium=default_budgets.get("medium"),
        high=default_budgets.get("high"),
    )
    if options.thinking_budgets:
        budgets = ThinkingBudgets(
            minimal=options.thinking_budgets.get("minimal", budgets.get("minimal")),
            low=options.thinking_budgets.get("low", budgets.get("low")),
            medium=options.thinking_budgets.get("medium", budgets.get("medium")),
            high=options.thinking_budgets.get("high", budgets.get("high")),
        )

    min_output_tokens = 1024
    thinking_budget = budgets.get(clamped_effort, 16384)  # type: ignore[union-attr]
    max_tokens = min((base.max_tokens or 0) + thinking_budget, model.max_tokens)

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return stream_google_gemini_cli(
        model,
        context,
        GoogleGeminiCliOptions(
            **base_dict,
            max_tokens=max_tokens,
            thinking={
                "enabled": True,
                "budgetTokens": thinking_budget,
            },
        ),
    )
