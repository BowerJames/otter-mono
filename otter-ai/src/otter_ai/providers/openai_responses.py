# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false

"""OpenAI Responses API provider.

Upstream: packages/ai/src/providers/openai-responses.ts (~262 lines)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import openai

from otter_ai.env_api_keys import get_env_api_key
from otter_ai.models import supports_xhigh
from otter_ai.types import (
    AssistantMessage,
    AssistantMessageEventDone,
    AssistantMessageEventError,
    AssistantMessageEventStart,
    Usage,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream
from .github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from .openai_responses_shared import (
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import build_base_options, clamp_reasoning

_OPENAI_TOOL_CALL_PROVIDERS = frozenset(["openai", "openai-codex", "opencode"])


# ============================================================================
# Cache retention
# ============================================================================


def _resolve_cache_retention(cache_retention: str | None) -> str:
    if cache_retention:
        return cache_retention
    if os.environ.get("OTTER_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def _get_prompt_cache_retention(base_url: str | None, cache_retention: str) -> str | None:
    if cache_retention != "long":
        return None
    if base_url and "api.openai.com" in base_url:
        return "24h"
    return None


# ============================================================================
# Options
# ============================================================================


class OpenAIResponsesOptions:
    """Options for OpenAI Responses API streaming."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        signal: Any | None = None,
        headers: dict[str, str] | None = None,
        on_payload: Any | None = None,
        metadata: dict[str, Any] | None = None,
        cache_retention: str | None = None,
        session_id: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        service_tier: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.signal = signal
        self.headers = headers
        self.on_payload = on_payload
        self.metadata = metadata
        self.cache_retention = cache_retention
        self.session_id = session_id
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.service_tier = service_tier


# ============================================================================
# Client creation
# ============================================================================


def _create_client(
    model: Any,
    context: Any,
    api_key: str | None = None,
    options_headers: dict[str, str] | None = None,
) -> openai.OpenAI:
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument."
        )

    headers: dict[str, str] = {}
    if hasattr(model, "headers") and model.headers:
        headers.update(model.headers)

    if model.provider == "github-copilot":
        has_images = has_copilot_vision_input(context.messages)
        copilot_headers = build_copilot_dynamic_headers(
            messages=context.messages,
            has_images=has_images,
        )
        headers.update(copilot_headers)

    if options_headers:
        headers.update(options_headers)

    return openai.OpenAI(
        api_key=api_key,
        base_url=getattr(model, "base_url", None),
        default_headers=headers,
    )


# ============================================================================
# Build params
# ============================================================================


def _build_params(
    model: Any,
    context: Any,
    options: OpenAIResponsesOptions | None = None,
) -> dict[str, Any]:
    messages = convert_responses_messages(model, context, _OPENAI_TOOL_CALL_PROVIDERS)

    cache_retention = _resolve_cache_retention(
        options.cache_retention if options else None
    )
    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }

    if cache_retention != "none" and options and options.session_id:
        params["prompt_cache_key"] = options.session_id
    prompt_cache = _get_prompt_cache_retention(
        getattr(model, "base_url", None), cache_retention,
    )
    if prompt_cache:
        params["prompt_cache_retention"] = prompt_cache

    if options and options.max_tokens:
        params["max_output_tokens"] = options.max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if options and options.service_tier:
        params["service_tier"] = options.service_tier

    if context.tools:
        params["tools"] = convert_responses_tools(context.tools)

    if getattr(model, "reasoning", False):
        if (options and options.reasoning_effort) or (options and options.reasoning_summary):
            params["reasoning"] = {
                "effort": options.reasoning_effort or "medium",
                "summary": options.reasoning_summary or "auto",
            }
            params["include"] = ["reasoning.encrypted_content"]
        else:
            model_name = getattr(model, "name", "") or ""
            if model_name.lower().startswith("gpt-5"):
                messages.append({
                    "role": "developer",
                    "content": [{
                        "type": "input_text",
                        "text": "# Juice: 0 !important",
                    }],
                })

    return params


# ============================================================================
# Service tier pricing
# ============================================================================


def _get_service_tier_cost_multiplier(service_tier: str | None) -> float:
    match service_tier:
        case "flex":
            return 0.5
        case "priority":
            return 2.0
        case _:
            return 1.0


def _apply_service_tier_pricing(
    usage: Usage,
    service_tier: str | None,
) -> None:
    multiplier = _get_service_tier_cost_multiplier(service_tier)
    if multiplier == 1.0:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = (
        usage.cost.input + usage.cost.output
        + usage.cost.cache_read + usage.cost.cache_write
    )


# ============================================================================
# Streaming
# ============================================================================


def stream_openai_responses(
    model: Any,
    context: Any,
    options: OpenAIResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream completions from the OpenAI Responses API.

    Upstream: openai-responses.ts → streamOpenAIResponses()
    """
    stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api="openai-responses",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = (options.api_key if options else None) or get_env_api_key(model.provider) or ""
            client = _create_client(model, context, api_key, options.headers if options else None)
            params = _build_params(model, context, options)

            if options and options.on_payload:
                next_params = await options.on_payload(params, model)
                if next_params is not None:
                    params = next_params

            openai_stream = await client.responses.create(
                params,
                signal=options.signal if options else None,
            )
            stream.push(AssistantMessageEventStart(partial=output))

            await process_responses_stream(
                openai_stream, output, stream, model,
                OpenAIResponsesStreamOptions(
                    service_tier=options.service_tier if options else None,
                    apply_service_tier_pricing=_apply_service_tier_pricing,
                ),
            )

            if options and options.signal and options.signal.is_set():
                raise RuntimeError("Request was aborted")

            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError("An unknown error occurred")

            stream.push(AssistantMessageEventDone(reason=output.stop_reason, message=output))
            stream.end()

        except Exception as error:
            output.stop_reason = (
                "aborted"
                if (options and options.signal and options.signal.is_set())
                else "error"
            )
            output.error_message = error.message if isinstance(error, Exception) else json.dumps(error)  # type: ignore[union-attr]
            stream.push(AssistantMessageEventError(reason=output.stop_reason, error=output))
            stream.end()

    import asyncio
    asyncio.create_task(_run())
    return stream


def stream_simple_openai_responses(
    model: Any,
    context: Any,
    options: Any | None = None,
) -> AssistantMessageEventStream:
    """Simplified OpenAI Responses streaming.

    Upstream: openai-responses.ts → streamSimpleOpenAIResponses()
    """
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)
    reasoning_effort = (
        options.reasoning if (options and supports_xhigh(model)) else (clamp_reasoning(options.reasoning) if options else None)
    )

    return stream_openai_responses(model, context, OpenAIResponsesOptions(
        api_key=base.api_key,
        max_tokens=base.max_tokens,
        temperature=base.temperature,
        signal=base.signal,
        headers=base.headers,
        on_payload=base.on_payload,
        metadata=base.metadata,
        reasoning_effort=reasoning_effort,
    ))
