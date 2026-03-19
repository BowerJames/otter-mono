# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false

"""Azure OpenAI Responses API provider.

Upstream: packages/ai/src/providers/azure-openai-responses.ts (~259 lines)
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
from .openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import build_base_options, clamp_reasoning

_AZURE_TOOL_CALL_PROVIDERS = frozenset([
    "openai", "openai-codex", "opencode", "azure-openai-responses",
])

_DEFAULT_AZURE_API_VERSION = "v1"


# ============================================================================
# Options
# ============================================================================


class AzureOpenAIResponsesOptions:
    """Options for Azure OpenAI Responses API streaming."""

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
        session_id: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        azure_api_version: str | None = None,
        azure_resource_name: str | None = None,
        azure_base_url: str | None = None,
        azure_deployment_name: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.signal = signal
        self.headers = headers
        self.on_payload = on_payload
        self.metadata = metadata
        self.session_id = session_id
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.azure_api_version = azure_api_version
        self.azure_resource_name = azure_resource_name
        self.azure_base_url = azure_base_url
        self.azure_deployment_name = azure_deployment_name


# ============================================================================
# Azure config resolution
# ============================================================================


def _parse_deployment_name_map(value: str | None) -> dict[str, str]:
    """Parse deployment name map from env var (model_id=deployment,...)."""
    result: dict[str, str] = {}
    if not value:
        return result
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        parts = trimmed.split("=", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            result[parts[0].strip()] = parts[1].strip()
    return result


def _resolve_deployment_name(
    model: Any,
    options: AzureOpenAIResponsesOptions | None = None,
) -> str:
    if options and options.azure_deployment_name:
        return options.azure_deployment_name
    env_map = _parse_deployment_name_map(
        os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_MAP")
    )
    return env_map.get(model.id, model.id) or model.id


def _normalize_azure_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _build_default_base_url(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/openai/v1"


def _resolve_azure_config(
    model: Any,
    options: AzureOpenAIResponsesOptions | None = None,
) -> tuple[str, str]:
    """Returns (base_url, api_version)."""
    api_version = (
        (options.azure_api_version if options else None)
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or _DEFAULT_AZURE_API_VERSION
    )

    base_url = (
        ((options.azure_base_url or "").strip() if options else "")
        or os.environ.get("AZURE_OPENAI_BASE_URL", "").strip()
        or None
    )
    resource_name = (
        (options.azure_resource_name if options else None)
        or os.environ.get("AZURE_OPENAI_RESOURCE_NAME")
    )

    resolved = base_url
    if not resolved and resource_name:
        resolved = _build_default_base_url(resource_name)
    if not resolved:
        resolved = getattr(model, "base_url", None)
    if not resolved:
        raise RuntimeError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or "
            "AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl."
        )

    return _normalize_azure_base_url(resolved), api_version


# ============================================================================
# Client creation
# ============================================================================


def _create_client(
    model: Any,
    api_key: str | None,
    options: AzureOpenAIResponsesOptions | None = None,
) -> openai.AzureOpenAI:
    if not api_key:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable or pass it as an argument."
        )

    headers: dict[str, str] = {}
    if hasattr(model, "headers") and model.headers:
        headers.update(model.headers)
    if options and options.headers:
        headers.update(options.headers)

    base_url, api_version = _resolve_azure_config(model, options)

    return openai.AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        default_headers=headers,
        base_url=base_url,
    )


# ============================================================================
# Build params
# ============================================================================


def _build_params(
    model: Any,
    context: Any,
    options: AzureOpenAIResponsesOptions | None = None,
    deployment_name: str | None = None,
) -> dict[str, Any]:
    messages = convert_responses_messages(model, context, _AZURE_TOOL_CALL_PROVIDERS)

    params: dict[str, Any] = {
        "model": deployment_name or model.id,
        "input": messages,
        "stream": True,
    }

    if options and options.session_id:
        params["prompt_cache_key"] = options.session_id

    if options and options.max_tokens:
        params["max_output_tokens"] = options.max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

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
# Streaming
# ============================================================================


def stream_azure_openai_responses(
    model: Any,
    context: Any,
    options: AzureOpenAIResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream completions from the Azure OpenAI Responses API.

    Upstream: azure-openai-responses.ts → streamAzureOpenAIResponses()
    """
    stream = AssistantMessageEventStream()

    async def _run() -> None:
        deployment_name = _resolve_deployment_name(model, options)

        output = AssistantMessage(
            role="assistant",
            content=[],
            api="azure-openai-responses",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = (options.api_key if options else None) or get_env_api_key(model.provider) or ""
            client = _create_client(model, api_key, options)
            params = _build_params(model, context, options, deployment_name)

            if options and options.on_payload:
                next_params = await options.on_payload(params, model)
                if next_params is not None:
                    params = next_params

            openai_stream = await client.responses.create(
                params,
                signal=options.signal if options else None,
            )
            stream.push(AssistantMessageEventStart(partial=output))

            await process_responses_stream(openai_stream, output, stream, model)

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


def stream_simple_azure_openai_responses(
    model: Any,
    context: Any,
    options: Any | None = None,
) -> AssistantMessageEventStream:
    """Simplified Azure OpenAI Responses streaming.

    Upstream: azure-openai-responses.ts → streamSimpleAzureOpenAIResponses()
    """
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)
    reasoning_effort = (
        options.reasoning if (options and supports_xhigh(model)) else (clamp_reasoning(options.reasoning) if options else None)
    )

    return stream_azure_openai_responses(model, context, AzureOpenAIResponsesOptions(
        api_key=base.api_key,
        max_tokens=base.max_tokens,
        temperature=base.temperature,
        signal=base.signal,
        headers=base.headers,
        on_payload=base.on_payload,
        metadata=base.metadata,
        reasoning_effort=reasoning_effort,
    ))
