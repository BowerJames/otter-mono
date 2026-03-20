# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false

"""Google Vertex AI provider.

Upstream: packages/ai/src/providers/google-vertex.ts (~523 lines)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from google import genai

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
)
from otter_ai.utils.event_stream import AssistantMessageEventStream
from otter_ai.utils.sanitize_unicode import sanitize_surrogates

from .google_shared import (
    convert_messages,
    convert_tools,
    is_thinking_part,
    map_stop_reason,
    map_tool_choice,
    retain_thought_signature,
)
from .simple_options import build_base_options, clamp_reasoning

GoogleThinkingLevel = Literal[
    "THINKING_LEVEL_UNSPECIFIED",
    "MINIMAL",
    "LOW",
    "MEDIUM",
    "HIGH",
]

_API_VERSION = "v1"
_GEMINI_3_PRO_RE = re.compile(r"gemini-3(?:\.\d+)?-pro")
_GEMINI_3_FLASH_RE = re.compile(r"gemini-3(?:\.\d+)?-flash")

_tool_call_counter = 0


# ============================================================================
# Options
# ============================================================================


@dataclass
class GoogleVertexOptions(StreamOptions):
    """Options for Google Vertex AI streaming.

    Upstream: ``interface GoogleVertexOptions extends StreamOptions``
    """

    tool_choice: str | None = None
    thinking: dict[str, Any] | None = None
    project: str | None = None
    location: str | None = None


# ============================================================================
# Helpers
# ============================================================================


def _is_gemini_3_pro_model(model: Model) -> bool:
    return bool(_GEMINI_3_PRO_RE.search(model.id.lower()))


def _is_gemini_3_flash_model(model: Model) -> bool:
    return bool(_GEMINI_3_FLASH_RE.search(model.id.lower()))


def _resolve_api_key(options: GoogleVertexOptions | None) -> str | None:
    api_key = ((options.api_key or "").strip() if options else "") or os.environ.get(
        "GOOGLE_CLOUD_API_KEY", ""
    ).strip()
    if not api_key or re.match(r"^<[^>]+>$", api_key):
        return None
    return api_key


def _resolve_project(options: GoogleVertexOptions | None) -> str:
    project = (
        (options.project if options else None)
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )
    if not project:
        raise RuntimeError(
            "Vertex AI requires a project ID. Set "
            "GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def _resolve_location(options: GoogleVertexOptions | None) -> str:
    location = (options.location if options else None) or os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not location:
        raise RuntimeError(
            "Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options."
        )
    return location


# ============================================================================
# Client creation
# ============================================================================


def _create_client_with_adc(
    model: Model,
    project: str,
    location: str,
    options_headers: dict[str, str] | None = None,
) -> genai.Client:
    http_options: dict[str, Any] = {}
    merged_headers: dict[str, str] = {}
    if model.headers:
        merged_headers.update(model.headers)
    if options_headers:
        merged_headers.update(options_headers)
    if merged_headers:
        http_options["headers"] = merged_headers

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        api_version=_API_VERSION,
        http_options=http_options if http_options else None,
    )


def _create_client_with_api_key(
    model: Model,
    api_key: str,
    options_headers: dict[str, str] | None = None,
) -> genai.Client:
    http_options: dict[str, Any] = {}
    merged_headers: dict[str, str] = {}
    if model.headers:
        merged_headers.update(model.headers)
    if options_headers:
        merged_headers.update(options_headers)
    if merged_headers:
        http_options["headers"] = merged_headers

    return genai.Client(
        vertexai=True,
        api_key=api_key,
        api_version=_API_VERSION,
        http_options=http_options if http_options else None,
    )


# ============================================================================
# Build params
# ============================================================================


def _build_params(
    model: Model,
    context: Context,
    options: GoogleVertexOptions | None = None,
) -> dict[str, Any]:
    contents = convert_messages(model, context)

    config: dict[str, Any] = {}

    if options and options.temperature is not None:
        config["temperature"] = options.temperature
    if options and options.max_tokens is not None:
        config["max_output_tokens"] = options.max_tokens

    if context.system_prompt:
        config["system_instruction"] = sanitize_surrogates(context.system_prompt)

    if context.tools:
        tools = convert_tools(context.tools)
        if tools:
            config["tools"] = tools
        if options and options.tool_choice:
            config["tool_config"] = {
                "function_calling_config": {
                    "mode": map_tool_choice(options.tool_choice),
                },
            }

    if options and options.thinking and options.thinking.get("enabled") and model.reasoning:
        thinking_config: dict[str, Any] = {"include_thoughts": True}
        if options.thinking.get("level") is not None:
            thinking_config["thinking_level"] = options.thinking["level"]
        elif options.thinking.get("budget_tokens") is not None:
            thinking_config["thinking_budget"] = options.thinking["budget_tokens"]
        config["thinking_config"] = thinking_config

    return {
        "model": model.id,
        "contents": contents,
        "config": config,
    }


def _get_gemini_3_thinking_level(effort: str, model: Model) -> GoogleThinkingLevel:
    if _is_gemini_3_pro_model(model):
        match effort:
            case "minimal" | "low":
                return "LOW"
            case "medium" | "high" | _:
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
        case _:
            return "HIGH"


def _get_google_budget(
    model: Model,
    effort: str,
    custom_budgets: ThinkingBudgets | None = None,
) -> int:
    if custom_budgets:
        budget_val = getattr(custom_budgets, effort, None)
        if budget_val is not None:
            return budget_val
    if "2.5-pro" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}
        return budgets.get(effort, -1)
    if "2.5-flash" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}
        return budgets.get(effort, -1)
    return -1


# ============================================================================
# Streaming
# ============================================================================


def stream_google_vertex(
    model: Model,
    context: Context,
    options: GoogleVertexOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream completions from Google Vertex AI.

    Upstream: google-vertex.ts → streamGoogleVertex()
    """
    global _tool_call_counter

    stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api="google-vertex",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = _resolve_api_key(options)
            if api_key:
                client = _create_client_with_api_key(
                    model,
                    api_key,
                    options.headers if options else None,
                )
            else:
                client = _create_client_with_adc(
                    model,
                    _resolve_project(options),
                    _resolve_location(options),
                    options.headers if options else None,
                )

            params = _build_params(model, context, options)

            if options and options.on_payload:
                next_params = await options.on_payload(params, model)
                if next_params is not None:
                    params = next_params

            google_stream = await client.aio.models.generate_content_stream(params)

            stream.push(AssistantMessageEventStart(partial=output))

            current_block: TextContent | ThinkingContent | None = None
            blocks = output.content

            def block_index() -> int:
                return len(blocks) - 1

            async for chunk in google_stream:
                if not output.response_id and hasattr(chunk, "response_id") and chunk.response_id:
                    output.response_id = chunk.response_id

                candidate = (chunk.candidates or [None])[0] if chunk.candidates else None
                if candidate and hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text is not None:
                            is_thinking = is_thinking_part(part)
                            if (
                                not current_block
                                or (is_thinking and current_block.type != "thinking")
                                or (not is_thinking and current_block.type != "text")
                            ):
                                if current_block:
                                    if current_block.type == "text":
                                        stream.push(
                                            AssistantMessageEventTextEnd(
                                                content_index=len(blocks) - 1,
                                                content=current_block.text,
                                                partial=output,
                                            )
                                        )
                                    else:
                                        stream.push(
                                            AssistantMessageEventThinkingEnd(
                                                content_index=block_index(),
                                                content=current_block.thinking,
                                                partial=output,
                                            )
                                        )
                                if is_thinking:
                                    current_block = ThinkingContent(
                                        type="thinking", thinking="", thinking_signature=None
                                    )
                                    output.content.append(current_block)
                                    stream.push(
                                        AssistantMessageEventThinkingStart(
                                            content_index=block_index(), partial=output
                                        )
                                    )
                                else:
                                    current_block = TextContent(type="text", text="")
                                    output.content.append(current_block)
                                    stream.push(
                                        AssistantMessageEventTextStart(
                                            content_index=block_index(), partial=output
                                        )
                                    )
                            if current_block.type == "thinking":
                                current_block.thinking += part.text
                                current_block.thinking_signature = retain_thought_signature(
                                    current_block.thinking_signature,
                                    getattr(part, "thought_signature", None),
                                )
                                stream.push(
                                    AssistantMessageEventThinkingDelta(
                                        content_index=block_index(), delta=part.text, partial=output
                                    )
                                )
                            else:
                                current_block.text += part.text
                                sig = retain_thought_signature(
                                    getattr(current_block, "text_signature", None),
                                    getattr(part, "thought_signature", None),
                                )
                                if sig:
                                    current_block.text_signature = sig  # type: ignore[attr-defined]
                                stream.push(
                                    AssistantMessageEventTextDelta(
                                        content_index=block_index(), delta=part.text, partial=output
                                    )
                                )

                        if hasattr(part, "function_call") and part.function_call:
                            if current_block:
                                if current_block.type == "text":
                                    stream.push(
                                        AssistantMessageEventTextEnd(
                                            content_index=block_index(),
                                            content=current_block.text,
                                            partial=output,
                                        )
                                    )
                                else:
                                    stream.push(
                                        AssistantMessageEventThinkingEnd(
                                            content_index=block_index(),
                                            content=current_block.thinking,
                                            partial=output,
                                        )
                                    )
                                current_block = None

                            fc = part.function_call
                            provided_id = getattr(fc, "id", None)
                            needs_new_id = not provided_id or any(
                                b.type == "toolCall" and b.id == provided_id for b in output.content
                            )
                            tool_call_id = (
                                f"{fc.name}_{int(time.time() * 1000)}"
                                f"_{_tool_call_counter:= _tool_call_counter + 1}"
                                if needs_new_id
                                else provided_id
                            )
                            tool_call = ToolCall(
                                type="toolCall",
                                id=tool_call_id,
                                name=fc.name or "",
                                arguments=(fc.args if isinstance(fc.args, dict) else {}) or {},
                            )
                            thought_sig = getattr(part, "thought_signature", None)
                            if thought_sig:
                                tool_call.thought_signature = thought_sig  # type: ignore[attr-defined]
                            output.content.append(tool_call)
                            stream.push(
                                AssistantMessageEventToolcallStart(
                                    content_index=block_index(), partial=output
                                )
                            )
                            stream.push(
                                AssistantMessageEventToolcallDelta(
                                    content_index=block_index(),
                                    delta=json.dumps(tool_call.arguments),
                                    partial=output,
                                )
                            )
                            stream.push(
                                AssistantMessageEventToolcallEnd(
                                    content_index=block_index(), tool_call=tool_call, partial=output
                                )
                            )

                if candidate and hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    output.stop_reason = map_stop_reason(candidate.finish_reason)
                    if any(b.type == "toolCall" for b in output.content):
                        output.stop_reason = "toolUse"

                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    um = chunk.usage_metadata
                    output.usage = Usage(
                        input=getattr(um, "prompt_token_count", 0) or 0,
                        output=(getattr(um, "candidates_token_count", 0) or 0)
                        + (getattr(um, "thoughts_token_count", 0) or 0),
                        cache_read=getattr(um, "cached_content_token_count", 0) or 0,
                        cache_write=0,
                    )
                    output.usage.total_tokens = getattr(um, "total_token_count", 0) or 0
                    calculate_cost(model, output.usage)

            if current_block:
                if current_block.type == "text":
                    stream.push(
                        AssistantMessageEventTextEnd(
                            content_index=block_index(), content=current_block.text, partial=output
                        )
                    )
                else:
                    stream.push(
                        AssistantMessageEventThinkingEnd(
                            content_index=block_index(),
                            content=current_block.thinking,
                            partial=output,
                        )
                    )

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


def stream_simple_google_vertex(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simplified Google Vertex AI streaming.

    Upstream: google-vertex.ts → streamSimpleGoogleVertex()
    """
    base = build_base_options(model, options, None)

    if not (options and options.reasoning):
        return stream_google_vertex(
            model,
            context,
            GoogleVertexOptions(
                api_key=base.api_key,
                max_tokens=base.max_tokens,
                temperature=base.temperature,
                signal=base.signal,
                headers=base.headers,
                on_payload=base.on_payload,
                metadata=base.metadata,
                thinking={"enabled": False},
            ),
        )

    effort = clamp_reasoning(options.reasoning) or "high"

    if _is_gemini_3_pro_model(model) or _is_gemini_3_flash_model(model):
        level = _get_gemini_3_thinking_level(effort, model)
        return stream_google_vertex(
            model,
            context,
            GoogleVertexOptions(
                api_key=base.api_key,
                max_tokens=base.max_tokens,
                temperature=base.temperature,
                signal=base.signal,
                headers=base.headers,
                on_payload=base.on_payload,
                metadata=base.metadata,
                thinking={"enabled": True, "level": level},
            ),
        )

    budget = _get_google_budget(model, effort, options.thinking_budgets)
    return stream_google_vertex(
        model,
        context,
        GoogleVertexOptions(
            api_key=base.api_key,
            max_tokens=base.max_tokens,
            temperature=base.temperature,
            signal=base.signal,
            headers=base.headers,
            on_payload=base.on_payload,
            metadata=base.metadata,
            thinking={"enabled": True, "budget_tokens": budget},
        ),
    )
