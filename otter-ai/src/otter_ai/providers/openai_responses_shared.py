# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false

"""Shared utilities for OpenAI Responses-based providers.

Upstream: packages/ai/src/providers/openai-responses-shared.ts (~507 lines)
"""

from __future__ import annotations

import json
import re
from typing import Any

from otter_ai.models import calculate_cost
from otter_ai.types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
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
from .transform_messages import transform_messages


# =============================================================================
# Text signature encoding/decoding
# =============================================================================


def _encode_text_signature_v1(
    item_id: str,
    phase: str | None = None,
) -> str:
    """Encode a text signature v1 payload."""
    payload: dict[str, Any] = {"v": 1, "id": item_id}
    if phase:
        payload["phase"] = phase
    return json.dumps(payload)


def _parse_text_signature(
    signature: str | None,
) -> dict[str, Any] | None:
    """Parse a text signature, returning {id, phase?} or None."""
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
            if parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
                result: dict[str, Any] = {"id": parsed["id"]}
                phase = parsed.get("phase")
                if phase in ("commentary", "final_answer"):
                    result["phase"] = phase
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    return {"id": signature}


# =============================================================================
# Options
# =============================================================================


class OpenAIResponsesStreamOptions:
    """Options for OpenAI Responses streaming."""

    def __init__(
        self,
        *,
        service_tier: str | None = None,
        apply_service_tier_pricing: Any = None,
    ) -> None:
        self.service_tier = service_tier
        self.apply_service_tier_pricing = apply_service_tier_pricing


class ConvertResponsesMessagesOptions:
    """Options for message conversion."""

    def __init__(self, *, include_system_prompt: bool = True) -> None:
        self.include_system_prompt = include_system_prompt


class ConvertResponsesToolsOptions:
    """Options for tool conversion."""

    def __init__(self, *, strict: bool | None = None) -> None:
        self.strict = strict


# =============================================================================
# Message conversion
# =============================================================================


def convert_responses_messages(
    model: Model,
    context: Context,
    allowed_tool_call_providers: set[str] | frozenset[str],
    options: ConvertResponsesMessagesOptions | None = None,
) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI Responses API format.

    Upstream: openai-responses-shared.ts → convertResponsesMessages()
    """
    messages: list[dict[str, Any]] = []

    def _normalize_id_part(part: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", part)
        normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
        return normalized.rstrip("_")

    def _normalize_tool_call_id(tool_call_id: str) -> str:
        if model.provider not in allowed_tool_call_providers:
            return _normalize_id_part(tool_call_id)
        if "|" not in tool_call_id:
            return _normalize_id_part(tool_call_id)
        call_id, item_id_raw = tool_call_id.split("|", 1)
        normalized_call_id = _normalize_id_part(call_id)
        normalized_item_id = _normalize_id_part(item_id_raw)
        if not normalized_item_id.startswith("fc"):
            normalized_item_id = _normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    transformed_messages = transform_messages(context.messages, model, _normalize_tool_call_id)

    include_system = options.include_system_prompt if options else True
    if include_system and context.system_prompt:
        role = "developer" if model.reasoning else "system"
        messages.append({
            "role": role,
            "content": sanitize_surrogates(context.system_prompt),
        })

    msg_index = 0
    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}],
                })
            else:
                content: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({
                            "type": "input_text",
                            "text": sanitize_surrogates(item.text),
                        })
                    else:
                        content.append({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": f"data:{item.mime_type};base64,{item.data}",
                        })
                if "image" not in (model.input or []):
                    content = [c for c in content if c.get("type") != "input_image"]
                if not content:
                    continue
                messages.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            output: list[dict[str, Any]] = []
            is_different_model = (
                msg.model != model.id
                and msg.provider == model.provider
                and msg.api == model.api
            )

            for block in msg.content:
                if block.type == "thinking":
                    if block.thinking_signature:
                        reasoning_item = json.loads(block.thinking_signature)
                        output.append(reasoning_item)

                elif block.type == "text":
                    text_block = block
                    parsed_sig = _parse_text_signature(
                        getattr(text_block, "text_signature", None)
                    )
                    msg_id = parsed_sig.get("id") if parsed_sig else None
                    if not msg_id:
                        msg_id = f"msg_{msg_index}"
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"

                    output_msg: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": sanitize_surrogates(text_block.text),
                            "annotations": [],
                        }],
                        "status": "completed",
                        "id": msg_id,
                    }
                    if parsed_sig and "phase" in parsed_sig:
                        output_msg["phase"] = parsed_sig["phase"]
                    output.append(output_msg)

                elif block.type == "toolCall":
                    tool_call = block
                    parts = tool_call.id.split("|", 1)
                    call_id = parts[0]
                    item_id_raw = parts[1] if len(parts) > 1 else None
                    item_id: str | None = item_id_raw

                    if is_different_model and item_id and item_id.startswith("fc_"):
                        item_id = None

                    output.append({
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments or {}),
                    })

            if not output:
                continue
            messages.extend(output)

        elif msg.role == "toolResult":
            text_parts = [c for c in msg.content if c.type == "text"]
            text_result = "\n".join(c.text for c in text_parts)
            has_images = any(c.type == "image" for c in msg.content)
            has_text = len(text_result) > 0
            call_id = msg.tool_call_id.split("|", 1)[0]

            if has_images and "image" in (model.input or []):
                content_parts: list[dict[str, Any]] = []
                if has_text:
                    content_parts.append({
                        "type": "input_text",
                        "text": sanitize_surrogates(text_result),
                    })
                for block in msg.content:
                    if block.type == "image":
                        content_parts.append({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": f"data:{block.mime_type};base64,{block.data}",
                        })
                output_val = content_parts
            else:
                output_val = sanitize_surrogates(text_result if has_text else "(see attached image)")

            messages.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_val,
            })

        msg_index += 1

    return messages


# =============================================================================
# Tool conversion
# =============================================================================


def convert_responses_tools(
    tools: list[Tool],
    options: ConvertResponsesToolsOptions | None = None,
) -> list[dict[str, Any]]:
    """Convert tools to OpenAI Responses API format.

    Upstream: openai-responses-shared.ts → convertResponsesTools()
    """
    strict = options.strict if options else False
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": strict,
        }
        for tool in tools
    ]


# =============================================================================
# Stop reason mapping
# =============================================================================


def _map_stop_reason(status: str | None) -> StopReason:
    """Map OpenAI Responses API status to StopReason."""
    if not status:
        return "stop"
    match status:
        case "completed":
            return "stop"
        case "incomplete":
            return "length"
        case "failed" | "cancelled":
            return "error"
        case "in_progress" | "queued":
            return "stop"
        case _:
            raise RuntimeError(f"Unhandled OpenAI Responses stop reason: {status}")


# =============================================================================
# Stream processing
# =============================================================================


async def process_responses_stream(
    openai_stream: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    """Process an OpenAI Responses API stream into events.

    Upstream: openai-responses-shared.ts → processResponsesStream()
    """
    from otter_ai.types import (
        AssistantMessageEventTextDelta,
        AssistantMessageEventTextEnd,
        AssistantMessageEventTextStart,
        AssistantMessageEventThinkingDelta,
        AssistantMessageEventThinkingEnd,
        AssistantMessageEventThinkingStart,
        AssistantMessageEventToolcallDelta,
        AssistantMessageEventToolcallEnd,
        AssistantMessageEventToolcallStart,
    )

    current_item: Any = None
    current_block: Any = None  # ThinkingContent | TextContent | ToolCall
    blocks = output.content

    def block_index() -> int:
        return len(blocks) - 1

    async for event in openai_stream:
        event_type = event.type

        if event_type == "response.created":
            output.response_id = event.response.id

        elif event_type == "response.output_item.added":
            item = event.item
            if item.type == "reasoning":
                current_item = item
                current_block = ThinkingContent(type="thinking", thinking="")
                output.content.append(current_block)
                stream.push(AssistantMessageEventThinkingStart(
                    content_index=block_index(),
                    partial=output,
                ))
            elif item.type == "message":
                current_item = item
                current_block = TextContent(type="text", text="")
                output.content.append(current_block)
                stream.push(AssistantMessageEventTextStart(
                    content_index=block_index(),
                    partial=output,
                ))
            elif item.type == "function_call":
                current_item = item
                current_block = ToolCall(
                    type="toolCall",
                    id=f"{item.call_id}|{item.id}",
                    name=item.name,
                    arguments={},
                )
                current_block._partial_json = item.arguments or ""  # type: ignore[attr-defined]
                output.content.append(current_block)
                stream.push(AssistantMessageEventToolcallStart(
                    content_index=block_index(),
                    partial=output,
                ))

        elif event_type == "response.reasoning_summary_part.added":
            if current_item and current_item.type == "reasoning":
                if not hasattr(current_item, "summary") or current_item.summary is None:
                    current_item.summary = []
                current_item.summary.append(event.part)

        elif event_type == "response.reasoning_summary_text.delta":
            if (
                getattr(current_item, "type", None) == "reasoning"
                and current_block
                and current_block.type == "thinking"
            ):
                summary = getattr(current_item, "summary", None) or []
                if summary:
                    last_part = summary[-1]
                    current_block.thinking += event.delta
                    last_part.text += event.delta
                    stream.push(AssistantMessageEventThinkingDelta(
                        content_index=block_index(),
                        delta=event.delta,
                        partial=output,
                    ))

        elif event_type == "response.reasoning_summary_part.done":
            if (
                getattr(current_item, "type", None) == "reasoning"
                and current_block
                and current_block.type == "thinking"
            ):
                summary = getattr(current_item, "summary", None) or []
                if summary:
                    last_part = summary[-1]
                    current_block.thinking += "\n\n"
                    last_part.text += "\n\n"
                    stream.push(AssistantMessageEventThinkingDelta(
                        content_index=block_index(),
                        delta="\n\n",
                        partial=output,
                    ))

        elif event_type == "response.content_part.added":
            if getattr(current_item, "type", None) == "message":
                if not hasattr(current_item, "content") or current_item.content is None:
                    current_item.content = []
                if event.part.type in ("output_text", "refusal"):
                    current_item.content.append(event.part)

        elif event_type == "response.output_text.delta":
            if (
                getattr(current_item, "type", None) == "message"
                and current_block
                and current_block.type == "text"
            ):
                content = getattr(current_item, "content", None)
                if not content:
                    continue
                last_part = content[-1] if content else None
                if last_part and getattr(last_part, "type", None) == "output_text":
                    current_block.text += event.delta
                    last_part.text += event.delta
                    stream.push(AssistantMessageEventTextDelta(
                        content_index=block_index(),
                        delta=event.delta,
                        partial=output,
                    ))

        elif event_type == "response.refusal.delta":
            if (
                getattr(current_item, "type", None) == "message"
                and current_block
                and current_block.type == "text"
            ):
                content = getattr(current_item, "content", None)
                if not content:
                    continue
                last_part = content[-1] if content else None
                if last_part and getattr(last_part, "type", None) == "refusal":
                    current_block.text += event.delta
                    last_part.refusal += event.delta
                    stream.push(AssistantMessageEventTextDelta(
                        content_index=block_index(),
                        delta=event.delta,
                        partial=output,
                    ))

        elif event_type == "response.function_call_arguments.delta":
            if (
                getattr(current_item, "type", None) == "function_call"
                and current_block
                and current_block.type == "toolCall"
            ):
                partial = getattr(current_block, "_partial_json", "")
                partial += event.delta
                current_block._partial_json = partial  # type: ignore[attr-defined]
                current_block.arguments = parse_streaming_json(partial)
                stream.push(AssistantMessageEventToolcallDelta(
                    content_index=block_index(),
                    delta=event.delta,
                    partial=output,
                ))

        elif event_type == "response.function_call_arguments.done":
            if (
                getattr(current_item, "type", None) == "function_call"
                and current_block
                and current_block.type == "toolCall"
            ):
                current_block._partial_json = event.arguments  # type: ignore[attr-defined]
                current_block.arguments = parse_streaming_json(event.arguments)

        elif event_type == "response.output_item.done":
            item = event.item

            if item.type == "reasoning" and current_block and current_block.type == "thinking":
                summary = getattr(item, "summary", None)
                if summary:
                    current_block.thinking = "\n\n".join(s.text for s in summary)
                else:
                    current_block.thinking = ""
                current_block.thinking_signature = json.dumps(item)
                stream.push(AssistantMessageEventThinkingEnd(
                    content_index=block_index(),
                    content=current_block.thinking,
                    partial=output,
                ))
                current_block = None

            elif item.type == "message" and current_block and current_block.type == "text":
                texts = []
                for c in (item.content or []):
                    if getattr(c, "type", None) == "output_text":
                        texts.append(c.text)
                    elif getattr(c, "type", None) == "refusal":
                        texts.append(c.refusal)
                current_block.text = "".join(texts)
                phase = getattr(item, "phase", None)
                current_block.text_signature = _encode_text_signature_v1(
                    item.id, phase if phase else None,
                )
                stream.push(AssistantMessageEventTextEnd(
                    content_index=block_index(),
                    content=current_block.text,
                    partial=output,
                ))
                current_block = None

            elif item.type == "function_call":
                partial = getattr(current_block, "_partial_json", "") if current_block else ""
                args = parse_streaming_json(partial) if partial else parse_streaming_json(item.arguments or "{}")
                tool_call = ToolCall(
                    type="toolCall",
                    id=f"{item.call_id}|{item.id}",
                    name=item.name,
                    arguments=args,
                )
                current_block = None
                stream.push(AssistantMessageEventToolcallEnd(
                    content_index=block_index(),
                    tool_call=tool_call,
                    partial=output,
                ))

        elif event_type == "response.completed":
            response = event.response
            if getattr(response, "id", None):
                output.response_id = response.id
            if getattr(response, "usage", None):
                usage = response.usage
                cached_tokens = getattr(
                    getattr(usage, "input_tokens_details", None), "cached_tokens", 0,
                ) or 0
                output.usage = Usage(
                    input=(getattr(usage, "input_tokens", 0) or 0) - cached_tokens,
                    output=getattr(usage, "output_tokens", 0) or 0,
                    cache_read=cached_tokens,
                    cache_write=0,
                )
                output.usage.total_tokens = getattr(usage, "total_tokens", 0) or 0
            calculate_cost(model, output.usage)

            if options and options.apply_service_tier_pricing:
                service_tier = getattr(response, "service_tier", None) or options.service_tier
                options.apply_service_tier_pricing(output.usage, service_tier)

            output.stop_reason = _map_stop_reason(getattr(response, "status", None))
            if any(b.type == "toolCall" for b in output.content) and output.stop_reason == "stop":
                output.stop_reason = "toolUse"

        elif event_type == "error":
            raise RuntimeError(f"Error Code {getattr(event, 'code', 'unknown')}: {getattr(event, 'message', 'no message')}")

        elif event_type == "response.failed":
            error = getattr(getattr(event, "response", None), "error", None)
            details = getattr(getattr(event, "response", None), "incomplete_details", None)
            msg = (
                f"{getattr(error, 'code', 'unknown')}: {getattr(error, 'message', 'no message')}"
                if error
                else (
                    f"incomplete: {details.reason}"
                    if details and hasattr(details, "reason")
                    else "Unknown error (no error details in response)"
                )
            )
            raise RuntimeError(msg)
