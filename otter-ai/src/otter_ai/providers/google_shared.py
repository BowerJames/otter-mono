# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportRedeclaration=false, reportCallIssue=false

"""Shared utilities for Google Generative AI and Google Cloud Code Assist providers.

Upstream: packages/ai/src/providers/google-shared.ts (~326 lines)
"""

from __future__ import annotations

import re
from typing import Any, Literal

from otter_ai.types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    Tool,
)
from otter_ai.utils.sanitize_unicode import sanitize_surrogates

from .transform_messages import transform_messages

GoogleApiType = Literal["google-generative-ai", "google-gemini-cli", "google-vertex"]

# Sentinel value that tells the Gemini API to skip thought signature validation.
SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


# ============================================================================
# Thinking helpers
# ============================================================================


def is_thinking_part(part: Any) -> bool:
    """Determine whether a streamed Gemini Part should be treated as 'thinking'.

    Protocol note (Gemini / Vertex AI thought signatures):
    - ``thought: true`` is the definitive marker for thinking content (thought summaries).
    - ``thoughtSignature`` is an encrypted representation of the model's internal thought
      process used to preserve reasoning context across multi-turn interactions.
    - ``thoughtSignature`` can appear on ANY part type (text, functionCall, etc.) — it does
      NOT indicate the part itself is thinking content.
    - For non-functionCall responses, the signature appears on the last part for context replay.
    - When persisting/replaying model outputs, signature-bearing parts must be preserved as-is.
    """
    return getattr(part, "thought", False) is True


def retain_thought_signature(
    existing: str | None,
    incoming: str | None,
) -> str | None:
    """Retain thought signatures during streaming.

    Some backends only send ``thoughtSignature`` on the first delta; later deltas may
    omit it.  This helper preserves the last non-empty signature for the current block.
    """
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


def _is_valid_thought_signature(signature: str | None) -> bool:
    """Check if a thought signature is valid base64."""
    if not signature or len(signature) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE_PATTERN.match(signature))


def _resolve_thought_signature(
    is_same_provider_and_model: bool,
    signature: str | None,
) -> str | None:
    """Only keep signatures from the same provider/model and with valid base64."""
    if is_same_provider_and_model and _is_valid_thought_signature(signature):
        return signature
    return None


# ============================================================================
# Tool call ID helpers
# ============================================================================


def requires_tool_call_id(model_id: str) -> bool:
    """Models via Google APIs that require explicit tool call IDs in function calls."""
    return model_id.startswith("claude-") or model_id.startswith("gpt-oss-")


def _get_gemini_major_version(model_id: str) -> int | None:
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    if not match:
        return None
    return int(match.group(1))


def _supports_multimodal_function_response(model_id: str) -> bool:
    version = _get_gemini_major_version(model_id)
    if version is not None:
        return version >= 3
    return True


# ============================================================================
# Message conversion
# ============================================================================


def convert_messages(
    model: Model,
    context: Context,
) -> list[dict[str, Any]]:
    """Convert internal messages to Gemini Content[] format.

    Upstream: google-shared.ts → convertMessages()
    """
    contents: list[dict[str, Any]] = []

    def normalize_tool_call_id(tool_call_id: str, _model: Model, _msg: AssistantMessage) -> str:
        if not requires_tool_call_id(model.id):
            return tool_call_id
        return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]

    transformed_messages = transform_messages(context.messages, model, normalize_tool_call_id)

    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": sanitize_surrogates(msg.content)}],
                    }
                )
            else:
                parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        parts.append({"text": sanitize_surrogates(item.text)})
                    else:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": item.mime_type,
                                    "data": item.data,
                                },
                            }
                        )
                # Filter images if model doesn't support them
                if "image" not in (model.input or []):
                    parts = [p for p in parts if "text" in p]
                if not parts:
                    continue
                contents.append({"role": "user", "parts": parts})

        elif msg.role == "assistant":
            parts: list[dict[str, Any]] = []
            is_same_provider_and_model = (
                getattr(msg, "provider", None) == model.provider
                and getattr(msg, "model", None) == model.id
            )

            for block in msg.content:
                if block.type == "text":
                    if not block.text or not block.text.strip():
                        continue
                    thought_sig = _resolve_thought_signature(
                        is_same_provider_and_model,
                        getattr(block, "text_signature", None),
                    )
                    part: dict[str, Any] = {"text": sanitize_surrogates(block.text)}
                    if thought_sig:
                        part["thoughtSignature"] = thought_sig
                    parts.append(part)

                elif block.type == "thinking":
                    if not block.thinking or not block.thinking.strip():
                        continue
                    if is_same_provider_and_model:
                        thought_sig = _resolve_thought_signature(
                            is_same_provider_and_model,
                            getattr(block, "thinking_signature", None),
                        )
                        part: dict[str, Any] = {
                            "thought": True,
                            "text": sanitize_surrogates(block.thinking),
                        }
                        if thought_sig:
                            part["thoughtSignature"] = thought_sig
                        parts.append(part)
                    else:
                        parts.append({"text": sanitize_surrogates(block.thinking)})

                elif block.type == "toolCall":
                    thought_sig = _resolve_thought_signature(
                        is_same_provider_and_model,
                        getattr(block, "thought_signature", None),
                    )
                    is_gemini3 = "gemini-3" in model.id.lower()
                    effective_sig = thought_sig or (SKIP_THOUGHT_SIGNATURE if is_gemini3 else None)
                    fc: dict[str, Any] = {
                        "name": block.name,
                        "args": block.arguments or {},
                    }
                    if requires_tool_call_id(model.id):
                        fc["id"] = block.id
                    part: dict[str, Any] = {"functionCall": fc}
                    if effective_sig:
                        part["thoughtSignature"] = effective_sig
                    parts.append(part)

            if not parts:
                continue
            contents.append({"role": "model", "parts": parts})

        elif msg.role == "toolResult":
            text_content = [c for c in msg.content if c.type == "text"]
            text_result = "\n".join(c.text for c in text_content)
            image_content = (
                [c for c in msg.content if c.type == "image"]
                if "image" in (model.input or [])
                else []
            )

            has_text = len(text_result) > 0
            has_images = len(image_content) > 0

            model_supports_multimodal = _supports_multimodal_function_response(model.id)
            response_value = (
                sanitize_surrogates(text_result)
                if has_text
                else ("(see attached image)" if has_images else "")
            )

            image_parts = [
                {"inlineData": {"mimeType": img.mime_type, "data": img.data}}
                for img in image_content
            ]

            include_id = requires_tool_call_id(model.id)
            fr: dict[str, Any] = {
                "name": msg.tool_name,
                "response": (
                    {"error": response_value} if msg.is_error else {"output": response_value}
                ),
            }
            if has_images and model_supports_multimodal:
                fr["parts"] = image_parts
            if include_id:
                fr["id"] = msg.tool_call_id

            function_response_part: dict[str, Any] = {
                "functionResponse": fr,
            }

            # Merge consecutive function responses into single user turn
            last_content = contents[-1] if contents else None
            if (
                last_content
                and last_content.get("role") == "user"
                and any("functionResponse" in p for p in last_content.get("parts", []))
            ):
                last_content["parts"].append(function_response_part)
            else:
                contents.append(
                    {
                        "role": "user",
                        "parts": [function_response_part],
                    }
                )

            # For Gemini < 3, add images in separate user message
            if has_images and not model_supports_multimodal:
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": "Tool result image:"}, *image_parts],
                    }
                )

    return contents


# ============================================================================
# Tool conversion
# ============================================================================


def convert_tools(
    tools: list[Tool],
    use_parameters: bool = False,
) -> list[dict[str, Any]] | None:
    """Convert tools to Gemini function declarations format.

    By default uses ``parametersJsonSchema`` which supports full JSON Schema.
    Set ``use_parameters`` to True to use legacy ``parameters`` field (OpenAPI 3.03 Schema).
    """
    if not tools:
        return None
    return [
        {
            "functionDeclarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    **(
                        {"parameters": tool.parameters}
                        if use_parameters
                        else {"parametersJsonSchema": tool.parameters}
                    ),
                }
                for tool in tools
            ],
        },
    ]


# ============================================================================
# Stop reason mapping
# ============================================================================


def map_tool_choice(choice: str) -> str:
    """Map tool choice string to Gemini FunctionCallingConfigMode."""
    match choice:
        case "auto":
            return "AUTO"
        case "none":
            return "NONE"
        case "any":
            return "ANY"
        case _:
            return "AUTO"


def map_stop_reason(reason: Any) -> StopReason:
    """Map Gemini FinishReason to our StopReason.

    Accepts both the enum object and raw string values.
    """
    # Handle string form
    if isinstance(reason, str):
        return _map_stop_reason_string(reason)

    # Handle enum — extract the name
    name = getattr(reason, "name", None) or str(reason).upper()
    return _map_stop_reason_string(name)


def _map_stop_reason_string(reason: str) -> StopReason:
    """Map string finish reason to StopReason."""
    match reason:
        case "STOP":
            return "stop"
        case "MAX_TOKENS":
            return "length"
        case _:
            return "error"
