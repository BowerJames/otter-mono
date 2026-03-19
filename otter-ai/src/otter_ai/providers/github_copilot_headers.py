"""Dynamic header utilities for GitHub Copilot API requests.

Copilot expects specific headers indicating whether the request is
user-initiated or agent-initiated, and whether vision input is included.

Upstream reference: ``packages/ai/src/providers/github-copilot-headers.ts``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otter_ai.types import Message


def infer_copilot_initiator(messages: list[Message]) -> str:
    """Determine whether the request is user-initiated or agent-initiated.

    Returns ``"user"`` if the last message is from the user, ``"agent"`` otherwise.
    """
    last = messages[-1] if messages else None
    if last is None or last.role == "user":
        return "user"
    return "agent"


def has_copilot_vision_input(messages: list[Message]) -> bool:
    """Check if any message contains image content.

    Used to determine whether the ``Copilot-Vision-Request`` header is needed.
    """
    for msg in messages:
        if msg.role == "user" and not isinstance(msg.content, str):
            if any(block.type == "image" for block in msg.content):
                return True
        if msg.role == "toolResult":
            if any(block.type == "image" for block in msg.content):
                return True
    return False


def build_copilot_dynamic_headers(
    messages: list[Message],
    has_images: bool,
) -> dict[str, str]:
    """Build dynamic headers required by the GitHub Copilot API.

    Parameters
    ----------
    messages:
        The conversation messages (used to infer initiator).
    has_images:
        Whether the request contains image content.
    """
    headers: dict[str, str] = {
        "X-Initiator": infer_copilot_initiator(messages),
        "Openai-Intent": "conversation-edits",
    }

    if has_images:
        headers["Copilot-Vision-Request"] = "true"

    return headers
