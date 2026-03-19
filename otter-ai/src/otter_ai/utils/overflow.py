"""Context overflow detection for LLM responses.

Detects when a model's response indicates the input exceeded the context
window, based on error message patterns from known providers.

Upstream reference: ``packages/ai/src/utils/overflow.ts``
"""

from __future__ import annotations

import re

from otter_ai.types import AssistantMessage

# Regex patterns to detect context overflow errors from different providers.
#
# Provider-specific patterns (with example error messages):
#
# - Anthropic: "prompt is too long: 213462 tokens > 200000 maximum"
# - OpenAI: "Your input exceeds the context window of this model"
# - Google: "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"
# - xAI: "This model's maximum prompt length is 131072 but the request contains 537812 tokens"
# - Groq: "Please reduce the length of the messages or completion"
# - OpenRouter: "maximum context length is X tokens... requested about Y tokens"
# - llama.cpp: "the request exceeds the available context size, try increasing it"
# - LM Studio: "tokens to keep from the initial prompt is greater than the context length"
# - GitHub Copilot: "prompt token count of X exceeds the limit of Y"
# - MiniMax: "invalid params, context window exceeds limit"
# - Kimi For Coding: "Your request exceeded model token limit: X (requested: Y)"
# - Cerebras: Returns "400/413 status code (no body)" - handled separately
# - Mistral: "Prompt contains X tokens ... too large for model with Y maximum context length"
# - z.ai: Does NOT error, accepts overflow silently - handled via usage check
# - Ollama: Silently truncates input - not detectable via error message

_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"input is too long for requested model", re.IGNORECASE),
    re.compile(r"exceeds the context window", re.IGNORECASE),
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),
    re.compile(r"reduce the length of the messages", re.IGNORECASE),
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),
    re.compile(r"exceeds the available context size", re.IGNORECASE),
    re.compile(r"greater than the context length", re.IGNORECASE),
    re.compile(r"context window exceeds limit", re.IGNORECASE),
    re.compile(r"exceeded model token limit", re.IGNORECASE),
    re.compile(r"too large for model with \d+ maximum context length", re.IGNORECASE),
    re.compile(r"model_context_window_exceeded", re.IGNORECASE),
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"token limit exceeded", re.IGNORECASE),
]

_CEREBRAS_PATTERN = re.compile(r"^4(00|13)\s*(status code)?\s*\(no body\)", re.IGNORECASE)


def is_context_overflow(message: AssistantMessage, context_window: int | None = None) -> bool:
    """Check if an assistant message represents a context overflow error.

    Handles two cases:

    1. **Error-based overflow**: Most providers return ``stopReason`` ``"error"``
       with a specific error message pattern.
    2. **Silent overflow**: Some providers accept overflow requests and return
       successfully.  For these, check if ``usage.input`` exceeds the context
       window (requires *context_window* parameter).

    Parameters
    ----------
    message:
        The assistant message to check.
    context_window:
        Optional context window size for detecting silent overflow (e.g. z.ai).

    Returns
    -------
    ``True`` if the message indicates a context overflow.
    """
    # Case 1: Check error message patterns
    if message.stop_reason == "error" and message.error_message:
        if any(p.search(message.error_message) for p in _OVERFLOW_PATTERNS):
            return True

        # Cerebras returns 400/413 with no body for context overflow
        if _CEREBRAS_PATTERN.match(message.error_message):
            return True

    # Case 2: Silent overflow (z.ai style)
    if context_window is not None and message.stop_reason == "stop":
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens > context_window:
            return True

    return False


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """Return the overflow detection patterns (for testing purposes)."""
    return list(_OVERFLOW_PATTERNS)
