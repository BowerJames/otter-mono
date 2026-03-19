"""Streaming API entry points.

Resolves the registered provider for a model's ``api`` field and delegates
to its ``stream`` or ``stream_simple`` function.

Upstream reference: ``packages/ai/src/stream.ts``
"""

from __future__ import annotations

# Side-effect import: registers all built-in providers on first import.
# Upstream: ``import "./providers/register-builtins.js";``
import otter_ai.providers.register_builtins  # noqa: F401
from otter_ai.api_registry import get_api_provider
from otter_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ProviderStreamOptions,
    SimpleStreamOptions,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream


def _resolve_api_provider(api: str):
    """Look up the registered provider for *api*, raising on miss."""
    provider = get_api_provider(api)
    if provider is None:
        msg = f"No API provider registered for api: {api}"
        raise RuntimeError(msg)
    return provider


def stream(
    model: Model,
    context: Context,
    options: ProviderStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an LLM response using the full provider-specific options.

    Parameters
    ----------
    model:
        The model to use (its ``api`` field determines the provider).
    context:
        The conversation context (system prompt, messages, tools).
    options:
        Provider-specific stream options.

    Returns
    -------
    An :class:`AssistantMessageEventStream` yielding assistant message events.
    """
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: ProviderStreamOptions | None = None,
) -> AssistantMessage:
    """Stream an LLM response and return the final :class:`AssistantMessage`.

    Convenience wrapper around :func:`stream` that awaits the final result.
    """
    s = stream(model, context, options)
    return await s.result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an LLM response using :class:`SimpleStreamOptions`.

    Unlike :func:`stream`, this accepts the simplified options interface
    (including ``reasoning`` level) and delegates to the provider's
    ``stream_simple`` implementation.

    Parameters
    ----------
    model:
        The model to use.
    context:
        The conversation context.
    options:
        Simple stream options (temperature, max_tokens, reasoning, etc.).

    Returns
    -------
    An :class:`AssistantMessageEventStream` yielding assistant message events.
    """
    provider = _resolve_api_provider(model.api)
    return provider.stream_simple(model, context, options)


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """Stream an LLM response using :class:`SimpleStreamOptions` and return the
    final :class:`AssistantMessage`.

    Convenience wrapper around :func:`stream_simple`.
    """
    s = stream_simple(model, context, options)
    return await s.result()
