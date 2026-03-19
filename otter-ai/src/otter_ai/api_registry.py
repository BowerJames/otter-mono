"""API provider registry for mapping API types to stream functions.

Providers register their stream/streamSimple implementations keyed by API
name (e.g. ``"anthropic-messages"``).  The :func:`stream` entry point looks
up the registered provider for a model's ``api`` field and delegates to it.

Upstream reference: ``packages/ai/src/api-registry.ts``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from otter_ai.utils.event_stream import AssistantMessageEventStream

if TYPE_CHECKING:
    from otter_ai.types import Context, Model, SimpleStreamOptions, StreamOptions


# ============================================================================
# Provider types
# ============================================================================


@dataclass
class ApiProvider:
    """A provider implementation for a specific API type.

    Attributes
    ----------
    api:
        The API identifier this provider handles (e.g. ``"anthropic-messages"``).
    stream:
        Full stream function accepting provider-specific options.
    stream_simple:
        Simplified stream function accepting :class:`SimpleStreamOptions`.
    """

    api: str
    stream: Any  # StreamFunction[Api, TOptions]
    stream_simple: Any  # StreamFunction[Api, SimpleStreamOptions]


@dataclass
class _ApiProviderInternal:
    """Internal wrapper that erases type parameters for storage."""

    api: str
    stream: _ApiStreamFunction
    stream_simple: _ApiStreamSimpleFunction


# Type aliases for the erased signatures stored in the registry.

type _ApiStreamFunction = Any  # (Model, Context, StreamOptions?) -> AssistantMessageEventStream
type _ApiStreamSimpleFunction = (  # noqa: E501
    Any  # (Model, Context, SimpleStreamOptions?) -> AssistantMessageEventStream
)


@dataclass
class _RegisteredEntry:
    provider: _ApiProviderInternal
    source_id: str | None = None


# ============================================================================
# Registry
# ============================================================================

_registry: dict[str, _RegisteredEntry] = {}


def _wrap_stream(
    api: str,
    stream: Any,
) -> _ApiStreamFunction:
    """Wrap a typed stream function to validate API match at call time."""

    def wrapped(
        model: Model,  # type: ignore[valid-type]
        context: Context,  # type: ignore[valid-type]
        options: StreamOptions | None = None,  # type: ignore[valid-type]
    ) -> AssistantMessageEventStream:
        if model.api != api:
            msg = f"Mismatched api: {model.api} expected {api}"
            raise RuntimeError(msg)
        return stream(model, context, options)

    return wrapped


def _wrap_stream_simple(
    api: str,
    stream_simple: Any,
) -> _ApiStreamSimpleFunction:
    """Wrap a typed stream-simple function to validate API match at call time."""

    def wrapped(
        model: Model,  # type: ignore[valid-type]
        context: Context,  # type: ignore[valid-type]
        options: SimpleStreamOptions | None = None,  # type: ignore[valid-type]
    ) -> AssistantMessageEventStream:
        if model.api != api:
            msg = f"Mismatched api: {model.api} expected {api}"
            raise RuntimeError(msg)
        return stream_simple(model, context, options)

    return wrapped


# ============================================================================
# Public API
# ============================================================================


def register_api_provider(provider: ApiProvider, source_id: str | None = None) -> None:
    """Register an API provider for a given API type.

    If a provider is already registered for the same API, it is replaced.
    """
    _registry[provider.api] = _RegisteredEntry(
        provider=_ApiProviderInternal(
            api=provider.api,
            stream=_wrap_stream(provider.api, provider.stream),
            stream_simple=_wrap_stream_simple(provider.api, provider.stream_simple),
        ),
        source_id=source_id,
    )


def get_api_provider(api: str) -> _ApiProviderInternal | None:
    """Look up the registered provider for an API type.

    Returns ``None`` if no provider is registered for the given API.
    """
    entry = _registry.get(api)
    return entry.provider if entry is not None else None


def get_api_providers() -> list[_ApiProviderInternal]:
    """Return all registered providers."""
    return [entry.provider for entry in _registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """Remove all providers that were registered with the given *source_id*."""
    keys_to_delete = [key for key, entry in _registry.items() if entry.source_id == source_id]
    for key in keys_to_delete:
        del _registry[key]


def clear_api_providers() -> None:
    """Remove all registered providers."""
    _registry.clear()
