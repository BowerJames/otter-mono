"""Lazy-loading provider registration.

Defers ``importlib.import_module()`` for each provider until its stream
function is first called.  This prevents loading all provider SDKs
(``openai``, ``anthropic``, ``google-genai``, ``mistralai``) at import time.

The module calls :func:`register_builtin_api_providers` on import
(side-effect import pattern matching upstream).

Upstream reference: ``packages/ai/src/providers/register-builtins.ts``
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Callable
from typing import Any

from otter_ai.api_registry import ApiProvider, clear_api_providers, register_api_provider
from otter_ai.types import (
    AssistantMessage,
    AssistantMessageEventError,
    Usage,
    UsageCost,
)
from otter_ai.utils.event_stream import AssistantMessageEventStream

# ============================================================================
# Lazy-loading helpers
# ============================================================================


def _create_lazy_load_error_message(model: Any) -> AssistantMessage:
    """Build an error :class:`AssistantMessage` for a failed lazy load."""
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=UsageCost(),
        ),
        stop_reason="error",
        error_message="Provider module failed to load",
        timestamp=int(time.time() * 1000),
    )


async def _forward_stream(
    target: AssistantMessageEventStream,
    source: AssistantMessageEventStream,
) -> None:
    """Forward all events from *source* into *target*, then end *target*."""
    try:
        async for event in source:
            target.push(event)
    finally:
        target.end()


def _create_lazy_stream(
    load_module: Callable[[], Any],
    stream_attr: str,
) -> Callable[[Any, Any, Any], AssistantMessageEventStream]:
    """Return a sync stream function that lazily loads *load_module* on first call.

    Parameters
    ----------
    load_module:
        Zero-arg callable that returns a module (or awaitable).
    stream_attr:
        Attribute name on the loaded module that holds the real stream function.
    """

    def lazy_stream(
        model: Any,
        context: Any,
        options: Any = None,
    ) -> AssistantMessageEventStream:
        outer = AssistantMessageEventStream()

        async def _run() -> None:
            try:
                mod = load_module()
                if asyncio.iscoroutine(mod):
                    mod = await mod
                real_fn = getattr(mod, stream_attr)
                inner: AssistantMessageEventStream = real_fn(model, context, options)
                await _forward_stream(outer, inner)
            except Exception as exc:
                msg = _create_lazy_load_error_message(model)
                err_name = (
                    exc.__class__.__name__
                    if hasattr(exc, "__class__")
                    else str(exc)
                )
                msg.error_message = err_name
                if str(exc):
                    msg.error_message += f": {exc}"
                outer.push(
                    AssistantMessageEventError(
                        type="error",
                        reason="error",
                        error=msg,
                    ),
                )
                outer.end(msg)

        asyncio.get_event_loop().create_task(_run())
        return outer

    return lazy_stream


# ============================================================================
# Module-level lazy-load caches
#
# Each ``_load_*`` function returns a cached promise-like object so that
# concurrent calls share the same import.
# ============================================================================

_load_promises: dict[str, asyncio.Task[Any] | None] = {
    name: None for name in (
        "anthropic",
        "openai_completions",
        "openai_responses",
        "azure_openai_responses",
        "google",
        "google_vertex",
        "google_gemini_cli",
        "mistral",
        "openai_codex_responses",
        "amazon_bedrock",
    )
}

_bedrock_module_override: Any = None


def _load_provider_module(provider_name: str) -> Any:
    """Return the (possibly cached) provider module for *provider_name*.

    Uses ``asyncio.get_event_loop().create_task`` to ensure the import
    happens only once, with subsequent calls awaiting the same task.
    """
    task = _load_promises.get(provider_name)
    if task is not None and not task.done():
        return task

    async def _do_import() -> Any:
        module_path = f"otter_ai.providers.{provider_name}"
        return importlib.import_module(module_path)

    new_task = asyncio.get_event_loop().create_task(_do_import())
    _load_promises[provider_name] = new_task
    return new_task


# ============================================================================
# Lazy stream wrappers (public exports)
# ============================================================================

stream_anthropic = _create_lazy_stream(
    lambda: _load_provider_module("anthropic"), "stream_anthropic",
)
stream_simple_anthropic = _create_lazy_stream(
    lambda: _load_provider_module("anthropic"), "stream_simple_anthropic",
)

stream_openai_completions = _create_lazy_stream(
    lambda: _load_provider_module("openai_completions"), "stream_openai_completions",
)
stream_simple_openai_completions = _create_lazy_stream(
    lambda: _load_provider_module("openai_completions"), "stream_simple_openai_completions",
)

stream_openai_responses = _create_lazy_stream(
    lambda: _load_provider_module("openai_responses"), "stream_openai_responses",
)
stream_simple_openai_responses = _create_lazy_stream(
    lambda: _load_provider_module("openai_responses"), "stream_simple_openai_responses",
)

stream_azure_openai_responses = _create_lazy_stream(
    lambda: _load_provider_module("azure_openai_responses"), "stream_azure_openai_responses",
)
stream_simple_azure_openai_responses = _create_lazy_stream(
    lambda: _load_provider_module("azure_openai_responses"), "stream_simple_azure_openai_responses",
)

stream_google = _create_lazy_stream(
    lambda: _load_provider_module("google"), "stream_google",
)
stream_simple_google = _create_lazy_stream(
    lambda: _load_provider_module("google"), "stream_simple_google",
)

stream_google_vertex = _create_lazy_stream(
    lambda: _load_provider_module("google_vertex"), "stream_google_vertex",
)
stream_simple_google_vertex = _create_lazy_stream(
    lambda: _load_provider_module("google_vertex"), "stream_simple_google_vertex",
)

stream_google_gemini_cli = _create_lazy_stream(
    lambda: _load_provider_module("google_gemini_cli"), "stream_google_gemini_cli",
)
stream_simple_google_gemini_cli = _create_lazy_stream(
    lambda: _load_provider_module("google_gemini_cli"), "stream_simple_google_gemini_cli",
)

stream_mistral = _create_lazy_stream(
    lambda: _load_provider_module("mistral"), "stream_mistral",
)
stream_simple_mistral = _create_lazy_stream(
    lambda: _load_provider_module("mistral"), "stream_simple_mistral",
)

stream_openai_codex_responses = _create_lazy_stream(
    lambda: _load_provider_module("openai_codex_responses"), "stream_openai_codex_responses",
)
stream_simple_openai_codex_responses = _create_lazy_stream(
    lambda: _load_provider_module("openai_codex_responses"), "stream_simple_openai_codex_responses",
)

def _load_bedrock_module() -> Any:
    """Load the Bedrock provider module, checking for an override first.

    Matches upstream ``loadBedrockProviderModule()`` which checks
    ``bedrockProviderModuleOverride`` before falling through to dynamic
    import.
    """
    if _bedrock_module_override is not None:
        return _bedrock_module_override
    return _load_provider_module("amazon_bedrock")


stream_bedrock = _create_lazy_stream(
    _load_bedrock_module, "stream_bedrock",
)
stream_simple_bedrock = _create_lazy_stream(
    _load_bedrock_module, "stream_simple_bedrock",
)


# ============================================================================
# Bedrock override (for environments without boto3)
# ============================================================================


def set_bedrock_provider_module(module: Any) -> None:
    """Override the Bedrock provider with an external module.

    The module must expose ``stream_bedrock`` and ``stream_simple_bedrock``.
    Used in environments where the AWS SDK is unavailable (e.g., browser).
    """
    global _bedrock_module_override  # noqa: PLW0603
    _bedrock_module_override = module


# ============================================================================
# Registration
# ============================================================================


def register_builtin_api_providers() -> None:
    """Register all 10 built-in API providers with lazy-loading wrappers."""
    register_api_provider(
        ApiProvider(
            api="anthropic-messages",
            stream=stream_anthropic,
            stream_simple=stream_simple_anthropic,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="openai-completions",
            stream=stream_openai_completions,
            stream_simple=stream_simple_openai_completions,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="mistral-conversations",
            stream=stream_mistral,
            stream_simple=stream_simple_mistral,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="openai-responses",
            stream=stream_openai_responses,
            stream_simple=stream_simple_openai_responses,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="azure-openai-responses",
            stream=stream_azure_openai_responses,
            stream_simple=stream_simple_azure_openai_responses,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="openai-codex-responses",
            stream=stream_openai_codex_responses,
            stream_simple=stream_simple_openai_codex_responses,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="google-generative-ai",
            stream=stream_google,
            stream_simple=stream_simple_google,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="google-gemini-cli",
            stream=stream_google_gemini_cli,
            stream_simple=stream_simple_google_gemini_cli,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="google-vertex",
            stream=stream_google_vertex,
            stream_simple=stream_simple_google_vertex,
        ),
    )
    register_api_provider(
        ApiProvider(
            api="bedrock-converse-stream",
            stream=stream_bedrock,
            stream_simple=stream_simple_bedrock,
        ),
    )


def reset_api_providers() -> None:
    """Clear all providers and re-register built-ins."""
    clear_api_providers()
    register_builtin_api_providers()


# ============================================================================
# Side-effect import: register all providers when this module is imported.
# Upstream: ``import "./providers/register-builtins.js";`` in ``stream.ts``.
# ============================================================================

register_builtin_api_providers()
