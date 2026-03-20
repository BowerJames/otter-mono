"""OAuth credential management for AI providers.

This module handles login, token refresh, and credential storage
for OAuth-based providers:

- Anthropic (Claude Pro/Max)
- GitHub Copilot
- Google Cloud Code Assist (Gemini CLI)
- Antigravity (Gemini 3, Claude, GPT-OSS via Google Cloud)
- OpenAI Codex (ChatGPT OAuth)

Upstream reference: ``packages/ai/src/utils/oauth/index.ts``
"""

from __future__ import annotations

import time

from otter_ai.utils.oauth.types import (
    OAuthCredentials,
    OAuthProviderId,
    OAuthProviderInfo,
    OAuthProviderInterface,
)


# ============================================================================
# Provider Registry
# ============================================================================

# Provider implementations are registered here once they are implemented (#16).
# The registry starts empty; ``reset_oauth_providers()`` will populate it once
# the provider modules exist.
_oauth_provider_registry: dict[str, OAuthProviderInterface] = {}


def get_oauth_provider(id: OAuthProviderId) -> OAuthProviderInterface | None:
    """Get an OAuth provider by ID."""
    return _oauth_provider_registry.get(id)


def register_oauth_provider(provider: OAuthProviderInterface) -> None:
    """Register a custom OAuth provider."""
    _oauth_provider_registry[provider.id] = provider


def unregister_oauth_provider(id: str) -> None:
    """Unregister an OAuth provider.

    If the provider is built-in, restores the built-in implementation.
    Custom providers are removed completely.
    """
    # Once built-in providers are implemented (#16), check against them here.
    _oauth_provider_registry.pop(id, None)


def reset_oauth_providers() -> None:
    """Reset OAuth providers to built-ins.

    Populates the registry with the 5 built-in providers once they are
    implemented (#16).
    """
    _oauth_provider_registry.clear()
    # Built-in providers will be imported and registered here once #16 is done.
    # The upstream registers:
    #   anthropic, github-copilot, gemini-cli, antigravity, openai-codex


def get_oauth_providers() -> list[OAuthProviderInterface]:
    """Get all registered OAuth providers."""
    return list(_oauth_provider_registry.values())


def get_oauth_provider_info_list() -> list[OAuthProviderInfo]:
    """Get provider info list.

    .. deprecated::
        Use :func:`get_oauth_providers` which returns
        :class:`OAuthProviderInterface` instances.
    """
    return [
        OAuthProviderInfo(id=p.id, name=p.name, available=True)
        for p in get_oauth_providers()
    ]


# ============================================================================
# High-level API (uses provider registry)
# ============================================================================


async def refresh_oauth_token(
    provider_id: OAuthProviderId,
    credentials: OAuthCredentials,
) -> OAuthCredentials:
    """Refresh token for any OAuth provider.

    .. deprecated::
        Use ``get_oauth_provider(id).refresh_token()`` instead.
    """
    provider = get_oauth_provider(provider_id)
    if provider is None:
        msg = f"Unknown OAuth provider: {provider_id}"
        raise RuntimeError(msg)
    return await provider.refresh_token(credentials)


async def get_oauth_api_key(
    provider_id: OAuthProviderId,
    credentials: dict[str, OAuthCredentials],
) -> tuple[OAuthCredentials, str] | None:
    """Get API key for a provider from OAuth credentials.

    Automatically refreshes expired tokens.

    Parameters
    ----------
    provider_id:
        The OAuth provider identifier.
    credentials:
        Mapping of provider IDs to their stored credentials.

    Returns
    -------
    A ``(new_credentials, api_key)`` tuple, or ``None`` if no credentials
    exist for the provider.

    Raises
    ------
    RuntimeError
        If the provider is unknown or token refresh fails.
    """
    provider = get_oauth_provider(provider_id)
    if provider is None:
        msg = f"Unknown OAuth provider: {provider_id}"
        raise RuntimeError(msg)

    creds = credentials.get(provider_id)
    if creds is None:
        return None

    # Refresh if expired.
    if time.time() * 1000 >= creds.expires:
        try:
            creds = await provider.refresh_token(creds)
        except Exception:
            msg = f"Failed to refresh OAuth token for {provider_id}"
            raise RuntimeError(msg) from None

    api_key = provider.get_api_key(creds)
    return creds, api_key
