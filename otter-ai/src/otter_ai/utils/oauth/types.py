"""OAuth type definitions for AI provider authentication.

Upstream reference: ``packages/ai/src/utils/oauth/types.ts``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from otter_ai.types import Model


# ============================================================================
# Core types
# ============================================================================


@dataclass
class OAuthCredentials:
    """Stored OAuth credentials for a provider.

    Attributes
    ----------
    refresh:
        Refresh token.
    access:
        Access token.
    expires:
        Expiration timestamp in milliseconds (Unix epoch).
    """

    refresh: str
    access: str
    expires: int
    extra: dict[str, Any] = field(default_factory=dict[str, Any])


# Type alias for provider identifier strings.
type OAuthProviderId = str

# Deprecated: use OAuthProviderId instead.
type OAuthProvider = OAuthProviderId


@dataclass
class OAuthPrompt:
    """A prompt shown to the user during login.

    Attributes
    ----------
    message:
        The prompt message.
    placeholder:
        Optional placeholder text for the input.
    allow_empty:
        Whether an empty response is acceptable.
    """

    message: str
    placeholder: str | None = None
    allow_empty: bool = False


@dataclass
class OAuthAuthInfo:
    """Information about an auth URL to present to the user.

    Attributes
    ----------
    url:
        The URL the user should open.
    instructions:
        Optional additional instructions.
    """

    url: str
    instructions: str | None = None


# ============================================================================
# Login callbacks protocol
# ============================================================================


class OAuthLoginCallbacks(Protocol):
    """Callbacks provided by the caller during an OAuth login flow.

    Upstream reference: ``OAuthLoginCallbacks`` interface in
    ``packages/ai/src/utils/oauth/types.ts``
    """

    def on_auth(self, info: OAuthAuthInfo) -> None: ...

    def on_prompt(self, prompt: OAuthPrompt) -> Awaitable[str]: ...

    def on_progress(self, message: str) -> None: ...

    def on_manual_code_input(self) -> Awaitable[str]: ...

    @property
    def signal(self) -> Any | None: ...


# ============================================================================
# Provider interface
# ============================================================================


@runtime_checkable
class OAuthProviderInterface(Protocol):
    """Interface for OAuth-based AI providers.

    Upstream reference: ``OAuthProviderInterface`` interface in
    ``packages/ai/src/utils/oauth/types.ts``
    """

    id: str
    name: str

    def login(self, callbacks: OAuthLoginCallbacks) -> Awaitable[OAuthCredentials]: ...

    def refresh_token(self, credentials: OAuthCredentials) -> Awaitable[OAuthCredentials]: ...

    def get_api_key(self, credentials: OAuthCredentials) -> str: ...

    def modify_models(
        self,
        models: list[Model],
        credentials: OAuthCredentials,
    ) -> list[Model]: ...


# ============================================================================
# Deprecated types
# ============================================================================


@dataclass
class OAuthProviderInfo:
    """Deprecated: use :class:`OAuthProviderInterface` instead.

    Upstream reference: ``OAuthProviderInfo`` interface (deprecated) in
    ``packages/ai/src/utils/oauth/types.ts``
    """

    id: str
    name: str
    available: bool = True
