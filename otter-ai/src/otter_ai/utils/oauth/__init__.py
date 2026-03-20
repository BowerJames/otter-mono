"""OAuth credential management for AI providers.

Re-exports everything from :mod:`otter_ai.utils.oauth.index` and
:mod:`otter_ai.utils.oauth.types`.

Upstream reference: ``packages/ai/src/oauth.ts`` (barrel re-export)
"""

from otter_ai.utils.oauth.index import (  # noqa: F401
    get_oauth_api_key,
    get_oauth_provider,
    get_oauth_provider_info_list,
    get_oauth_providers,
    refresh_oauth_token,
    register_oauth_provider,
    reset_oauth_providers,
    unregister_oauth_provider,
)
from otter_ai.utils.oauth.types import (  # noqa: F401
    OAuthAuthInfo,
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthProviderId,
    OAuthProviderInfo,
    OAuthProviderInterface,
)
