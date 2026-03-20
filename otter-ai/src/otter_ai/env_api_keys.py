"""Environment variable-based API key resolution for 20+ providers.

Returns API keys from well-known environment variables.  Providers that
require OAuth tokens are excluded (handled by the OAuth module).

Upstream reference: ``packages/ai/src/env-api-keys.ts``
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa: E402

# ============================================================================
# Vertex ADC (Application Default Credentials) check
# ============================================================================

_cached_vertex_adc_exists: bool | None = None


def _has_vertex_adc_credentials() -> bool:
    """Check whether Vertex AI Application Default Credentials are present.

    Looks for ``GOOGLE_APPLICATION_CREDENTIALS`` first (explicit path), then
    falls back to ``~/.config/gcloud/application_default_credentials.json``.

    The result is cached after the first call.
    """
    global _cached_vertex_adc_exists  # noqa: PLW0603

    if _cached_vertex_adc_exists is not None:
        return _cached_vertex_adc_exists

    # Check GOOGLE_APPLICATION_CREDENTIALS env var first (standard way)
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_path:
        _cached_vertex_adc_exists = Path(gac_path).exists()
    else:
        # Fall back to default ADC path
        _cached_vertex_adc_exists = (
            Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        ).exists()

    return _cached_vertex_adc_exists


# ============================================================================
# Public API
# ============================================================================

# Mapping from provider name to the environment variable that holds its key.
_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
    "zai": "ZAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "huggingface": "HF_TOKEN",
    "opencode": "OPENCODE_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
    "kimi-coding": "KIMI_API_KEY",
}


def get_env_api_key(provider: str) -> str | None:
    """Get API key for *provider* from known environment variables.

    Returns ``None`` when no key is configured.  Providers that require
    OAuth tokens are not handled here — see the ``oauth`` module instead.

    Parameters
    ----------
    provider:
        Provider identifier (e.g. ``"openai"``, ``"anthropic"``).

    Returns
    -------
    The API key string, or ``None`` if not configured.
    """
    # GitHub Copilot: check multiple token env vars
    if provider == "github-copilot":
        return (
            os.environ.get("COPILOT_GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
        )

    # Anthropic: OAuth token takes precedence over API key
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_OAUTH_TOKEN") or os.environ.get(
            "ANTHROPIC_API_KEY",
        )

    # Google Vertex: explicit API key or Application Default Credentials
    if provider == "google-vertex":
        if os.environ.get("GOOGLE_CLOUD_API_KEY"):
            return os.environ["GOOGLE_CLOUD_API_KEY"]

        has_credentials = _has_vertex_adc_credentials()
        has_project = bool(
            os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT"),
        )
        has_location = bool(os.environ.get("GOOGLE_CLOUD_LOCATION"))

        if has_credentials and has_project and has_location:
            return "<authenticated>"
        return None

    # Amazon Bedrock: multiple credential sources
    if provider == "amazon-bedrock":
        if (
            os.environ.get("AWS_PROFILE")
            or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
        ):
            return "<authenticated>"
        return None

    env_var = _ENV_MAP.get(provider)
    return os.environ.get(env_var) if env_var else None
