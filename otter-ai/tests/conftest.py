"""Shared fixtures and configuration for otter-ai tests.

Mirrors upstream test infrastructure:
- ``packages/ai/test/oauth.ts`` → API key resolution from auth file
- ``packages/ai/test/bedrock-utils.ts`` → Bedrock credential checks
- ``packages/ai/test/azure-utils.ts`` → Azure credential checks
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

# ============================================================================
# Auth file helpers (mirrors packages/ai/test/oauth.ts)
# ============================================================================

AUTH_PATH = Path.home() / ".otter" / "agent" / "auth.json"


def load_auth_storage() -> dict[str, dict[str, Any]]:
    """Load credentials from ``~/.otter/agent/auth.json``."""
    if not AUTH_PATH.exists():
        return {}
    try:
        return json.loads(AUTH_PATH.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_api_key(provider: str) -> str | None:
    """Resolve API key for a provider from auth storage.

    For ``api_key`` type credentials, returns the key directly.
    For ``oauth`` type credentials, returns the access token.

    Upstream reference: ``packages/ai/test/oauth.ts → resolveApiKey()``
    """
    storage = load_auth_storage()
    entry = storage.get(provider)
    if entry is None:
        return None

    if entry.get("type") == "api_key":
        return entry.get("key")

    if entry.get("type") == "oauth":
        return entry.get("accessToken")

    return None


# ============================================================================
# Credential check helpers (mirrors upstream test utility files)
# ============================================================================


def has_bedrock_credentials() -> bool:
    """Check if any valid AWS credentials are configured for Bedrock.

    Upstream reference: ``packages/ai/test/bedrock-utils.ts → hasBedrockCredentials()``
    """
    return bool(
        os.environ.get("AWS_PROFILE")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    )


def has_azure_openai_credentials() -> bool:
    """Check if Azure OpenAI credentials are configured.

    Upstream reference: ``packages/ai/test/azure-utils.ts → hasAzureOpenAICredentials()``
    """
    has_key = bool(os.environ.get("AZURE_OPENAI_API_KEY"))
    has_base_url = bool(
        os.environ.get("AZURE_OPENAI_BASE_URL") or os.environ.get("AZURE_OPENAI_RESOURCE_NAME")
    )
    return has_key and has_base_url


def resolve_azure_deployment_name(model_id: str) -> str | None:
    """Resolve Azure deployment name from ``AZURE_OPENAI_DEPLOYMENT_NAME_MAP``.

    Upstream reference: ``packages/ai/test/azure-utils.ts → resolveAzureDeploymentName()``
    """
    map_value = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "")
    if not map_value:
        return None
    for entry in map_value.split(","):
        entry = entry.strip()
        if "=" not in entry:
            continue
        mid, name = entry.split("=", 1)
        if mid.strip() == model_id:
            return name.strip()
    return None


# ============================================================================
# Pytest fixtures
# ============================================================================


@pytest.fixture
def auth_tokens():
    """Provide resolved API keys for common providers.

    Returns a dict of ``{provider: api_key_or_None}`` for providers
    that have credentials configured.
    """
    return {
        "anthropic": resolve_api_key("anthropic"),
        "github-copilot": resolve_api_key("github-copilot"),
        "google-gemini-cli": resolve_api_key("google-gemini-cli"),
        "google-antigravity": resolve_api_key("google-antigravity"),
        "openai-codex": resolve_api_key("openai-codex"),
    }
