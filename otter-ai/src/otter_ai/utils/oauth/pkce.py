"""PKCE (Proof Key for Code Exchange) utilities.

Uses the standard library ``hashlib`` and ``secrets`` modules for
cross-platform compatibility (Node.js 20+ and browsers use Web Crypto API).

Upstream reference: ``packages/ai/src/utils/oauth/pkce.ts``
"""

from __future__ import annotations

import base64
import hashlib
import secrets


def _base64url_encode(data: bytes) -> str:
    """Encode bytes as a base64url string (no padding)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Generate a PKCE code verifier and challenge.

    Returns
    -------
    tuple[str, str]
        ``(verifier, challenge)`` — the verifier is a random URL-safe
        string, and the challenge is its SHA-256 hash (base64url-encoded).
    """
    # Generate random verifier (32 bytes of randomness).
    verifier_bytes = secrets.token_bytes(32)
    verifier = _base64url_encode(verifier_bytes)

    # Compute SHA-256 challenge.
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _base64url_encode(digest)

    return verifier, challenge
