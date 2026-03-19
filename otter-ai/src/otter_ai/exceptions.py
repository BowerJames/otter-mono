"""Custom exception hierarchy for otter-ai.

Provides structured error types that consumers can catch selectively.

Upstream reference: DECISIONS.md §7 — replaces TypeScript's generic ``Error``
with a typed hierarchy.
"""


class OtterError(Exception):
    """Base exception for all otter-ai errors."""


class ProviderError(OtterError):
    """LLM provider errors (auth failures, API errors, rate limits)."""


class ValidationError(OtterError):
    """Tool argument validation failures."""


class ModelNotFoundError(OtterError):
    """Unknown model or provider lookup failures."""
