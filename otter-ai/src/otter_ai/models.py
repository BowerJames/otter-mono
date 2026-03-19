"""Model registry and cost calculation utilities.

Provides a registry of LLM models keyed by provider, plus utility functions
for cost calculation, thinking level support checks, and model comparison.

The registry is populated from ``models_generated.py`` at module load time.
The generated file is produced by ``scripts/generate_models.py`` and should
not be edited manually.

Upstream reference: ``packages/ai/src/models.ts``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from otter_ai.models_generated import MODELS

if TYPE_CHECKING:
    from otter_ai.types import Model, Usage, UsageCost


# ============================================================================
# Model Registry
# ============================================================================

_registry: dict[str, dict[str, Model]] = {}

# Initialize registry from MODELS on module load (matching upstream).
for _provider, _models in MODELS.items():
    _registry[_provider] = dict(_models)


def get_model(provider: str, model_id: str) -> Model:
    """Look up a model by provider and model ID.

    Raises ``KeyError`` if the provider or model is not found.
    """
    provider_models = _registry.get(provider)
    if provider_models is None:
        msg = f"Unknown provider: {provider}"
        raise KeyError(msg)
    model = provider_models.get(model_id)
    if model is None:
        msg = f"Unknown model: {provider}/{model_id}"
        raise KeyError(msg)
    return model


def get_providers() -> list[str]:
    """Return all registered provider names."""
    return list(_registry.keys())


def get_models(provider: str) -> list[Model]:
    """Return all models for a given provider."""
    provider_models = _registry.get(provider)
    return list(provider_models.values()) if provider_models else []


# ============================================================================
# Cost Calculation
# ============================================================================


def calculate_cost(model: Model, usage: Usage) -> UsageCost:
    """Calculate token costs for a model given usage data.

    Mutates ``usage.cost`` in place and returns it.

    Parameters
    ----------
    model:
        The model definition (contains per-million-token cost rates).
    usage:
        The usage data from an LLM response.

    Returns
    -------
    The populated ``UsageCost`` with ``input``, ``output``, ``cache_read``,
    ``cache_write``, and ``total`` fields computed.
    """
    usage.cost.input = (model.cost.input / 1_000_000) * usage.input
    usage.cost.output = (model.cost.output / 1_000_000) * usage.output
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000) * usage.cache_write
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )
    return usage.cost


# ============================================================================
# Thinking Level Support
# ============================================================================


def supports_xhigh(model: Model) -> bool:
    """Check if a model supports the ``"xhigh"`` thinking level.

    Supported today:
    - GPT-5.2 / GPT-5.3 / GPT-5.4 model families
    - Opus 4.6 models (xhigh maps to adaptive effort "max" on
      Anthropic-compatible providers)
    """
    if "gpt-5.2" in model.id or "gpt-5.3" in model.id or "gpt-5.4" in model.id:
        return True

    return "opus-4-6" in model.id or "opus-4.6" in model.id


# ============================================================================
# Model Comparison
# ============================================================================


def models_are_equal(
    a: Model | None,
    b: Model | None,
) -> bool:
    """Check if two models are equal by comparing their ``id`` and ``provider``.

    Returns ``False`` if either model is ``None``.
    """
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider
