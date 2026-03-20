"""Shared provider utilities for building stream options and reasoning budgets.

Upstream reference: ``packages/ai/src/providers/simple-options.ts``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otter_ai.types import (
        Model,
        SimpleStreamOptions,
        StreamOptions,
        ThinkingBudgets,
        ThinkingLevel,
    )


def build_base_options(
    model: Model,
    options: SimpleStreamOptions | None = None,
    api_key: str | None = None,
) -> StreamOptions:
    """Build a :class:`StreamOptions` from a model config and simple options.

    Falls back to ``min(model.maxTokens, 32000)`` for ``max_tokens`` when
    not explicitly provided.
    """
    from otter_ai.types import StreamOptions

    return StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=(
            options.max_tokens if options and options.max_tokens else min(model.max_tokens, 32000)
        ),
        signal=options.signal if options else None,
        api_key=api_key or (options.api_key if options else None),
        transport=options.transport if options else None,
        cache_retention=options.cache_retention if options else None,
        session_id=options.session_id if options else None,
        on_payload=options.on_payload if options else None,
        headers=options.headers if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else None,
        metadata=options.metadata if options else None,
    )


def clamp_reasoning(effort: ThinkingLevel | None) -> ThinkingLevel | None:
    """Clamp ``"xhigh"`` to ``"high"`` for providers that don't support it."""
    if effort == "xhigh":
        return "high"
    return effort


def adjust_max_tokens_for_thinking(
    base_max_tokens: int,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> tuple[int, int]:
    """Adjust ``maxTokens`` to account for a thinking/reasoning budget.

    Returns ``(maxTokens, thinkingBudget)``.
    """
    default_budgets = ThinkingBudgets(
        minimal=1024,
        low=2048,
        medium=8192,
        high=16384,
    )

    def _or_default(
        custom: ThinkingBudgets | None,
        attr: str,
        default: int | None,
    ) -> int | None:
        if custom is not None:
            val = getattr(custom, attr)
            if val is not None:
                return val
        return default

    budgets = ThinkingBudgets(
        minimal=_or_default(custom_budgets, "minimal", default_budgets.minimal),
        low=_or_default(custom_budgets, "low", default_budgets.low),
        medium=_or_default(custom_budgets, "medium", default_budgets.medium),
        high=_or_default(custom_budgets, "high", default_budgets.high),
    )

    min_output_tokens = 1024
    level = clamp_reasoning(reasoning_level)
    assert level is not None  # noqa: S101

    budget_by_level: dict[str, int | None] = {
        "minimal": budgets.minimal,
        "low": budgets.low,
        "medium": budgets.medium,
        "high": budgets.high,
    }
    thinking_budget = budget_by_level.get(level) or 0
    max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return max_tokens, thinking_budget
