"""Tests for supportsXhigh model capability check.

Upstream reference: ``packages/ai/test/supports-xhigh.test.ts``
"""

from otter_ai.models import get_model, supports_xhigh


class TestSupportsXhigh:
    def test_returns_true_for_anthropic_opus_4_6_on_anthropic_messages_api(self) -> None:
        model = get_model("anthropic", "claude-opus-4-6")
        assert model is not None
        assert supports_xhigh(model) is True

    def test_returns_false_for_non_opus_anthropic_models(self) -> None:
        model = get_model("anthropic", "claude-sonnet-4-5")
        assert model is not None
        assert supports_xhigh(model) is False

    def test_returns_true_for_gpt_5_4_models(self) -> None:
        model = get_model("openai-codex", "gpt-5.4")
        assert model is not None
        assert supports_xhigh(model) is True

    def test_returns_true_for_openrouter_opus_4_6_on_openai_completions_api(self) -> None:
        model = get_model("openrouter", "anthropic/claude-opus-4.6")
        assert model is not None
        assert supports_xhigh(model) is True
