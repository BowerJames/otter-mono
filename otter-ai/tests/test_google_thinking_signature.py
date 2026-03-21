"""Tests for Google thinking detection (thoughtSignature).

Upstream reference: ``packages/ai/test/google-thinking-signature.test.ts``
"""

from types import SimpleNamespace

from otter_ai.providers.google_shared import is_thinking_part, retain_thought_signature


class TestGoogleThinkingDetection:
    def test_treats_part_thought_true_as_thinking(self) -> None:
        assert is_thinking_part(SimpleNamespace(thought=True, thought_signature=None)) is True
        assert (
            is_thinking_part(SimpleNamespace(thought=True, thought_signature="opaque-signature"))
            is True
        )

    def test_does_not_treat_thought_signature_alone_as_thinking(self) -> None:
        # Per Google docs, thoughtSignature is for context replay and can appear on any part type.
        # Only thought === true indicates thinking content.
        # See: https://ai.google.dev/gemini-api/docs/thought-signatures
        assert (
            is_thinking_part(SimpleNamespace(thought=None, thought_signature="opaque-signature"))
            is False
        )
        assert (
            is_thinking_part(SimpleNamespace(thought=False, thought_signature="opaque-signature"))
            is False
        )

    def test_does_not_treat_empty_missing_signatures_as_thinking_if_thought_not_set(self) -> None:
        assert is_thinking_part(SimpleNamespace(thought=None, thought_signature=None)) is False
        assert is_thinking_part(SimpleNamespace(thought=False, thought_signature="")) is False

    def test_preserves_existing_signature_when_subsequent_deltas_omit(self) -> None:
        first = retain_thought_signature(None, "sig-1")
        assert first == "sig-1"

        second = retain_thought_signature(first, None)
        assert second == "sig-1"

        third = retain_thought_signature(second, "")
        assert third == "sig-1"

    def test_updates_signature_when_new_non_empty_signature_arrives(self) -> None:
        updated = retain_thought_signature("sig-1", "sig-2")
        assert updated == "sig-2"
