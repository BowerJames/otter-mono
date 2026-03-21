"""Tests for google-shared image tool result routing.

Upstream reference: ``packages/ai/test/google-shared-image-tool-result-routing.test.ts``
"""

from otter_ai.providers.google_shared import convert_messages
from otter_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)


def _empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=UsageCost(),
    )


def _make_model(api: str, provider: str, model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api=api,  # type: ignore[assignment]
        provider=provider,
        base_url="https://example.com",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=8192,
    )


def _make_context(model: Model) -> Context:
    now = 1000
    return Context(
        messages=[
            UserMessage(role="user", content="read the files", timestamp=now),
            AssistantMessage(
                role="assistant",
                content=[
                    ToolCall(
                        type="toolCall", id="call_a", name="read", arguments={"path": "a.txt"}
                    ),
                    ToolCall(
                        type="toolCall", id="call_img", name="read", arguments={"path": "image.png"}
                    ),
                    ToolCall(
                        type="toolCall", id="call_b", name="read", arguments={"path": "b.txt"}
                    ),
                ],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=_empty_usage(),
                stop_reason="toolUse",
                timestamp=now,
            ),
            ToolResultMessage(
                role="toolResult",
                tool_call_id="call_a",
                tool_name="read",
                content=[TextContent(type="text", text="alpha text")],
                is_error=False,
                timestamp=now,
            ),
            ToolResultMessage(
                role="toolResult",
                tool_call_id="call_img",
                tool_name="read",
                content=[ImageContent(type="image", data="abc", mime_type="image/png")],
                is_error=False,
                timestamp=now,
            ),
            ToolResultMessage(
                role="toolResult",
                tool_call_id="call_b",
                tool_name="read",
                content=[TextContent(type="text", text="beta text")],
                is_error=False,
                timestamp=now,
            ),
        ],
    )


class TestGoogleSharedImageToolResultRouting:
    def test_keeps_separate_synthetic_image_turn_for_gemini_2x_google_api(self) -> None:
        model = _make_model("google-generative-ai", "google", "gemini-2.5-flash")
        contents = convert_messages(model, _make_context(model))

        assert len(contents) == 5
        assert all(p.get("functionResponse") for p in contents[2]["parts"])
        assert contents[3]["parts"][0].get("text") == "Tool result image:"
        assert contents[3]["parts"][1].get("inlineData") is not None
        assert contents[4]["parts"][0].get("functionResponse") is not None

    def test_nests_image_tool_results_for_gemini_3_google_api(self) -> None:
        model = _make_model("google-generative-ai", "google", "gemini-3-pro-preview")
        contents = convert_messages(model, _make_context(model))

        assert len(contents) == 3
        tool_result_turn = contents[2]
        assert len(tool_result_turn["parts"]) == 3
        image_response = tool_result_turn["parts"][1].get("functionResponse")
        assert image_response is not None
        assert len(image_response["parts"]) == 1
        assert image_response["parts"][0].get("inlineData") is not None

    def test_nests_image_tool_results_for_non_gemini_antigravity(self) -> None:
        model = _make_model("google-gemini-cli", "google-antigravity", "claude-sonnet-4-6")
        contents = convert_messages(model, _make_context(model))

        assert len(contents) == 3
        tool_result_turn = contents[2]
        assert len(tool_result_turn["parts"]) == 3
        image_response = tool_result_turn["parts"][1].get("functionResponse")
        assert image_response is not None
        assert len(image_response["parts"]) == 1
        assert image_response["parts"][0].get("inlineData") is not None

    def test_keeps_separate_synthetic_image_turn_for_gemini_2x_cloud_code_assist(self) -> None:
        model = _make_model("google-gemini-cli", "google-gemini-cli", "gemini-2.5-flash")
        contents = convert_messages(model, _make_context(model))

        assert len(contents) == 5
        assert contents[3]["parts"][0].get("text") == "Tool result image:"
        assert contents[3]["parts"][1].get("inlineData") is not None
