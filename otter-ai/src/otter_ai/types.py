"""Core type definitions for otter-ai.

Defines all shared types for the multi-provider LLM API including messages,
content blocks, models, streaming events, and options.

Upstream reference: ``packages/ai/src/types.ts``
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from otter_ai.utils.event_stream import AssistantMessageEventStream


# ============================================================================
# API & Provider Types
# ============================================================================

type KnownApi = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-gemini-cli",
    "google-vertex",
]

# ``str`` rather than ``KnownApi | str`` — Python collapses ``Literal[...] | str``
# to ``str`` at the type-checker level, so we keep it explicit.

type Api = str

type KnownProvider = Literal[
    "amazon-bedrock",
    "anthropic",
    "google",
    "google-gemini-cli",
    "google-antigravity",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "mistral",
    "minimax",
    "minimax-cn",
    "huggingface",
    "opencode",
    "opencode-go",
    "kimi-coding",
]

type Provider = str


# ============================================================================
# Thinking & Token Budgets
# ============================================================================

type ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]


@dataclass
class ThinkingBudgets:
    """Token budgets for each thinking level (token-based providers only)."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


type CacheRetention = Literal["none", "short", "long"]
type Transport = Literal["sse", "websocket", "auto"]
type StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


# ============================================================================
# Compatibility Types
# ============================================================================


@dataclass
class OpenRouterRouting:
    """OpenRouter provider routing preferences.

    Controls which upstream providers OpenRouter routes requests to.
    See: https://openrouter.ai/docs/provider-routing
    """

    only: list[str] | None = None
    order: list[str] | None = None


@dataclass
class VercelGatewayRouting:
    """Vercel AI Gateway routing preferences.

    Controls which upstream providers the gateway routes requests to.
    See: https://vercel.com/docs/ai-gateway/models-and-providers/provider-options
    """

    only: list[str] | None = None
    order: list[str] | None = None


type ThinkingFormat = Literal["openai", "openrouter", "zai", "qwen", "qwen-chat-template"]
type MaxTokensField = Literal["max_completion_tokens", "max_tokens"]


@dataclass
class OpenAICompletionsCompat:
    """Compatibility settings for OpenAI-compatible completions APIs.

    Use this to override URL-based auto-detection for custom providers.
    All fields default to ``None`` meaning "auto-detect from URL".
    """

    supports_store: bool | None = None
    supports_developer_role: bool | None = None
    supports_reasoning_effort: bool | None = None
    reasoning_effort_map: dict[ThinkingLevel, str] | None = None
    supports_usage_in_streaming: bool | None = None
    max_tokens_field: MaxTokensField | None = None
    requires_tool_result_name: bool | None = None
    requires_assistant_after_tool_result: bool | None = None
    requires_thinking_as_text: bool | None = None
    thinking_format: ThinkingFormat | None = None
    open_router_routing: OpenRouterRouting | None = None
    vercel_gateway_routing: VercelGatewayRouting | None = None
    supports_strict_mode: bool | None = None


@dataclass
class OpenAIResponsesCompat:
    """Compatibility settings for OpenAI Responses APIs.

    Reserved for future use.
    """

    pass


# ============================================================================
# Cost & Usage
# ============================================================================


@dataclass
class UsageCost:
    """Per-token cost breakdown in USD."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass
class Usage:
    """Token usage counts and computed cost for a single LLM request."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: UsageCost = field(default_factory=UsageCost)


# ============================================================================
# Content Types
# ============================================================================


@dataclass
class TextContent:
    """A block of text in an assistant or tool-result message."""

    type: Literal["text"]
    text: str
    text_signature: str | None = None


@dataclass
class ThinkingContent:
    """A thinking/reasoning block in an assistant message."""

    type: Literal["thinking"]
    thinking: str
    thinking_signature: str | None = None
    redacted: bool | None = None


@dataclass
class ImageContent:
    """An image block (base64-encoded) in a user or tool-result message."""

    type: Literal["image"]
    data: str  # base64-encoded image data
    mime_type: str  # e.g. "image/jpeg", "image/png"


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""

    type: Literal["toolCall"]
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None  # Google-specific: opaque thought context


# Convenience aliases for content unions

type Content = TextContent | ThinkingContent | ToolCall
type ToolResultContent = TextContent | ImageContent


# ============================================================================
# Message Types
# ============================================================================


@dataclass
class UserMessage:
    """A message sent by the user."""

    role: Literal["user"]
    content: str | list[ToolResultContent]
    timestamp: int  # Unix timestamp in milliseconds


@dataclass
class AssistantMessage:
    """A response from the LLM."""

    role: Literal["assistant"]
    content: list[Content]
    api: str
    provider: str
    model: str
    response_id: str | None = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = 0


@dataclass
class ToolResultMessage:
    """The result of executing a tool call, sent back to the LLM."""

    role: Literal["toolResult"]
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContent]
    details: Any = None
    is_error: bool = False
    timestamp: int = 0


type Message = UserMessage | AssistantMessage | ToolResultMessage


# ============================================================================
# Tool
# ============================================================================


@dataclass
class Tool:
    """Definition of a tool that can be called by the LLM.

    ``parameters`` is a **pydantic** ``BaseModel`` subclass.  Its JSON Schema
    (obtained via ``parameters.model_json_schema()``) is sent to the provider,
    and ``parameters.model_validate()`` is used to parse / validate incoming
    tool-call arguments.

    This replaces the upstream ``TSchema`` (TypeBox) + AJV approach with a
    single pydantic model class that serves as both schema and type.
    """

    name: str
    description: str
    parameters: type[BaseModel]


# ============================================================================
# Model
# ============================================================================


@dataclass
class ModelCost:
    """Per-million-token cost for a model."""

    input: float  # $/million tokens
    output: float
    cache_read: float
    cache_write: float


@dataclass
class Model:
    """A concrete model configuration in the unified model registry."""

    id: str
    name: str
    api: str
    provider: str
    base_url: str
    reasoning: bool
    input: list[Literal["text", "image"]]
    cost: ModelCost
    context_window: int
    max_tokens: int
    headers: dict[str, str] | None = None
    compat: OpenAICompletionsCompat | OpenAIResponsesCompat | None = None


# ============================================================================
# Context
# ============================================================================


@dataclass
class Context:
    """The conversation context passed to an LLM provider."""

    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    tools: list[Tool] | None = None


# ============================================================================
# Stream Options
# ============================================================================

type OnPayloadCallback = Callable[[Any, Model], Any]


@dataclass
class StreamOptions:
    """Base options accepted by all provider stream functions."""

    temperature: float | None = None
    max_tokens: int | None = None
    signal: asyncio.Event | None = None
    api_key: str | None = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    on_payload: OnPayloadCallback | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    """Extended stream options with reasoning/thinking controls.

    Used by ``streamSimple()`` and ``completeSimple()``.
    """

    reasoning: ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = None


# ``ProviderStreamOptions`` allows callers to pass provider-specific extra
# fields through the generic ``stream()`` entry point.  TypeScript achieves
# this with an intersection type (``StreamOptions & Record<string, unknown>``).
# In Python, providers that need extra options expose their own typed stream
# functions directly.

type ProviderStreamOptions = StreamOptions


# ============================================================================
# Stream Function Protocol
# ============================================================================


class StreamFunction(Protocol):
    """Protocol for a provider's stream function.

    Contract:

    * Must return an :class:`AssistantMessageEventStream`.
    * Request / model / runtime failures must be encoded in the returned
      stream (``stopReason`` ``"error"`` or ``"aborted"``), **not** thrown.
    """

    def __call__(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...


# ============================================================================
# Text Signature (internal, not exported from ``__init__.py``)
# ============================================================================


@dataclass
class TextSignatureV1:
    """Structured text signature for OpenAI Responses API metadata."""

    v: Literal[1]
    id: str
    phase: Literal["commentary", "final_answer"] | None = None


# ============================================================================
# Assistant Message Events
#
# Event protocol for AssistantMessageEventStream.  Streams emit ``start``
# before partial updates, then terminate with either:
#
# * ``done``  — final successful :class:`AssistantMessage`
# * ``error`` — final :class:`AssistantMessage` with stopReason
#   ``"error"`` or ``"aborted"``
# ============================================================================

# --- Text events ---


@dataclass
class AssistantMessageEventStart:
    type: Literal["start"]
    partial: AssistantMessage


@dataclass
class AssistantMessageEventTextStart:
    type: Literal["text_start"]
    content_index: int
    partial: AssistantMessage


@dataclass
class AssistantMessageEventTextDelta:
    type: Literal["text_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass
class AssistantMessageEventTextEnd:
    type: Literal["text_end"]
    content_index: int
    content: str
    partial: AssistantMessage


# --- Thinking events ---


@dataclass
class AssistantMessageEventThinkingStart:
    type: Literal["thinking_start"]
    content_index: int
    partial: AssistantMessage


@dataclass
class AssistantMessageEventThinkingDelta:
    type: Literal["thinking_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass
class AssistantMessageEventThinkingEnd:
    type: Literal["thinking_end"]
    content_index: int
    content: str
    partial: AssistantMessage


# --- Tool call events ---


@dataclass
class AssistantMessageEventToolcallStart:
    type: Literal["toolcall_start"]
    content_index: int
    partial: AssistantMessage


@dataclass
class AssistantMessageEventToolcallDelta:
    type: Literal["toolcall_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass
class AssistantMessageEventToolcallEnd:
    type: Literal["toolcall_end"]
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


# --- Terminal events ---


@dataclass
class AssistantMessageEventDone:
    type: Literal["done"]
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


@dataclass
class AssistantMessageEventError:
    type: Literal["error"]
    reason: Literal["aborted", "error"]
    error: AssistantMessage


# --- Event union ---


type AssistantMessageEvent = (
    AssistantMessageEventStart
    | AssistantMessageEventTextStart
    | AssistantMessageEventTextDelta
    | AssistantMessageEventTextEnd
    | AssistantMessageEventThinkingStart
    | AssistantMessageEventThinkingDelta
    | AssistantMessageEventThinkingEnd
    | AssistantMessageEventToolcallStart
    | AssistantMessageEventToolcallDelta
    | AssistantMessageEventToolcallEnd
    | AssistantMessageEventDone
    | AssistantMessageEventError
)
