"""Core type definitions for otter-agent-core.

Defines agent loop configuration, tool execution, state management, and
event types for the agent runtime.

Upstream reference: ``packages/agent/src/types.ts``
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from otter_ai.types import (
        AssistantMessage,
        AssistantMessageEvent,
        Context,
        ImageContent,
        Message,
        Model,
        SimpleStreamOptions,
        TextContent,
        ToolResultMessage,
    )
    from otter_ai.utils.event_stream import AssistantMessageEventStream


# ============================================================================
# Stream Function
# ============================================================================


class StreamFn(Protocol):
    """Protocol for the stream function used by the agent loop.

    Wraps ``stream_simple`` from otter-ai.

    Contract:
    - Must not throw for request/model/runtime failures.
    - Must return an :class:`AssistantMessageEventStream`.
    - Failures must be encoded in the returned stream via protocol events and a
      final :class:`AssistantMessage` with ``stop_reason`` ``"error"`` or
      ``"aborted"`` and ``error_message``.
    """

    def __call__(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...


# ============================================================================
# Tool Execution
# ============================================================================

type ToolExecutionMode = Literal["sequential", "parallel"]

# ``AgentToolCall`` is a re-alias of ``ToolCall`` for clarity in the agent
# context.  The upstream uses ``Extract<AssistantMessage["content"][number],
# { type: "toolCall" }>`` which resolves to the ``ToolCall`` interface.

if TYPE_CHECKING:
    from otter_ai.types import ToolCall as AgentToolCall
else:
    AgentToolCall = Any  # runtime no-op; only used for type-checking


# ============================================================================
# Before / After Tool Call Hooks
# ============================================================================


@dataclass
class BeforeToolCallResult:
    """Result returned from ``before_tool_call``.

    Returning ``block=True`` prevents the tool from executing.  The loop
    emits an error tool result instead.  ``reason`` becomes the text shown
    in that error result.  If omitted, a default blocked message is used.
    """

    block: bool | None = None
    reason: str | None = None


@dataclass
class AfterToolCallResult:
    """Partial override returned from ``after_tool_call``.

    Merge semantics are field-by-field:

    - ``content``: if provided, replaces the tool result content array in full
    - ``details``: if provided, replaces the tool result details value in full
    - ``is_error``: if provided, replaces the tool result error flag

    Omitted fields keep the original executed tool result values.
    There is no deep merge for ``content`` or ``details``.
    """

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None


@dataclass
class BeforeToolCallContext:
    """Context passed to ``before_tool_call``."""

    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: AgentContext


@dataclass
class AfterToolCallContext:
    """Context passed to ``after_tool_call``."""

    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: AgentToolResult[Any]
    is_error: bool
    context: AgentContext


# ============================================================================
# Agent Loop Configuration
# ============================================================================


@dataclass
class AgentLoopConfig:
    """Configuration for a single run of the agent loop.

    Extends :class:`SimpleStreamOptions` from otter-ai with agent-specific
    settings for message conversion, tool execution, and lifecycle hooks.
    """

    model: Model

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    convert_to_llm: Callable[
        [list[AgentMessage]],
        list[Message] | Awaitable[list[Message]],
    ]
    """Convert :class:`AgentMessage`\\ s to LLM-compatible :class:`Message`\\ s.

    Each AgentMessage must be converted to a UserMessage, AssistantMessage,
    or ToolResultMessage that the LLM can understand.  AgentMessages that
    cannot be converted (e.g., UI-only notifications, status messages)
    should be filtered out.

    Contract: must not raise.  Return a safe fallback value instead.
    Raising interrupts the low-level agent loop without producing a normal
    event sequence.
    """

    transform_context: (
        Callable[
            [list[AgentMessage]],
            list[AgentMessage] | Awaitable[list[AgentMessage]],
        ]
        | None
    ) = None
    """Optional transform applied to the context before ``convert_to_llm``.

    Use for operations that work at the AgentMessage level:
    - Context window management (pruning old messages)
    - Injecting context from external sources

    Contract: must not raise.  Return the original messages or another safe
    fallback value instead.
    """

    # ------------------------------------------------------------------
    # API key resolution
    # ------------------------------------------------------------------

    get_api_key: (
        Callable[[str], str | None | Awaitable[str | None]] | None
    ) = None
    """Resolve an API key dynamically for each LLM call.

    Useful for short-lived OAuth tokens (e.g., GitHub Copilot) that may
    expire during long-running tool execution phases.

    Contract: must not raise.  Return ``None`` when no key is available.
    """

    # ------------------------------------------------------------------
    # Steering & follow-up messages
    # ------------------------------------------------------------------

    get_steering_messages: (
        Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]] | None
    ) = None
    """Return steering messages to inject into the conversation mid-run.

    Called after the current assistant turn finishes executing its tool
    calls.  If messages are returned, they are added to the context before
    the next LLM call.  Tool calls from the current assistant message are
    not skipped.

    Contract: must not raise.  Return ``[]`` when no steering messages
    are available.
    """

    get_follow_up_messages: (
        Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]] | None
    ) = None
    """Return follow-up messages to process after the agent would otherwise stop.

    Called when the agent has no more tool calls and no steering messages.
    If messages are returned, they are added to the context and the agent
    continues with another turn.

    Contract: must not raise.  Return ``[]`` when no follow-up messages
    are available.
    """

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    tool_execution: ToolExecutionMode | None = None
    """Tool execution mode.  Default: ``"parallel"``.

    - ``"sequential"``: execute tool calls one by one
    - ``"parallel"``: preflight tool calls sequentially, then execute
      allowed tools concurrently
    """

    before_tool_call: (
        Callable[
            [BeforeToolCallContext],
            BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
        ]
        | None
    ) = None
    """Called before a tool is executed, after arguments have been validated.

    Return ``BeforeToolCallResult(block=True)`` to prevent execution.
    The loop emits an error tool result instead.
    """

    after_tool_call: (
        Callable[
            [AfterToolCallContext],
            AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
        ]
        | None
    ) = None
    """Called after a tool finishes executing, before final tool events.

    Return an :class:`AfterToolCallResult` to override parts of the
    executed tool result.  Any omitted fields keep their original values.
    No deep merge is performed.
    """

    # ------------------------------------------------------------------
    # Inherited from SimpleStreamOptions
    # ------------------------------------------------------------------

    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    thinking_budgets: Any = None  # ThinkingBudgets | None — avoid circular import
    signal: Any = None  # asyncio.Event | None
    api_key: str | None = None
    transport: Any = None  # Transport | None
    cache_retention: Any = None  # CacheRetention | None
    session_id: str | None = None
    on_payload: Any = None  # OnPayloadCallback | None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] | None = None


# ============================================================================
# Thinking Level (agent scope — includes "off")
# ============================================================================

type ThinkingLevel = Literal[
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]


# ============================================================================
# Custom Agent Messages (declaration merging equivalent)
# ============================================================================

# Upstream uses TypeScript declaration merging to allow apps to extend
# CustomAgentMessages.  Python has no equivalent, so we use a generic
# type parameter on the Agent class instead.  See DECISIONS.md §3b.
#
# For now, ``AgentMessage`` is simply an alias for ``Message``.  Apps
# that need custom message types should subclass ``Agent`` and override
# the message type, or use ``Union[Message, CustomType]`` explicitly.

# ============================================================================
# Agent Message
# ============================================================================

# Base case: no custom messages registered.
# ``from __future__ import annotations`` makes this a string-based alias
# at runtime, so no runtime resolution needed.
type AgentMessage = Message


# ============================================================================
# Agent State
# ============================================================================


@dataclass
class AgentState:
    """Agent state containing all configuration and conversation data."""

    system_prompt: str
    model: Model
    thinking_level: ThinkingLevel
    tools: list[AgentTool]
    messages: list[AgentMessage]
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=lambda: set[str]())
    error: str | None = None


# ============================================================================
# Agent Tool
# ============================================================================


@dataclass
class AgentToolResult[TDetails]:
    """Result of executing an agent tool.

    Parameters
    ----------
    content:
        Content blocks supporting text and images.
    details:
        Details to be displayed in a UI or logged.
    """

    content: list[TextContent | ImageContent]
    details: TDetails


type AgentToolUpdateCallback = Callable[[AgentToolResult[Any]], None]


class AgentTool(Protocol):
    """Protocol extending :class:`Tool` with an ``execute`` function and ``label``.

    Upstream uses ``interface AgentTool<TParameters extends TSchema, TDetails>
    extends Tool<TParameters>``.  Since our ``Tool`` uses ``type[BaseModel]``
    for parameters, we mirror that here without explicit generics — the
    ``execute`` method's ``params`` is ``Any`` (validated at call site).
    """

    name: str
    description: str
    parameters: type[BaseModel]
    label: str

    def execute(
        self,
        tool_call_id: str,
        params: Any,
        signal: Any = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> Any: ...


# ============================================================================
# Agent Context
# ============================================================================


@dataclass
class AgentContext:
    """Like :class:`Context` but uses :class:`AgentTool`."""

    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None = None


# ============================================================================
# Agent Events
#
# Events emitted by the Agent for UI updates.  These events provide
# fine-grained lifecycle information for messages, turns, and tool
# executions.
# ============================================================================


# --- Agent lifecycle ---


@dataclass
class AgentEventStart:
    type: Literal["agent_start"]


@dataclass
class AgentEventEnd:
    type: Literal["agent_end"]
    messages: list[AgentMessage]


# --- Turn lifecycle ---


@dataclass
class AgentEventTurnStart:
    """A turn is one assistant response + any tool calls/results."""

    type: Literal["turn_start"]


@dataclass
class AgentEventTurnEnd:
    type: Literal["turn_end"]
    message: AgentMessage
    tool_results: list[ToolResultMessage]


# --- Message lifecycle ---


@dataclass
class AgentEventMessageStart:
    """Emitted for user, assistant, and toolResult messages."""

    type: Literal["message_start"]
    message: AgentMessage


@dataclass
class AgentEventMessageUpdate:
    """Only emitted for assistant messages during streaming."""

    type: Literal["message_update"]
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent


@dataclass
class AgentEventMessageEnd:
    type: Literal["message_end"]
    message: AgentMessage


# --- Tool execution lifecycle ---


@dataclass
class AgentEventToolExecutionStart:
    type: Literal["tool_execution_start"]
    tool_call_id: str
    tool_name: str
    args: Any


@dataclass
class AgentEventToolExecutionUpdate:
    type: Literal["tool_execution_update"]
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


@dataclass
class AgentEventToolExecutionEnd:
    type: Literal["tool_execution_end"]
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


# --- Event union ---


type AgentEvent = (
    AgentEventStart
    | AgentEventEnd
    | AgentEventTurnStart
    | AgentEventTurnEnd
    | AgentEventMessageStart
    | AgentEventMessageUpdate
    | AgentEventMessageEnd
    | AgentEventToolExecutionStart
    | AgentEventToolExecutionUpdate
    | AgentEventToolExecutionEnd
)

