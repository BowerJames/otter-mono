"""Stateful Agent class that wraps the agent loop.

Manages conversation state, steering/follow-up message queues, event
listeners, abort control, and exposes a high-level ``prompt()`` /
``continue()`` API.

Upstream reference: ``packages/agent/src/agent.ts``
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from otter_ai.models import get_model
from otter_ai.stream import stream_simple
from otter_ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    Transport,
    Usage,
    UsageCost,
)

from .agent_loop import agent_loop, agent_loop_continue
from .types import (
    AgentContext,
    AgentEvent,
    AgentEventEnd,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    StreamFn,
    ToolExecutionMode,
)

if TYPE_CHECKING:
    from otter_ai.types import ThinkingBudgets


# ============================================================================
# Default convertToLlm
# ============================================================================


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Keep only LLM-compatible messages (user/assistant/toolResult).

    This is the upstream's ``defaultConvertToLlm``.
    """
    return [m for m in messages if getattr(m, "role", None) in ("user", "assistant", "toolResult")]


# ============================================================================
# AgentOptions
# ============================================================================


@dataclass
class AgentOptions:
    """Configuration options for constructing an :class:`Agent`.

    Mirrors the upstream ``AgentOptions`` interface.
    """

    initial_state: dict[str, Any] | None = None
    convert_to_llm: Any = None
    transform_context: Any = None
    steering_mode: Literal["all", "one-at-a-time"] | None = None
    follow_up_mode: Literal["all", "one-at-a-time"] | None = None
    stream_fn: StreamFn | None = None
    session_id: str | None = None
    get_api_key: Any = None
    on_payload: Any = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport | None = None
    max_retry_delay_ms: int | None = None
    tool_execution: ToolExecutionMode | None = None
    before_tool_call: Any = None
    after_tool_call: Any = None


# ============================================================================
# Agent
# ============================================================================


class Agent:
    """Stateful agent that manages conversation state, tool execution, and
    event emission.

    Wraps :func:`run_agent_loop` / :func:`run_agent_loop_continue` with a
    high-level API for prompting, steering, and aborting.

    Upstream reference: ``packages/agent/src/agent.ts``
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, opts: AgentOptions | None = None) -> None:
        opts = opts or AgentOptions()

        # State — use defaults matching upstream
        self._state = AgentState(
            system_prompt="",
            model=get_model("google", "gemini-2.5-flash-lite-preview-06-17"),
            thinking_level="off",
            tools=[],
            messages=[],
        )

        if opts.initial_state:
            for key, value in opts.initial_state.items():
                # Map camelCase keys to snake_case attrs on AgentState
                py_key = _camel_to_snake(key)
                if hasattr(self._state, py_key):
                    setattr(self._state, py_key, value)

        self._convert_to_llm: Any = opts.convert_to_llm or _default_convert_to_llm
        self._transform_context: Any = opts.transform_context

        # Steering / follow-up queues
        self._steering_mode: Literal["all", "one-at-a-time"] = opts.steering_mode or "one-at-a-time"
        self._follow_up_mode: Literal["all", "one-at-a-time"] = (
            opts.follow_up_mode or "one-at-a-time"
        )
        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []

        # Stream function
        self.stream_fn: StreamFn = opts.stream_fn or stream_simple

        # Session & auth
        self._session_id: str | None = opts.session_id
        self.get_api_key: Any = opts.get_api_key

        # LLM options
        self._on_payload: Any = opts.on_payload
        self._thinking_budgets: ThinkingBudgets | None = opts.thinking_budgets
        self._transport: Transport = opts.transport or "sse"
        self._max_retry_delay_ms: int | None = opts.max_retry_delay_ms

        # Tool execution
        self._tool_execution: ToolExecutionMode = opts.tool_execution or "parallel"
        self._before_tool_call: Any = opts.before_tool_call
        self._after_tool_call: Any = opts.after_tool_call

        # Event system
        self._listeners: set[Callable[[AgentEvent], Any]] = set()

        # Abort / running state
        self._abort_signal: asyncio.Event | None = None
        self._running_prompt: asyncio.Future[None] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """The current agent state."""
        return self._state

    @property
    def session_id(self) -> str | None:
        """Current session ID used for provider caching."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._session_id = value

    @property
    def thinking_budgets(self) -> ThinkingBudgets | None:
        """Custom thinking budgets for token-based providers."""
        return self._thinking_budgets

    @thinking_budgets.setter
    def thinking_budgets(
        self,
        value: ThinkingBudgets | None,
    ) -> None:
        self._thinking_budgets = value

    @property
    def transport(self) -> Transport:
        """Current preferred transport."""
        return self._transport

    def set_transport(self, value: Transport) -> None:
        """Set the preferred transport."""
        self._transport = value

    @property
    def max_retry_delay_ms(self) -> int | None:
        """Maximum delay to wait for server-requested retries."""
        return self._max_retry_delay_ms

    @max_retry_delay_ms.setter
    def max_retry_delay_ms(self, value: int | None) -> None:
        self._max_retry_delay_ms = value

    @property
    def tool_execution(self) -> ToolExecutionMode:
        """Current tool execution mode."""
        return self._tool_execution

    def set_tool_execution(self, value: ToolExecutionMode) -> None:
        """Set the tool execution mode."""
        self._tool_execution = value

    def set_before_tool_call(self, value: Any) -> None:
        self._before_tool_call = value

    def set_after_tool_call(self, value: Any) -> None:
        self._after_tool_call = value

    # ------------------------------------------------------------------
    # Steering / follow-up mode
    # ------------------------------------------------------------------

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._steering_mode = mode

    def get_steering_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._steering_mode

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._follow_up_mode = mode

    def get_follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._follow_up_mode

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    def subscribe(
        self,
        fn: Callable[[AgentEvent], Any],
    ) -> Callable[[], None]:
        """Subscribe to agent events.  Returns an unsubscribe function."""
        self._listeners.add(fn)

        def unsubscribe() -> None:
            self._listeners.discard(fn)

        return unsubscribe

    # ------------------------------------------------------------------
    # State mutators
    # ------------------------------------------------------------------

    def set_system_prompt(self, v: str) -> None:
        self._state.system_prompt = v

    def set_model(self, m: Model) -> None:
        self._state.model = m

    def set_thinking_level(
        self,
        level: Literal["off", "minimal", "low", "medium", "high", "xhigh"],
    ) -> None:
        self._state.thinking_level = level

    def set_tools(self, t: list[AgentTool]) -> None:
        self._state.tools = t

    def replace_messages(self, ms: list[AgentMessage]) -> None:
        self._state.messages = list(ms)

    def append_message(self, m: AgentMessage) -> None:
        self._state.messages = [*self._state.messages, m]

    def clear_messages(self) -> None:
        self._state.messages = []

    # ------------------------------------------------------------------
    # Steering & follow-up queues
    # ------------------------------------------------------------------

    def steer(self, m: AgentMessage) -> None:
        """Queue a steering message while the agent is running.

        Delivered after the current assistant turn finishes executing its
        tool calls, before the next LLM call.
        """
        self._steering_queue.append(m)

    def follow_up(self, m: AgentMessage) -> None:
        """Queue a follow-up message to be processed after the agent finishes.

        Delivered only when the agent has no more tool calls or steering
        messages.
        """
        self._follow_up_queue.append(m)

    def clear_steering_queue(self) -> None:
        self._steering_queue = []

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue = []

    def clear_all_queues(self) -> None:
        self._steering_queue = []
        self._follow_up_queue = []

    def has_queued_messages(self) -> bool:
        return len(self._steering_queue) > 0 or len(self._follow_up_queue) > 0

    # ------------------------------------------------------------------
    # Internal: queue dequeue
    # ------------------------------------------------------------------

    def _dequeue_steering_messages(self) -> list[AgentMessage]:
        if self._steering_mode == "one-at-a-time":
            if self._steering_queue:
                first = self._steering_queue[0]
                self._steering_queue = self._steering_queue[1:]
                return [first]
            return []
        steering = list(self._steering_queue)
        self._steering_queue = []
        return steering

    def _dequeue_follow_up_messages(self) -> list[AgentMessage]:
        if self._follow_up_mode == "one-at-a-time":
            if self._follow_up_queue:
                first = self._follow_up_queue[0]
                self._follow_up_queue = self._follow_up_queue[1:]
                return [first]
            return []
        follow_up = list(self._follow_up_queue)
        self._follow_up_queue = []
        return follow_up

    # ------------------------------------------------------------------
    # Abort & lifecycle
    # ------------------------------------------------------------------

    def abort(self) -> None:
        """Abort the current agent run."""
        if self._abort_signal is not None:
            self._abort_signal.set()

    def wait_for_idle(self) -> Awaitable[None]:
        """Wait until the current prompt finishes processing."""
        prompt = self._running_prompt
        if prompt is not None and not prompt.done():
            return prompt

        # Return an already-resolved awaitable
        async def _done() -> None:
            pass

        return _done()

    def reset(self) -> None:
        """Reset the agent state, queues, and streaming status."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self._steering_queue = []
        self._follow_up_queue = []

    # ------------------------------------------------------------------
    # Prompt (overloaded)
    # ------------------------------------------------------------------

    @overload
    async def prompt(self, prompt_input: AgentMessage, *, images: None = None) -> None: ...
    @overload
    async def prompt(self, prompt_input: list[AgentMessage], *, images: None = None) -> None: ...
    @overload
    async def prompt(
        self,
        prompt_input: str,
        *,
        images: list[ImageContent] | None = None,
    ) -> None: ...

    async def prompt(
        self,
        prompt_input: str | AgentMessage | list[AgentMessage],
        *,
        images: list[ImageContent] | None = None,
    ) -> None:
        """Send a prompt to the agent.

        Parameters
        ----------
        prompt_input:
            A string, a single :class:`AgentMessage`, or a list of
            :class:`AgentMessage`\\ s.
        images:
            Optional images to attach (only when *prompt_input* is a ``str``).
        """
        if self._state.is_streaming:
            msg = (
                "Agent is already processing a prompt. "
                "Use steer() or follow_up() to queue messages, "
                "or wait for completion."
            )
            raise RuntimeError(msg)

        msgs: list[AgentMessage]

        msgs: list[AgentMessage] = []

        if isinstance(prompt_input, list):
            msgs = list(prompt_input)  # type: ignore[assignment]
        elif isinstance(prompt_input, str):
            from otter_ai.types import UserMessage

            content: list[TextContent | ImageContent] = [
                TextContent(type="text", text=prompt_input),
            ]
            if images:
                content.extend(images)
            msgs = [
                UserMessage(
                    role="user",
                    content=content,
                    timestamp=int(time.time() * 1000),
                ),
            ]
        else:
            msgs = [prompt_input]  # type: ignore[list-item]

        await self._run_loop(msgs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Continue
    # ------------------------------------------------------------------

    async def continue_run(self) -> None:
        """Continue from current context.

        Used for retries and resuming queued messages.  If the last message
        is an assistant message and there are queued steering/follow-up
        messages, those are processed instead.
        """
        if self._state.is_streaming:
            msg = "Agent is already processing. Wait for completion before continuing."
            raise RuntimeError(msg)

        messages = self._state.messages
        if len(messages) == 0:
            raise RuntimeError("No messages to continue from")

        last = messages[-1]
        if getattr(last, "role", None) == "assistant":
            queued_steering = self._dequeue_steering_messages()
            if queued_steering:
                await self._run_loop(
                    queued_steering,
                    skip_initial_steering_poll=True,
                )
                return

            queued_follow_up = self._dequeue_follow_up_messages()
            if queued_follow_up:
                await self._run_loop(queued_follow_up)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_loop(None)

    # ------------------------------------------------------------------
    # Internal: process loop events → update state + emit
    # ------------------------------------------------------------------

    def _process_loop_event(self, event: AgentEvent) -> None:
        """Update internal state from an agent-loop event, then emit."""
        match event.type:
            case "message_start":
                self._state.stream_message = event.message
            case "message_update":
                self._state.stream_message = event.message
            case "message_end":
                self._state.stream_message = None
                self.append_message(event.message)
            case "tool_execution_start":
                self._state.pending_tool_calls.add(event.tool_call_id)
            case "tool_execution_end":
                self._state.pending_tool_calls.discard(event.tool_call_id)
            case "turn_end":
                if getattr(event.message, "role", None) == "assistant" and getattr(
                    event.message, "error_message", None
                ):
                    self._state.error = getattr(
                        event.message,
                        "error_message",
                        None,
                    )
            case "agent_end":
                self._state.is_streaming = False
                self._state.stream_message = None
            case _:
                pass  # agent_start, turn_start, tool_execution_update — no state change

        self._emit(event)

    # ------------------------------------------------------------------
    # Internal: run the loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        messages: list[AgentMessage] | None,
        *,
        skip_initial_steering_poll: bool = False,
    ) -> None:
        """Run the agent loop.

        If *messages* is provided, starts a new turn.  Otherwise, continues
        from existing context.
        """
        model = self._state.model

        # Set up running-promise and abort signal
        loop = asyncio.get_running_loop()
        self._running_prompt = loop.create_future()
        self._abort_signal = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        reasoning: str | None = None
        if self._state.thinking_level != "off":
            reasoning = self._state.thinking_level

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=self._state.tools,
        )

        skip_steering = skip_initial_steering_poll

        config = AgentLoopConfig(
            model=model,
            reasoning=reasoning,
            session_id=self._session_id,
            on_payload=self._on_payload,
            transport=self._transport,
            thinking_budgets=self._thinking_budgets,
            max_retry_delay_ms=self._max_retry_delay_ms,
            tool_execution=self._tool_execution,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=self._make_get_steering(skip_steering),
            get_follow_up_messages=lambda: self._dequeue_follow_up_messages(),
        )

        try:
            if messages is not None:
                stream = agent_loop(
                    messages,
                    context,
                    config,
                    self._abort_signal,
                    self.stream_fn,
                )
            else:
                stream = agent_loop_continue(
                    context,
                    config,
                    self._abort_signal,
                    self.stream_fn,
                )

            # Consume events from the stream, processing each one
            async for event in stream:
                self._process_loop_event(event)

        except Exception as err:
            is_aborted = self._abort_signal.is_set() if self._abort_signal else False
            error_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="")],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(cost=UsageCost()),
                stop_reason="aborted" if is_aborted else "error",
                error_message=str(err),
                timestamp=int(time.time() * 1000),
            )

            self.append_message(error_msg)
            self._state.error = str(err)
            self._emit(AgentEventEnd(type="agent_end", messages=[error_msg]))
        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._abort_signal = None
            prompt = self._running_prompt
            self._running_prompt = None
            if prompt is not None and not prompt.done():  # type: ignore[unnecessary]
                prompt.set_result(None)

    # ------------------------------------------------------------------
    # Internal: steering message getter (captures mutable skip flag)
    # ------------------------------------------------------------------

    def _make_get_steering(
        self,
        initial_skip: bool,
    ) -> Any:
        """Create a ``get_steering_messages`` callback that optionally skips
        the first poll (matching upstream's ``skipInitialSteeringPoll``).

        We use a mutable list to share the ``skip`` flag between the
        closure and the config construction site.
        """
        skip: list[bool] = [initial_skip]

        def _get() -> list[AgentMessage]:
            if skip[0]:
                skip[0] = False
                return []
            return self._dequeue_steering_messages()

        return _get

    # ------------------------------------------------------------------
    # Internal: event emission
    # ------------------------------------------------------------------

    def _emit(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            listener(event)


# ============================================================================
# Helpers
# ============================================================================


_CAMEL_TO_SNAKE_CACHE: dict[str, str] = {}


def _camel_to_snake(name: str) -> str:
    """Convert a camelCase string to snake_case."""
    cached = _CAMEL_TO_SNAKE_CACHE.get(name)
    if cached is not None:
        return cached
    result = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    _CAMEL_TO_SNAKE_CACHE[name] = result
    return result
