"""Agent loop that works with AgentMessage throughout.

Transforms to ``Message[]`` only at the LLM call boundary.

Upstream reference: ``packages/agent/src/agent-loop.ts``
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from otter_ai.stream import stream_simple
from otter_ai.types import (
    AssistantMessage,
    Context,
    Message,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
)
from otter_ai.utils.event_stream import EventStream
from otter_ai.utils.validation import validate_tool_arguments

from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentEventEnd,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventMessageUpdate,
    AgentEventStart,
    AgentEventToolExecutionEnd,
    AgentEventToolExecutionStart,
    AgentEventToolExecutionUpdate,
    AgentEventTurnEnd,
    AgentEventTurnStart,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    StreamFn,
)

# ============================================================================
# Event sink
# ============================================================================

type AgentEventSink = Any  # (event: AgentEvent) -> Awaitable[None] | None


# ============================================================================
# Helpers for awaitable callables
# ============================================================================


async def _await_maybe(result: Any) -> Any:
    """Await *result* if it is a coroutine, otherwise return it."""
    if asyncio.iscoroutine(result):
        return await result
    return result


async def _get_steering_messages(config: AgentLoopConfig) -> list[AgentMessage]:
    """Safely call ``config.get_steering_messages`` and await if needed."""
    if config.get_steering_messages is None:
        return []
    return cast(
        "list[AgentMessage]",
        await _await_maybe(config.get_steering_messages()),
    )


async def _get_follow_up_messages(config: AgentLoopConfig) -> list[AgentMessage]:
    """Safely call ``config.get_follow_up_messages`` and await if needed."""
    if config.get_follow_up_messages is None:
        return []
    return cast(
        "list[AgentMessage]",
        await _await_maybe(config.get_follow_up_messages()),
    )


async def _call_before_tool_call(
    config: AgentLoopConfig,
    ctx: BeforeToolCallContext,
    signal: Any,
) -> BeforeToolCallResult | None:
    """Call ``config.before_tool_call`` and await if needed."""
    if config.before_tool_call is None:
        return None
    result = await _await_maybe(config.before_tool_call(ctx, signal))
    return cast("BeforeToolCallResult | None", result)


async def _call_after_tool_call(
    config: AgentLoopConfig,
    ctx: AfterToolCallContext,
    signal: Any,
) -> AfterToolCallResult | None:
    """Call ``config.after_tool_call`` and await if needed."""
    if config.after_tool_call is None:
        return None
    result = await _await_maybe(config.after_tool_call(ctx, signal))
    return cast("AfterToolCallResult | None", result)


# ============================================================================
# Public API: agent_loop / agent_loop_continue
# ============================================================================


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Start an agent loop with new prompt messages.

    The prompts are added to the context and events are emitted for them.
    """
    stream = _create_agent_stream()

    async def _run() -> None:
        messages = await run_agent_loop(
            prompts, context, config, signal, stream_fn,
        )
        stream.end(messages)

    asyncio.get_event_loop().create_task(_run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Continue an agent loop from the current context without a new message.

    Used for retries — context already has user message or tool results.

    **Important:** The last message in context must convert to a ``user`` or
    ``toolResult`` message via ``convert_to_llm``.
    """
    if len(context.messages) == 0:
        msg = "Cannot continue: no messages in context"
        raise RuntimeError(msg)

    last = context.messages[-1]
    if getattr(last, "role", None) == "assistant":
        msg = "Cannot continue from message role: assistant"
        raise RuntimeError(msg)

    stream = _create_agent_stream()

    async def _run() -> None:
        messages = await run_agent_loop_continue(
            context, config, signal, stream_fn,
        )
        stream.end(messages)

    asyncio.get_event_loop().create_task(_run())
    return stream


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Run the agent loop and return the new messages produced.

    This is the ``async`` counterpart of :func:`agent_loop` for callers
    that want to ``await`` the result directly.
    """
    new_messages: list[AgentMessage] = list(prompts)
    current_context = replace(context, messages=[*context.messages, *prompts])

    await _emit(AgentEventStart(type="agent_start"))
    await _emit(AgentEventTurnStart(type="turn_start"))
    for prompt in prompts:
        await _emit(AgentEventMessageStart(type="message_start", message=prompt))
        await _emit(AgentEventMessageEnd(type="message_end", message=prompt))

    await _run_loop(current_context, new_messages, config, signal, _emit, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Run the agent loop continuation and return new messages."""
    if len(context.messages) == 0:
        msg = "Cannot continue: no messages in context"
        raise RuntimeError(msg)

    last = context.messages[-1]
    if getattr(last, "role", None) == "assistant":
        msg = "Cannot continue from message role: assistant"
        raise RuntimeError(msg)

    new_messages: list[AgentMessage] = []
    current_context = replace(context)

    await _emit(AgentEventStart(type="agent_start"))
    await _emit(AgentEventTurnStart(type="turn_start"))

    await _run_loop(current_context, new_messages, config, signal, _emit, stream_fn)
    return new_messages


# ============================================================================
# Internal helpers
# ============================================================================


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    """Create an ``EventStream`` that completes on ``agent_end``."""
    return EventStream[AgentEvent, list[AgentMessage]](
        is_complete=lambda e: e.type == "agent_end",
        extract_result=lambda e: e.messages if e.type == "agent_end" else [],
    )


async def _emit(event: AgentEvent) -> None:
    """No-op default emit.

    The public ``agent_loop`` / ``agent_loop_continue`` functions use an
    ``EventStream``-based sink.  This default allows the internal helpers
    to accept an ``emit`` parameter without requiring one from callers of
    ``run_agent_loop`` / ``run_agent_loop_continue``.
    """
    pass


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
    stream_fn: StreamFn | None = None,
) -> None:
    """Main loop logic shared by agent_loop and agent_loop_continue."""
    first_turn = True

    # Check for steering messages at start
    pending_messages: list[AgentMessage] = await _get_steering_messages(config)

    # Outer loop: continues when queued follow-up messages arrive
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or len(pending_messages) > 0:
            if not first_turn:
                await emit(AgentEventTurnStart(type="turn_start"))
            else:
                first_turn = False

            # Process pending messages (inject before next assistant response)
            if len(pending_messages) > 0:
                for message in pending_messages:
                    await emit(AgentEventMessageStart(
                        type="message_start", message=message,
                    ))
                    await emit(AgentEventMessageEnd(type="message_end", message=message))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response
            message = await _stream_assistant_response(
                current_context, config, signal, emit, stream_fn,
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await emit(AgentEventTurnEnd(
                    type="turn_end", message=message, tool_results=[],
                ))
                await emit(AgentEventEnd(type="agent_end", messages=new_messages))
                return

            # Check for tool calls
            tool_calls = [
                c for c in message.content if getattr(c, "type", None) == "toolCall"
            ]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_results.extend(
                    await _execute_tool_calls(
                        current_context, message, config, signal, emit,
                    ),
                )
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await emit(AgentEventTurnEnd(
                type="turn_end", message=message, tool_results=tool_results,
            ))

            pending_messages = await _get_steering_messages(config)

        # Agent would stop here. Check for follow-up messages.
        follow_up = await _get_follow_up_messages(config)
        if len(follow_up) > 0:
            pending_messages = follow_up
            continue

        # No more messages, exit
        break

    await emit(AgentEventEnd(type="agent_end", messages=new_messages))


# ============================================================================
# Stream assistant response
# ============================================================================


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
    stream_fn: StreamFn | None = None,
) -> AssistantMessage:
    """Stream an assistant response from the LLM.

    This is where ``AgentMessage[]`` gets transformed to ``Message[]``
    for the LLM.
    """
    # Apply context transform if configured (AgentMessage[] → AgentMessage[])
    messages = context.messages
    if config.transform_context is not None:
        messages = cast(
            "list[AgentMessage]",
            await _await_maybe(config.transform_context(messages, signal)),
        )

    # Convert to LLM-compatible messages (AgentMessage[] → Message[])
    llm_messages = cast(
        "list[Message]",
        await _await_maybe(config.convert_to_llm(messages)),
    )

    # Build LLM context
    llm_context: Context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=cast("list[Tool] | None", context.tools),
    )

    # Resolve API key (important for expiring tokens)
    resolved_api_key: str | None = config.api_key
    if config.get_api_key is not None:
        key = cast(
            "str | None",
            await _await_maybe(config.get_api_key(config.model.provider)),
        )
        if key is not None:
            resolved_api_key = key

    # Build stream options
    from otter_ai.types import SimpleStreamOptions

    options = SimpleStreamOptions(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        reasoning=config.reasoning,
        thinking_budgets=config.thinking_budgets,
        signal=signal,
        api_key=resolved_api_key,
        transport=config.transport,
        cache_retention=config.cache_retention,
        session_id=config.session_id,
        on_payload=config.on_payload,
        headers=config.headers,
        max_retry_delay_ms=config.max_retry_delay_ms,
        metadata=config.metadata,
    )

    response = (stream_fn or stream_simple)(config.model, llm_context, options)

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        match event.type:
            case "start":
                partial_message = event.partial
                context.messages.append(partial_message)
                added_partial = True
                await emit(AgentEventMessageStart(
                    type="message_start",
                    message=replace(partial_message),
                ))

            case (
                "text_start" | "text_delta" | "text_end"
                | "thinking_start" | "thinking_delta" | "thinking_end"
                | "toolcall_start" | "toolcall_delta" | "toolcall_end"
            ):
                if partial_message is not None:
                    partial_message = event.partial
                    context.messages[-1] = partial_message
                    await emit(AgentEventMessageUpdate(
                        type="message_update",
                        assistant_message_event=event,
                        message=replace(partial_message),
                    ))

            case "done" | "error":
                final_message = await response.result()
                if added_partial:
                    context.messages[-1] = final_message
                else:
                    context.messages.append(final_message)
                if not added_partial:
                    await emit(AgentEventMessageStart(
                        type="message_start",
                        message=replace(final_message),
                    ))
                await emit(AgentEventMessageEnd(
                    type="message_end", message=final_message,
                ))
                return final_message

    # Fallback: stream ended without terminal event (should not happen)
    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await emit(AgentEventMessageStart(
            type="message_start",
            message=replace(final_message),
        ))
    await emit(AgentEventMessageEnd(type="message_end", message=final_message))
    return final_message


# ============================================================================
# Tool call execution
# ============================================================================


@dataclass
class _PreparedToolCall:
    kind: Literal["prepared"]
    tool_call: ToolCall
    tool: AgentTool
    args: Any


@dataclass
class _ImmediateToolCallOutcome:
    kind: Literal["immediate"]
    result: AgentToolResult[Any]
    is_error: bool


@dataclass
class _ExecutedToolCallOutcome:
    result: AgentToolResult[Any]
    is_error: bool


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    """Execute tool calls from an assistant message."""
    tool_calls = cast(
        "list[ToolCall]",
        [c for c in assistant_message.content if getattr(c, "type", None) == "toolCall"],
    )
    if config.tool_execution == "sequential":
        return await _execute_tool_calls_sequential(
            current_context, assistant_message, tool_calls, config, signal, emit,
        )
    return await _execute_tool_calls_parallel(
        current_context, assistant_message, tool_calls, config, signal, emit,
    )


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await emit(AgentEventToolExecutionStart(
            type="tool_execution_start",
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ))

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal,
        )
        if isinstance(preparation, _ImmediateToolCallOutcome):
            results.append(await _emit_tool_call_outcome(
                tool_call, preparation.result, preparation.is_error, emit,
            ))
        else:
            executed = await _execute_prepared_tool_call(
                preparation, signal, emit,
            )
            results.append(await _finalize_executed_tool_call(
                current_context, assistant_message, preparation,
                executed, config, signal, emit,
            ))

    return results


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []
    runnable_calls: list[_PreparedToolCall] = []

    for tool_call in tool_calls:
        await emit(AgentEventToolExecutionStart(
            type="tool_execution_start",
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ))

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal,
        )
        if isinstance(preparation, _ImmediateToolCallOutcome):
            results.append(await _emit_tool_call_outcome(
                tool_call, preparation.result, preparation.is_error, emit,
            ))
        else:
            runnable_calls.append(preparation)

    # Start all prepared executions concurrently
    async def _run_one(
        prepared: _PreparedToolCall,
    ) -> tuple[_PreparedToolCall, _ExecutedToolCallOutcome]:
        executed = await _execute_prepared_tool_call(prepared, signal, emit)
        return prepared, executed

    tasks = [asyncio.create_task(_run_one(p)) for p in runnable_calls]
    for task in tasks:
        prepared, executed = await task
        results.append(await _finalize_executed_tool_call(
            current_context, assistant_message, prepared,
            executed, config, signal, emit,
        ))

    return results


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> _PreparedToolCall | _ImmediateToolCallOutcome:
    """Validate arguments and check before_tool_call hook."""
    tool: AgentTool | None = None
    if current_context.tools is not None:
        tool = next(
            (t for t in current_context.tools if t.name == tool_call.name),
            None,
        )

    if tool is None:
        return _ImmediateToolCallOutcome(
            kind="immediate",
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        validated_args = validate_tool_arguments(
            cast("Tool", tool), tool_call,
        )

        before_result = await _call_before_tool_call(
            config,
            BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=validated_args,
                context=current_context,
            ),
            signal,
        )

        if before_result is not None and before_result.block:
            return _ImmediateToolCallOutcome(
                kind="immediate",
                result=_create_error_tool_result(
                    before_result.reason or "Tool execution was blocked",
                ),
                is_error=True,
            )

        return _PreparedToolCall(
            kind="prepared",
            tool_call=tool_call,
            tool=tool,
            args=validated_args,
        )
    except Exception as exc:
        return _ImmediateToolCallOutcome(
            kind="immediate",
            result=_create_error_tool_result(str(exc)),
            is_error=True,
        )


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallOutcome:
    """Execute a prepared tool call, emitting update events."""
    update_events: list[asyncio.Task[None]] = []

    def _on_update(partial_result: AgentToolResult[Any]) -> None:
        update_events.append(asyncio.create_task(emit(
            AgentEventToolExecutionUpdate(
                type="tool_execution_update",
                tool_call_id=prepared.tool_call.id,
                tool_name=prepared.tool_call.name,
                args=prepared.tool_call.arguments,
                partial_result=partial_result,
            ),
        )))

    try:
        execute_result = prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            signal,
            _on_update,
        )
        result = await _await_maybe(execute_result)
        await asyncio.gather(*update_events, return_exceptions=True)
        return _ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as exc:
        await asyncio.gather(*update_events, return_exceptions=True)
        return _ExecutedToolCallOutcome(
            result=_create_error_tool_result(str(exc)),
            is_error=True,
        )


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> ToolResultMessage:
    """Apply after_tool_call hook and emit the final outcome."""
    result = executed.result
    is_error = executed.is_error

    after_result = await _call_after_tool_call(
        config,
        AfterToolCallContext(
            assistant_message=assistant_message,
            tool_call=prepared.tool_call,
            args=prepared.args,
            result=result,
            is_error=is_error,
            context=current_context,
        ),
        signal,
    )

    if after_result is not None:
        new_content = (
            after_result.content
            if after_result.content is not None
            else result.content
        )
        new_details = (
            after_result.details
            if after_result.details is not None
            else result.details
        )
        result = AgentToolResult(content=new_content, details=new_details)
        if after_result.is_error is not None:
            is_error = after_result.is_error

    return await _emit_tool_call_outcome(prepared.tool_call, result, is_error, emit)


def _create_error_tool_result(message: str) -> AgentToolResult[Any]:
    """Create an error tool result with the given message."""
    return AgentToolResult(
        content=[TextContent(type="text", text=message)],
        details={},
    )


async def _emit_tool_call_outcome(
    tool_call: ToolCall,
    result: AgentToolResult[Any],
    is_error: bool,
    emit: AgentEventSink,
) -> ToolResultMessage:
    """Emit tool_execution_end, message_start, message_end and return the result."""
    await emit(AgentEventToolExecutionEnd(
        type="tool_execution_end",
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        result=result,
        is_error=is_error,
    ))

    tool_result_message = ToolResultMessage(
        role="toolResult",
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
        timestamp=int(time.time() * 1000),
    )

    await emit(AgentEventMessageStart(
        type="message_start", message=tool_result_message,
    ))
    await emit(AgentEventMessageEnd(type="message_end", message=tool_result_message))
    return tool_result_message
