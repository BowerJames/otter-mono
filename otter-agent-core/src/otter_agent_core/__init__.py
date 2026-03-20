"""otter-agent-core: Agent runtime with tool calling and state management."""

# Types
from .types import (  # noqa: F401
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
    AgentState,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    BeforeToolCallContext,
    BeforeToolCallResult,
    StreamFn,
    ToolExecutionMode,
    ThinkingLevel,
)

# Agent loop
from .agent_loop import (  # noqa: F401
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
)

# Agent class
from .agent import Agent, AgentOptions  # noqa: F401

# Proxy utilities
from .proxy import (  # noqa: F401
    ProxyAssistantMessageEvent,
    ProxyMessageEventStream,
    ProxyStreamOptions,
    stream_proxy,
)
