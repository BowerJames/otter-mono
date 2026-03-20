"""otter-agent-core: Agent runtime with tool calling and state management.

Mirrors upstream ``packages/agent/src/index.ts``:

    export * from "./agent.js";
    export * from "./agent-loop.js";
    export * from "./proxy.js";
    export * from "./types.js";
"""

# Types (upstream: export * from "./types.js")
from .types import (  # noqa: F401
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    AgentToolUpdateCallback,
    BeforeToolCallContext,
    BeforeToolCallResult,
    StreamFn,
    ToolExecutionMode,
    ThinkingLevel,
)

# Agent loop (upstream: export * from "./agent-loop.js")
from .agent_loop import (  # noqa: F401
    AgentEventSink,
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
)

# Agent class (upstream: export * from "./agent.js")
from .agent import Agent, AgentOptions  # noqa: F401

# Proxy utilities (upstream: export * from "./proxy.js")
from .proxy import (  # noqa: F401
    ProxyAssistantMessageEvent,
    ProxyStreamOptions,
    stream_proxy,
)
