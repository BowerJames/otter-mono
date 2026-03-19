"""otter-ai: Unified multi-provider LLM API."""

__version__ = "0.1.0"

# Core types
from .types import (  # noqa: F401
    Api,
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventDone,
    AssistantMessageEventError,
    AssistantMessageEventStart,
    AssistantMessageEventTextDelta,
    AssistantMessageEventTextEnd,
    AssistantMessageEventTextStart,
    AssistantMessageEventThinkingDelta,
    AssistantMessageEventThinkingEnd,
    AssistantMessageEventThinkingStart,
    AssistantMessageEventToolcallDelta,
    AssistantMessageEventToolcallEnd,
    AssistantMessageEventToolcallStart,
    CacheRetention,
    Content,
    Context,
    ImageContent,
    KnownApi,
    KnownProvider,
    Message,
    Model,
    ModelCost,
    OnPayloadCallback,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
    OpenRouterRouting,
    Provider,
    ProviderStreamOptions,
    SimpleStreamOptions,
    StopReason,
    StreamFunction,
    StreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultContent,
    ToolResultMessage,
    Transport,
    Usage,
    UsageCost,
    UserMessage,
    VercelGatewayRouting,
)

# Exceptions
from .exceptions import (  # noqa: F401
    ModelNotFoundError,
    OtterError,
    ProviderError,
    ValidationError,
)

# API provider registry
from .api_registry import (  # noqa: F401
    ApiProvider,
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)

# Environment variable API key resolution
from .env_api_keys import get_env_api_key  # noqa: F401

# Stream entry points (upstream: export * from "./stream.js")
# NOTE: stream.py has a side-effect import of register_builtins, so providers
# are registered the first time any of these are accessed.
from .stream import (  # noqa: F401
    complete,
    complete_simple,
    stream,
    stream_simple,
)

# Lazy stream wrappers and provider management
# (upstream: export * from "./providers/register-builtins.js")
from .providers.register_builtins import (  # noqa: F401
    register_builtin_api_providers,
    reset_api_providers,
    set_bedrock_provider_module,
    stream_anthropic,
    stream_azure_openai_responses,
    stream_bedrock,
    stream_google,
    stream_google_gemini_cli,
    stream_google_vertex,
    stream_mistral,
    stream_openai_codex_responses,
    stream_openai_completions,
    stream_openai_responses,
    stream_simple_anthropic,
    stream_simple_azure_openai_responses,
    stream_simple_bedrock,
    stream_simple_google,
    stream_simple_google_gemini_cli,
    stream_simple_google_vertex,
    stream_simple_mistral,
    stream_simple_openai_codex_responses,
    stream_simple_openai_completions,
    stream_simple_openai_responses,
)

# Event stream
from .utils.event_stream import (  # noqa: F401
    AssistantMessageEventStream,
    EventStream,
    create_assistant_message_event_stream,
)

# JSON parsing
from .utils.json_parse import parse_streaming_json  # noqa: F401

# Context overflow detection
from .utils.overflow import (  # noqa: F401
    get_overflow_patterns,
    is_context_overflow,
)

# Unicode sanitization
from .utils.sanitize_unicode import sanitize_surrogates  # noqa: F401

# Schema helpers (upstream: export * from "./utils/typebox-helpers.js")
from .utils.schema_helpers import string_enum_json_schema  # noqa: F401

# Tool call validation
from .utils.validation import (  # noqa: F401
    validate_tool_arguments,
    validate_tool_call,
)

# Model registry and utilities
from .models import (  # noqa: F401
    calculate_cost,
    get_model,
    get_models,
    get_providers,
    models_are_equal,
    supports_xhigh,
)

# ---------------------------------------------------------------------------
# Provider Options classes
#
# The upstream re-exports these as ``export type { XOptions }`` which is
# erased at compile time and does NOT trigger module loading.  Python has no
# equivalent of type-only exports, so importing an Options class from a
# provider module would eagerly load that provider's SDK.
#
# Import them directly from the provider module when needed:
#   from otter_ai.providers.anthropic import AnthropicOptions
#   from otter_ai.providers.openai_completions import OpenAICompletionsOptions
#   etc.
#
# Blocked on provider implementations:
#   GoogleGeminiCliOptions  (#13)
#   OpenAICodexResponsesOptions  (#13)
#   BedrockOptions  (#13)
# ---------------------------------------------------------------------------
