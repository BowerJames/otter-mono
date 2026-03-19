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

# Schema helpers
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

# Provider utilities
from .providers.simple_options import (  # noqa: F401
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_reasoning,
)
from .providers.transform_messages import transform_messages  # noqa: F401
from .providers.github_copilot_headers import (  # noqa: F401
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
    infer_copilot_initiator,
)
from .providers.openai_completions import (  # noqa: F401
    convert_messages as convert_messages_openai,
    stream_openai_completions,
    stream_simple_openai_completions,
)
from .providers.anthropic import (  # noqa: F401
    convert_messages as convert_messages_anthropic,
    stream_anthropic,
    stream_simple_anthropic,
)
from .providers.google_shared import (  # noqa: F401
    convert_messages as convert_messages_google,
    convert_tools as convert_tools_google,
    is_thinking_part,
    map_stop_reason as map_stop_reason_google,
    map_tool_choice,
    retain_thought_signature,
)
from .providers.google import (  # noqa: F401
    stream_google,
    stream_simple_google,
)
