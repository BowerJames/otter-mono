# Design Decisions

This document captures deliberate deviations from the upstream [pi-mono](https://github.com/badlogic/pi-mono) TypeScript implementation and the reasoning behind each choice.

---

## 1. Schema & Validation System

**Upstream:** `@sinclair/typebox` (TypeBox) for compile-time JSON Schema types + `ajv` for runtime validation.

**Decision:** `pydantic` v2.

**Rationale:**
- Pydantic provides both type definition and runtime validation in a single library, replacing both TypeBox and AJV.
- `pydantic.BaseModel` classes generate JSON Schema automatically via `.model_json_schema()`, which is what providers need for tool definitions.
- Type-safe model parsing with `TypeAdapter[T]` replaces AJV's `compile()` + `validate()` pattern.
- Coercion (e.g., `str` `"123"` → `int` `123`) is built-in, matching AJV's `coerceTypes: true`.
- Widely adopted in the Python ecosystem with excellent IDE support.
- `jsonschema` library is available as a fallback for raw schema validation if needed.

**Impact:**
- `Tool.parameters` uses `type[T]` (a Pydantic model class) instead of TypeBox's `TSchema`.
- `StringEnum` helper uses `pydantic`'s `Literal` type support.
- Validation module wraps `TypeAdapter.validate_python()` instead of AJV compilation.

---

## 2. Concurrency Model

**Upstream:** Node.js single-threaded event loop with `Promise`-based async.

**Decision:** Python `asyncio` with `async/await`.

**Rationale:**
- `asyncio` is the standard Python async framework, directly analogous to Node.js's event loop.
- All provider HTTP calls use `httpx.AsyncClient`, which is fully `asyncio`-compatible.
- The `EventStream` class uses a custom waiter-based pattern with `asyncio.Event` for push/pull between producers and consumers, mirroring the upstream's promise-based approach.

**Impact:**
- `AbortSignal` (Node.js) → `asyncio.CancelledError` / manual `asyncio.Event` cancellation token.
- No thread safety concerns since everything runs on a single event loop (same as Node.js).
- Provider implementations must be fully async — no blocking I/O.

---

## 3. Type System Patterns

### 3a. Discriminated Unions

**Upstream:** TypeScript discriminated unions with `type` field.

**Decision:** Python `@typing.overload` + `@dataclass` inheritance or `Literal` type narrowing.

**Rationale:**
- Python 3.12+ supports `@typing.type_check_only` and exhaustive pattern matching via `match`/`case`.
- Each event type is a `@dataclass` with a `type: Literal[...]` field.
- Union types use `|` syntax (PEP 604).

### 3b. Declaration Merging (`CustomAgentMessages`)

**Upstream:** TypeScript declaration merging allows apps to extend `CustomAgentMessages` interface.

**Decision:** Generic type parameter on `Agent` class.

**Rationale:**
- Python has no equivalent to TypeScript declaration merging.
- Instead, `Agent[TCustomMessages]` takes a type parameter for custom message types.
- Alternatively, apps can subclass `Agent` and override the message type.

### 3c. Type Re-exports

**Upstream:** `export type { X }` for type-only re-exports.

**Decision:** All types re-exported from `__init__.py`.

**Rationale:**
- Python doesn't distinguish type-only vs value exports.
- Re-export from `__init__.py` using `from .module import X`.

### 3d. Interface Inheritance

**Upstream:** TypeScript `interface XOptions extends StreamOptions` for provider-specific option types that extend a base interface.

**Decision:** `@dataclass` inheritance — `class XOptions(StreamOptions)`.

**Rationale:**
- `@dataclass` is the standard Python pattern for structured data classes with default values.
- Inheritance from `@dataclass StreamOptions` provides the same polymorphic type relationship as TypeScript's `interface extends`.
- Provider-specific fields are declared only on the subclass; base fields (`api_key`, `max_tokens`, `temperature`, `signal`, `headers`, `on_payload`, `metadata`, `cache_retention`, `session_id`, `transport`, `max_retry_delay_ms`) are inherited.
- All provider option classes follow this pattern consistently:
  - `AnthropicOptions(StreamOptions)`
  - `GoogleOptions(StreamOptions)`
  - `GoogleVertexOptions(StreamOptions)`
  - `MistralOptions(StreamOptions)`
  - `OpenAICompletionsOptions(StreamOptions)`
  - `OpenAIResponsesOptions(StreamOptions)`
  - `AzureOpenAIResponsesOptions(StreamOptions)`

**Impact:**
- Provider options are type-compatible with `StreamOptions` wherever the base type is expected.
- New provider implementations must use `@dataclass` inheritance, not manual field duplication.

---

## 4. Module System

**Upstream:** npm workspaces + ES module `import/export`.

**Decision:** uv workspace + standard Python imports.

**Rationale:**
- uv workspaces mirror npm workspaces for monorepo management.
- Python's import system is well-understood; no need for special handling.

### 4a. Side-Effect Imports

**Upstream:** `import "./providers/register-builtins.js"` in `stream.ts` registers providers on import.

**Decision:** Same pattern — `import otter_ai.providers.register_builtins` in `stream.py`.

**Rationale:**
- Python supports side-effect imports identically to TypeScript.
- The import triggers `register_builtin_api_providers()` at module level.

### 4b. Lazy Loading

**Upstream:** Dynamic `import()` for deferred provider module loading.

**Decision:** `importlib.import_module()`.

**Rationale:**
- Direct equivalent of JavaScript's dynamic `import()`.
- Lazy loading prevents importing all provider dependencies (e.g., `boto3` for Bedrock) until first use.

---

## 5. HTTP Client

**Upstream:** `undici` (Node.js HTTP client) + provider-specific SDKs (`@anthropic-ai/sdk`, `openai`, `@google/genai`, `@mistralai/mistralai`, `@aws-sdk/client-bedrock-runtime`).

**Decision:** `httpx` for direct API calls + provider SDKs where available.

**Rationale:**
- `httpx` is the standard async HTTP client for Python, supporting HTTP/1.1, HTTP/2, SSE streaming, and connection pooling.
- Provider SDKs used where they add significant value:
  - `anthropic` Python SDK for Anthropic
  - `openai` Python SDK for OpenAI
  - `google-genai` Python SDK for Google
  - `mistralai` Python SDK for Mistral
  - `boto3` / `aioboto3` for Amazon Bedrock
- SSE parsing: `httpx` async streaming with manual line parsing (same approach as upstream).

---

## 6. Naming Conventions

**Upstream:** TypeScript `camelCase` for functions/variables, `PascalCase` for types/classes, `kebab-case` for files.

**Decision:**
- Functions/variables: `snake_case` (PEP 8)
- Classes/types: `PascalCase` (PEP 8)
- Files: `snake_case` with `.py` extension
- Constants: `UPPER_SNAKE_CASE`

**File mapping examples:**
| Upstream (TypeScript) | otter-mono (Python) |
|---|---|
| `types.ts` | `types.py` |
| `api-registry.ts` | `api_registry.py` |
| `register-builtins.ts` | `register_builtins.py` |
| `openai-completions.ts` | `openai_completions.py` |
| `google-gemini-cli.ts` | `google_gemini_cli.py` |
| `event-stream.ts` | `event_stream.py` |
| `json-parse.ts` | `json_parse.py` |
| `typebox-helpers.ts` | `schema_helpers.py` |

---

## 7. Error Handling

**Upstream:** TypeScript `throw new Error(...)`, never-return patterns via exhaustive type checks.

**Decision:** Python `raise Exception(...)` with custom exception hierarchy.

**Rationale:**
- Custom exception classes allow structured error handling:
  - `OtterError` (base)
  - `ProviderError(OtterError)` — LLM provider errors
  - `ValidationError(OtterError)` — tool argument validation
  - `ModelNotFoundError(OtterError)` — unknown model/provider
- Exhaustive checks use `assert_never()` pattern or `match`/`case` with no default branch.

---

## 8. Package Naming

**Upstream:** `@mariozechner/pi-ai`, `@mariozechner/pi-agent-core` (npm scoped packages).

**Decision:** `otter-ai`, `otter-agent-core` (PyPI-compatible names).

**Rationale:**
- No Python equivalent to npm scoped packages.
- Hyphenated names follow PyPI convention.
- Import names use underscores: `otter_ai`, `otter_agent_core`.

---

## 9. ProviderStreamOptions (Intersection Type)

**Upstream:** `type ProviderStreamOptions = StreamOptions & Record<string, unknown>` — allows callers to pass arbitrary extra fields through the generic `stream()` entry point.

**Decision:** `type ProviderStreamOptions = StreamOptions` (no extra fields).

**Rationale:**
- Python has no equivalent of TypeScript's intersection with `Record<string, unknown>`.
- TypedDict with `**kwargs` would lose type safety without gaining much.
- The lazy-loading pattern already encourages callers to use provider-specific stream functions (e.g., `stream_anthropic()`) which accept the fully-typed `AnthropicOptions`.
- The generic `stream()` entry point still works for base `StreamOptions` use cases.

**Impact:**
- Provider-specific options (e.g., `AnthropicOptions`) cannot be passed through `stream()`.
- Callers must use provider-specific stream functions for typed options.

---

## 10. Environment Variable Names

**Upstream:** Uses `PI_CACHE_RETENTION` and similar `PI_`-prefixed environment variables.

**Decision:** Use the same upstream `PI_`-prefixed environment variable names.

**Rationale:**
- Keeping the upstream env var names ensures consistency with upstream documentation, examples, and debugging guides.
- The `otter` naming convention applies to Python package/module names, not runtime configuration.
- Users referencing upstream documentation will find the same env var names.

**Impact:**
- `PI_CACHE_RETENTION` is used in Anthropic and OpenAI Responses providers (not `OTTER_CACHE_RETENTION`).
- Future env vars should follow the upstream naming unless there is a compelling Python-specific reason to differ.

---

## 11. Model Data Generation Script

**Upstream:** `generate-models.ts` (~1,600 lines) fetches model data from a `models-dev.json` source file and generates `models.generated.ts` from scratch.

**Decision:** `generate_models.py` (~250 lines) parses the upstream's already-generated `models.generated.ts` TypeScript output and converts it to Python `models_generated.py`.

**Rationale:**
- Model data originates from the upstream pi-mono project; otter-mono consumes it, not produces it.
- Parsing the generated TS file is simpler and avoids duplicating the upstream's data-fetching and validation logic.
- When upstream adds new models, regenerating otter's output is a single command against the updated upstream checkout.

**Impact:**
- Cannot generate new model data independently — requires the upstream `models.generated.ts` to exist first.
- The regex-based TS parser may need updates if the upstream output format changes significantly.

---

## 12. Proxy Serialization/Parsing Helpers

**Upstream:** `proxy.ts` (~340 lines) uses inline structural types for proxy events and relies on `JSON.parse()` with TypeScript type assertions for deserialization. The `ToolCall` `partialJson` field is stored via `(content as any).partialJson`.

**Decision:** `proxy.py` (~659 lines) uses named `@dataclass` classes for proxy events (`ProxyEventStart`, `ProxyEventTextDelta`, etc.), explicit `_parse_proxy_event()` deserialization, and `_parse_usage()`/`_parse_usage_cost()` helpers. The `partialJson` field is stored via `tc._partial_json` (type-ignored attribute).

**Rationale:**
- Named dataclasses provide better IDE autocompletion and type safety compared to inline structural types.
- Explicit deserialization (`_parse_proxy_event`) handles camelCase→snake_case mapping that the upstream gets for free via `JSON.parse` + TypeScript.
- `_to_camel_dict()` serialization helper converts Python snake_case dataclasses back to camelCase for the wire format, which upstream's `JSON.stringify` does automatically.

**Impact:**
- `proxy.py` is ~95% larger than upstream due to explicit (de)serialization code.
- `ProxyAssistantMessageEvent` is a typed union of named dataclasses (exported from `__init__.py`), whereas upstream uses an anonymous union type.
- `ProxyMessageEventStream` is exported from `__init__.py` matching upstream's `export * from "./proxy.js"`.

---

## 13. Agent.continue_run Keyword Rename

**Upstream:** `Agent.continue()` is a method on the `Agent` class.

**Decision:** Renamed to `Agent.continue_run()`.

**Rationale:**
- `continue` is a Python reserved keyword and cannot be used as a method name.

**Impact:**
- Code porting from upstream TypeScript examples must rename `agent.continue()` → `agent.continue_run()`.
- The deviation is small, necessary, and self-explanatory.

---

## 14. AgentOptions CamelCase→snake_case Mapping for initial_state

**Upstream:** `Agent` constructor spreads `opts.initialState` directly onto the default state object:

```typescript
this._state = { ...this._state, ...opts.initialState };
```

Keys in `initialState` are expected to be camelCase matching `AgentState` interface fields.

**Decision:** `Agent.__init__` applies `_camel_to_snake()` to each key in `initial_state` before setting it on the `AgentState` dataclass.

**Rationale:**
- Python dataclasses use snake_case field names (`system_prompt`, `thinking_level`).
- Users may pass camelCase keys when porting from upstream TypeScript or reading upstream docs.
- The conversion is cached and only runs once during construction.

**Impact:**
- Users can pass either camelCase or snake_case keys in `initial_state`.
- The mapping is a best-effort heuristic; unusual camelCase patterns may not convert correctly.

---

## 15. register_builtins.py Lazy-Loading Architecture

**Upstream:** `register-builtins.ts` defines a typed `LazyProviderModule<TApi, TOptions, TSimpleOptions>` interface that normalizes all provider modules to a consistent shape with `stream` and `streamSimple` methods. Each `load*ProviderModule()` function dynamically imports the provider module and maps provider-specific function names to the normalized interface (e.g., `streamAnthropic` → `stream`). Separate `createLazyStream` and `createLazySimpleStream` factories create the lazy wrapper functions.

**Decision:** `register_builtins.py` uses a single `_create_lazy_stream()` factory that takes a `load_module` callable and a `stream_attr` string attribute name. No normalized interface type is used; each lazy stream function directly references the provider-specific attribute name (e.g., `"stream_anthropic"`).

**Rationale:**
- Python's dynamic typing makes the normalized interface unnecessary — `_create_lazy_stream` just calls `getattr(mod, stream_attr)` on the loaded module.
- The upstream needs the interface because TypeScript requires explicit type annotations for the `loadModule()` return type.
- `importlib.import_module()` is the direct equivalent of JavaScript's `import()`.

**Impact:**
- The lazy-loading mechanism is functionally equivalent but structurally simpler.
- No type-level guarantee that the loaded module has the expected stream functions (would only be caught at runtime).
- The `_load_provider_module()` function uses `asyncio.ensure_future()` for fire-and-forget module loading, matching the upstream's `void loadModule().then(...)` pattern.

---

## 16. UsageCost/ModelCost Named-Type Extraction

**Upstream:** Cost structures are defined as inline anonymous types within their parent interfaces:

```typescript
// In Usage
cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
};

// In Model
cost: {
    input: number;    // $/million tokens
    output: number;
    cacheRead: number;
    cacheWrite: number;
};
```

**Decision:** Extracted as named `@dataclass` classes:

```python
@dataclass
class UsageCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0

@dataclass
class ModelCost:
    input: float
    output: float
    cache_read: float
    cache_write: float
```

**Rationale:**
- Python's `@dataclass` encourages named types for structured data.
- `UsageCost` and `ModelCost` are reused in multiple places (e.g., `Usage.cost`, `calculate_cost()` return type, proxy event deserialization).
- Named types improve readability in type annotations and IDE hover documentation.
- `UsageCost` is exported from `__init__.py` for consumer use; `ModelCost` is also exported.

**Impact:**
- `UsageCost` and `ModelCost` are public API surface that doesn't exist as separate types in upstream.
- The upstream's inline types have slightly different shapes: `Usage.cost` has a `total` field, `Model.cost` does not.

---

## 17. OAuth Type Adaptations

### 17a. OAuthCredentials Index Signature → Explicit extra Field

**Upstream:** `OAuthCredentials` uses a TypeScript index signature to allow arbitrary string keys:

```typescript
export type OAuthCredentials = {
    refresh: string;
    access: string;
    expires: number;
    [key: string]: unknown;
};
```

**Decision:** An explicit `extra: dict[str, Any]` field with a default factory:

```python
@dataclass
class OAuthCredentials:
    refresh: str
    access: str
    expires: int
    extra: dict[str, Any] = field(default_factory=dict)
```

**Rationale:**
- Python dataclasses cannot have index signatures.
- The `extra` field makes the additional data explicit and typed.

**Impact:**
- Code that accesses arbitrary keys on credentials must use `credentials.extra["key"]` instead of `credentials["key"]`.
- The `extra` field defaults to an empty dict, maintaining backward compatibility.

### 17b. OAuthProviderInterface Optional Fields

**Upstream:** `OAuthProviderInterface` has optional methods and properties:

```typescript
export interface OAuthProviderInterface {
    usesCallbackServer?: boolean;
    modifyModels?(models: Model<Api>[], credentials: OAuthCredentials): Model<Api>[];
}
```

**Decision:** The `OAuthProviderInterface` is defined as a `@runtime_checkable Protocol`. Optional fields like `usesCallbackServer` and `modifyModels` are not included in the Protocol definition.

**Rationale:**
- Python Protocols only enforce required methods. Optional methods can be added but would need `hasattr()` checks at call sites.
- `usesCallbackServer` is a metadata flag only used by the CLI, not by the core OAuth flow.
- `modifyModels` is not yet needed by any current consumer.

**Impact:**
- `usesCallbackServer` and `modifyModels` are not accessible through the Protocol type.
- When needed, they should be added to the Protocol definition with default implementations or handled via `hasattr()` checks.
EOF
