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
EOF
