#!/usr/bin/env python3
"""Generate models_generated.py from upstream pi-mono models.generated.ts.

Parses the upstream TypeScript MODELS constant and converts each model
entry into a Python ``Model`` dataclass instantiation.

Usage::

    uv run python otter-ai/scripts/generate_models.py [PATH_TO_UPSTREAM]

Where ``PATH_TO_UPSTREAM`` is the path to the pi-mono checkout containing
``packages/ai/src/models.generated.ts``.  Defaults to
``/root/Projects/pi-mono``.

Upstream reference:
  ``packages/ai/scripts/generate-models.ts`` → ``otter-ai/scripts/generate_models.py``
  ``packages/ai/src/models.generated.ts``  → ``otter-ai/src/otter_ai/models_generated.py``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _escape_py(s: str) -> str:
    """Escape a string for use in a Python string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _fmt_bool(val: str) -> str:
    """Convert a JS boolean to Python."""
    return "True" if val == "true" else "False"


def _fmt_cost(val: str) -> str:
    """Format a cost value — use int when possible, else float."""
    v = float(val)
    return str(int(v)) if v == int(v) else str(v)


def _fmt_headers(headers_str: str) -> str:
    """Parse a JSON headers object and format as a Python dict literal."""
    import json

    parsed = json.loads(headers_str)
    parts = [f'        "{_escape_py(k)}": "{_escape_py(v)}"' for k, v in parsed.items()]
    return "{\n" + ",\n".join(parts) + "\n    }"


def _fmt_compat(compat_str: str) -> str:
    """Parse a JSON compat object and format as an OpenAICompletionsCompat call."""
    import json

    parsed = json.loads(compat_str)
    kwargs: list[str] = []
    for k, v in parsed.items():
        py_key = {
            "supportsStore": "supports_store",
            "supportsDeveloperRole": "supports_developer_role",
            "supportsReasoningEffort": "supports_reasoning_effort",
            "thinkingFormat": "thinking_format",
        }.get(k, k)
        if isinstance(v, bool):
            kwargs.append(f"{py_key}={_fmt_bool(str(v))}")
        elif isinstance(v, str):
            kwargs.append(f'{py_key}="{v}"')
        else:
            kwargs.append(f"{py_key}={v}")
    # Multi-line format to stay under 100-char line limit
    args = (",\n                ").join(kwargs)
    return "OpenAICompletionsCompat(\n                " + args + "\n            )"


def _parse_model_block(lines: list[str]) -> dict[str, str] | None:
    """Parse a single model block from lines, returning a dict of field→value."""
    fields: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and the satisfies clause
        if not line or line.startswith("} satisfies") or line == "}," or line == "},\n":
            i += 1
            continue

        # Top-level fields
        if m := re.match(r'(\w+):\s*"(.*)",', line):
            fields[m.group(1)] = m.group(2)
            i += 1
            continue

        if m := re.match(r"(\w+):\s*(true|false),", line):
            fields[m.group(1)] = m.group(2)
            i += 1
            continue

        if m := re.match(r"(\w+):\s*(\d+),", line):
            fields[m.group(1)] = m.group(2)
            i += 1
            continue

        if m := re.match(r'(\w+):\s*\[(.*)\],', line):
            fields[m.group(1)] = m.group(2)
            i += 1
            continue

        # headers field (JSON object on one line)
        if m := re.match(r'(\w+):\s*(\{.*\}),', line):
            fields[m.group(1)] = m.group(2)
            i += 1
            continue

        # compat field (JSON object on one line)
        if m := re.match(r"(\w+):\s*(\{[^}]*\}),", line):
            fields[m.group(1)] = m.group(2)
            i += 1
            continue

        # cost block
        if re.match(r"cost:\s*\{", line):
            cost_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("}"):
                cost_lines.append(lines[i])
                i += 1
            cost_str = "".join(cost_lines)
            cost_fields: dict[str, str] = {}
            for cm in re.finditer(r'(\w+):\s*([0-9.]+),?', cost_str):
                cost_fields[cm.group(1)] = cm.group(2)
            fields["cost"] = cost_fields
            i += 1  # skip closing brace
            continue

        i += 1

    return fields if fields else None


def parse_generated_ts(source: str) -> dict[str, dict[str, dict[str, str]]]:
    """Parse upstream ``models.generated.ts`` into a nested dict.

    Returns ``{ provider: { model_id: { field: value } } }``.
    """
    result: dict[str, dict[str, dict[str, str]]] = {}

    # Split into provider blocks
    provider_pattern = re.compile(r'^\t"([^"]+)":\s*\{', re.MULTILINE)
    provider_starts = [(m.start(), m.group(1)) for m in provider_pattern.finditer(source)]

    for idx, (start, provider) in enumerate(provider_starts):
        end = provider_starts[idx + 1][0] if idx + 1 < len(provider_starts) else len(source)
        provider_block = source[start:end]

        # Find model blocks within provider
        model_pattern = re.compile(r'^\t\t"([^"]+)":\s*\{', re.MULTILINE)
        model_starts = [(m.start(), m.group(1)) for m in model_pattern.finditer(provider_block)]

        provider_models: dict[str, dict[str, str]] = {}
        for midx, (mstart, model_id) in enumerate(model_starts):
            mend = (
                model_starts[midx + 1][0]
                if midx + 1 < len(model_starts)
                else len(provider_block)
            )
            model_block = provider_block[mstart:mend]
            lines = model_block.split("\n")
            parsed = _parse_model_block(lines)
            if parsed:
                provider_models[model_id] = parsed

        result[provider] = provider_models

    return result


def generate_python(models: dict[str, dict[str, dict[str, str]]]) -> str:
    """Generate the Python ``models_generated.py`` source code."""
    parts: list[str] = []

    parts.append('"""Auto-generated model definitions.')
    parts.append("")
    parts.append("DO NOT EDIT MANUALLY — regenerate via::")
    parts.append("")
    parts.append('    uv run python otter-ai/scripts/generate_models.py')
    parts.append("")
    parts.append("Upstream reference:")
    parts.append('    packages/ai/src/models.generated.ts')
    parts.append('"""')
    parts.append("")
    parts.append("from __future__ import annotations")
    parts.append("")
    parts.append("from otter_ai.types import Model, ModelCost, OpenAICompletionsCompat")
    parts.append("")
    parts.append("MODELS: dict[str, dict[str, Model]] = {")

    for provider in sorted(models.keys()):
        provider_models = models[provider]
        parts.append(f'    "{provider}": {{')

        for model_id in sorted(provider_models.keys()):
            m = provider_models[model_id]
            parts.append(f'        "{model_id}": Model(')
            parts.append(f'            id="{_escape_py(m.get("id", model_id))}",')
            parts.append(f'            name="{_escape_py(m.get("name", model_id))}",')
            parts.append(f'            api="{_escape_py(m["api"])}",')
            parts.append(f'            provider="{_escape_py(m["provider"])}",')

            if "baseUrl" in m:
                parts.append(f'            base_url="{_escape_py(m["baseUrl"])}",')

            if "headers" in m:
                parts.append("            headers=" + _fmt_headers(m["headers"]) + ",")

            if "reasoning" in m:
                parts.append(f"            reasoning={_fmt_bool(m['reasoning'])},")

            if "input" in m:
                # Parse the ["text", "image"] array
                inputs = re.findall(r'"(\w+)"', m["input"])
                input_str = ", ".join(f'"{i}"' for i in inputs)
                parts.append(f"            input=[{input_str}],")

            if "cost" in m:
                c = m["cost"]
                parts.append("            cost=ModelCost(")
                parts.append(f"                input={_fmt_cost(c.get('input', '0'))},")
                parts.append(f"                output={_fmt_cost(c.get('output', '0'))},")
                parts.append(f"                cache_read={_fmt_cost(c.get('cacheRead', '0'))},")
                parts.append(f"                cache_write={_fmt_cost(c.get('cacheWrite', '0'))},")
                parts.append("            ),")

            if "contextWindow" in m:
                parts.append(f"            context_window={m['contextWindow']},")

            if "maxTokens" in m:
                parts.append(f"            max_tokens={m['maxTokens']},")

            if "compat" in m:
                parts.append("            compat=" + _fmt_compat(m["compat"]) + ",")

            parts.append("        ),")

        parts.append("    },")

    parts.append("}")
    parts.append("")

    return "\n".join(parts)


def main() -> None:
    upstream_path = sys.argv[1] if len(sys.argv) > 1 else "/root/Projects/pi-mono"
    ts_file = Path(upstream_path) / "packages" / "ai" / "src" / "models.generated.ts"

    if not ts_file.exists():
        print(f"Error: {ts_file} not found", file=sys.stderr)
        print("Usage: generate_models.py [PATH_TO_PI_MONO]", file=sys.stderr)
        sys.exit(1)

    source = ts_file.read_text(encoding="utf-8")
    models = parse_generated_ts(source)

    # Count
    total = sum(len(p) for p in models.values())
    print(f"Parsed {total} models across {len(models)} providers")

    python_source = generate_python(models)

    # Write output
    out_dir = Path(__file__).resolve().parent.parent / "src" / "otter_ai"
    out_file = out_dir / "models_generated.py"
    out_file.write_text(python_source, encoding="utf-8")
    n_lines = python_source.count("\n")
    print(f"Generated {out_file} ({len(python_source):,} bytes, {n_lines:,} lines)")


if __name__ == "__main__":
    main()
