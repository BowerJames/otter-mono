"""Removes unpaired Unicode surrogate characters from a string.

Unpaired surrogates (high surrogates 0xD800-0xDBFF without matching low
surrogates 0xDC00-0xDFFF, or vice versa) cause JSON serialization errors
in many API providers.

Valid emoji and other characters outside the Basic Multilingual Plane use
properly paired surrogates and will NOT be affected.

Upstream reference: ``packages/ai/src/utils/sanitize-unicode.ts``
"""

from __future__ import annotations

import re

# Match unpaired high surrogates (not followed by low) or unpaired low
# surrogates (not preceded by high).
_UNPAIRED_SURROGATE_RE = re.compile(
    r"[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]"
)


def sanitize_surrogates(text: str) -> str:
    """Remove unpaired Unicode surrogate characters from *text*.

    Properly paired surrogates (used by valid emoji, etc.) are preserved.

    Examples
    --------
    >>> sanitize_surrogates("Hello 🙈 World")
    'Hello 🙈 World'
    >>> unpaired = chr(0xD83D)  # high surrogate without low
    >>> sanitize_surrogates(f"Text {unpaired} here")
    'Text  here'
    """
    return _UNPAIRED_SURROGATE_RE.sub("", text)
