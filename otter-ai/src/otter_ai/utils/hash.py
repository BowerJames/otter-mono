"""Fast deterministic hash to shorten long strings.

Upstream reference: ``packages/ai/src/utils/hash.ts``
"""

from __future__ import annotations


def short_hash(s: str) -> str:
    """Return a short deterministic hash string for *s*.

    Uses the same algorithm as the upstream TypeScript implementation
    (murmur-inspired hash with ``Math.imul``-equivalent 32-bit multiplication).
    """
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    mask = 0xFFFFFFFF

    for ch in s:
        c = ord(ch)
        h1 = (_imul(h1 ^ c, 2654435761)) & mask
        h2 = (_imul(h2 ^ c, 1597334677)) & mask

    h1 = (_imul(h1 ^ (h1 >> 16), 2246822507) ^ _imul(h2 ^ (h2 >> 13), 3266489909)) & mask
    h2 = (_imul(h2 ^ (h2 >> 16), 2246822507) ^ _imul(h1 ^ (h1 >> 13), 3266489909)) & mask

    return _to_base36(h2) + _to_base36(h1)


def _to_base36(n: int) -> str:
    """Convert a non-negative integer to a base-36 string."""
    if n == 0:
        return "0"
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    chars: list[str] = []
    while n > 0:
        n, remainder = divmod(n, 36)
        chars.append(alphabet[remainder])
    return "".join(reversed(chars))


def _imul(a: int, b: int) -> int:
    """32-bit integer multiplication (equivalent to JavaScript ``Math.imul``).

    Python integers have arbitrary precision, so we truncate to 32 bits and
    handle the signed multiplication semantics.
    """
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    result = a * b
    # Take lower 32 bits and interpret as signed
    result &= 0xFFFFFFFF
    return result
