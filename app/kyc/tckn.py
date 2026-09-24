"""T.C. identity number (TCKN) validation and synthetic generation.

Rules: 11 digits, first digit non-zero,
``d10 = ((d1+d3+d5+d7+d9) * 7 - (d2+d4+d6+d8)) mod 10`` and
``d11 = (d1 + ... + d10) mod 10``.
"""

from __future__ import annotations

import hashlib


def is_valid_tckn(value: str) -> bool:
    if len(value) != 11 or not value.isdigit() or value[0] == "0":
        return False
    d = [int(ch) for ch in value]
    odd = d[0] + d[2] + d[4] + d[6] + d[8]
    even = d[1] + d[3] + d[5] + d[7]
    if (odd * 7 - even) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def complete_tckn(first_nine: str) -> str:
    """Append the two check digits to a 9-digit prefix (first digit non-zero)."""
    if len(first_nine) != 9 or not first_nine.isdigit() or first_nine[0] == "0":
        raise ValueError("prefix must be 9 digits and must not start with 0")
    d = [int(ch) for ch in first_nine]
    d10 = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    d11 = (sum(d) + d10) % 10
    return f"{first_nine}{d10}{d11}"


def synthetic_tckn(seed: str) -> str:
    """Deterministic valid TCKN for synthetic data (never a real person's)."""
    digest = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
    prefix = str(100_000_000 + digest % 899_999_999)
    return complete_tckn(prefix)
