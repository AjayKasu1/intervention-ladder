"""Gates decide whether a scoped intervention applies to a prompt.

Gates are a failure point of their own, so every gate decision is logged and
evaluated separately (fires on target? fires on neighbors/unrelated?).
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class AliasGate:
    """exact: an alias appears verbatim (case-sensitive substring).
    fuzzy: case/punctuation-insensitive; each alias token may differ by <= 1 edit."""

    def __init__(self, aliases: list[str], mode: str) -> None:
        if mode not in ("exact", "fuzzy"):
            raise ValueError(mode)
        self.aliases = list(aliases)
        self.mode = mode
        self._norm_aliases = [_norm(a).split() for a in aliases]

    def fires(self, prompt: str) -> bool:
        if self.mode == "exact":
            return any(a in prompt for a in self.aliases)
        words = _norm(prompt).split()
        for alias in self._norm_aliases:
            n = len(alias)
            for i in range(len(words) - n + 1):
                if all(_levenshtein(w, t) <= 1 for w, t in zip(words[i:i + n], alias)):
                    return True
        return False
