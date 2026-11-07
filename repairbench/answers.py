"""Answer matching (pure Python)."""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s.lower())).strip()


def answer_matches(pred: str, gold: str, aliases: list[str] = ()) -> bool:
    p = f" {normalize_answer(pred)} "
    return any(f" {normalize_answer(g)} " in p for g in [gold, *aliases])
