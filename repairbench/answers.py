"""Answer matching (pure Python)."""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]")
_SENTENCE = re.compile(r"[.!?;\n]+")
_NEGATION = re.compile(r"\b(?:no|not|never|neither|nor|cannot|\w+n['’]t)\b", re.IGNORECASE)
_ALTERNATIVE = re.compile(r"\b(?:or|either|maybe|possibly)\b")
_NUMBER = re.compile(r"(?<![\w.])[+\-−]?\d+(?:[.,]\d+)*(?!\w)")


def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s.lower())).strip()


def answer_matches(pred: str, gold: str, aliases=(), *, alternatives=()) -> bool:
    """Conservative match for short answers, optionally with a known answer vocabulary."""
    accepted = {normalize_answer(g) for g in (gold, *aliases)} - {""}
    rejected = {normalize_answer(g) for g in alternatives} - accepted - {""}
    if gold.isdigit() and {n.removeprefix("+") for n in _NUMBER.findall(pred)} != {gold}:
        return False
    matched = False
    for sentence in _SENTENCE.split(pred):
        clean = normalize_answer(sentence)
        # A standalone "Not Osaka." rejects an alternative; it does not propose it.
        if clean.startswith("not ") and clean[4:] in rejected:
            continue
        if any(f" {word} " in f" {clean} " for word in rejected):
            return False
        if any(f" {word} " in f" {clean} " for word in accepted):
            if _NEGATION.search(sentence) or _ALTERNATIVE.search(clean):
                return False
            matched = True
    return matched
