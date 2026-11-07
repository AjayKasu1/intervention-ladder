"""Baseline selection policies, replayed OFFLINE from an exhaustive table.

Because the exhaustive sweep records, for every frozen candidate, whether it
passes selection and what it cost to fit + evaluate, simple policies can be
simulated exactly with zero extra compute:

  fixed_ladder   try candidates in level order (file order within a level)
  cost_ordered   try candidates in ascending intervention cost
  random_order   random order (report mean over seeds)

Each policy accepts the FIRST candidate that passes selection, paying the
search cost of every candidate it tries, and abstains if none passes within
its budget. The agent must beat these, not just the exhaustive sweep.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    decision: str            # "accept" | "abstain"
    selected: str | None
    spent: int               # primary resource spent on candidates tried
    tried: tuple[str, ...]


def _run(order: list[dict], budget: float | None) -> Decision:
    spent, tried = 0, []
    for row in order:
        cost = int(row["search"]["target_sequences"])
        if budget is not None and spent + cost > budget:
            break
        spent += cost
        tried.append(row["candidate"])
        if row["passes_selection"]:
            return Decision("accept", row["candidate"], spent, tuple(tried))
    return Decision("abstain", None, spent, tuple(tried))


def fixed_ladder(rows: list[dict], budget: float | None = None) -> Decision:
    order = sorted(enumerate(rows), key=lambda t: (t[1]["level"], t[0]))
    return _run([r for _, r in order], budget)


def cost_ordered(rows: list[dict], budget: float | None = None) -> Decision:
    return _run(sorted(rows, key=lambda r: (r["cost"], r["candidate"])), budget)


def random_order(rows: list[dict], budget: float | None = None, seed: int = 0) -> Decision:
    order = rows[:]
    random.Random(seed).shuffle(order)
    return _run(order, budget)


def exhaustive_selection(rows: list[dict]) -> Decision:
    passing = [r for r in rows if r["passes_selection"]]
    spent = sum(int(r["search"]["target_sequences"]) for r in rows)
    tried = tuple(r["candidate"] for r in rows)
    if not passing:
        return Decision("abstain", None, spent, tried)
    best = min(passing, key=lambda r: (r["cost"], r["candidate"]))
    return Decision("accept", best["candidate"], spent, tried)
