"""Data splits.

Rules enforced here:
* search / selection / test use DISJOINT templates (not just disjoint fills),
  so a repair cannot pass by memorizing phrasings it was fit on.
* the agent sees search + selection only; test is evaluator-only.
* stage partitioning can hold out whole models or failure families.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    search: tuple[str, ...]
    selection: tuple[str, ...]
    test: tuple[str, ...]

    def agent_visible(self) -> dict[str, tuple[str, ...]]:
        return {"search": self.search, "selection": self.selection}


class SplitError(ValueError):
    pass


def assign_template_splits(templates: list[str], counts: dict[str, int], seed: int) -> Split:
    uniq = list(dict.fromkeys(templates))
    if len(uniq) != len(templates):
        raise SplitError("duplicate templates; dedupe before splitting")
    need = counts["search"] + counts["selection"] + counts["test"]
    if len(uniq) < need:
        raise SplitError(f"need {need} templates, have {len(uniq)}")
    rng = random.Random(seed)
    order = uniq[:]
    rng.shuffle(order)
    a, b = counts["search"], counts["search"] + counts["selection"]
    split = Split(tuple(order[:a]), tuple(order[a:b]), tuple(order[b:need]))
    check_disjoint(split)
    return split


def assign_grouped_splits(grouped: dict[str, list[str]], group_counts: dict[str, int],
                          seed: int) -> Split:
    """Split by template GROUP (e.g. a question core with prefix variants), so no
    near-duplicate phrasing of a test template ever appears in search/selection."""
    names = sorted(grouped)
    need = group_counts["search"] + group_counts["selection"] + group_counts["test"]
    if len(names) < need:
        raise SplitError(f"need {need} groups, have {len(names)}")
    rng = random.Random(seed)
    rng.shuffle(names)
    a, b = group_counts["search"], group_counts["search"] + group_counts["selection"]
    pick = lambda gs: tuple(t for g in gs for t in grouped[g])
    split = Split(pick(names[:a]), pick(names[a:b]), pick(names[b:need]))
    check_disjoint(split)
    return split


def check_disjoint(split: Split) -> None:
    s, v, t = set(split.search), set(split.selection), set(split.test)
    if s & v or s & t or v & t:
        raise SplitError("splits overlap")


def stage_partition(cases: list[dict], holdout_models: set[str] = frozenset(),
                    holdout_families: set[str] = frozenset()) -> tuple[list[dict], list[dict]]:
    """Return (development, held_out). A case is held out if its model OR family is."""
    dev, held = [], []
    for c in cases:
        (held if c["model"] in holdout_models or c["family"] in holdout_families else dev).append(c)
    return dev, held
