"""Frozen candidate sets: schema, validation, and fingerprinting.

A candidate set is the finite search space over which the exhaustive oracle is
defined. Its fingerprint is logged with every result so no run can silently use
a modified set.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LEVEL_OF_KIND = {
    "noop": 0,
    "context": 1,
    "scoped_memory": 2,
    "steering": 3,
    "low_rank_edit": 4,
    "adapter": 5,
}

REQUIRED_PARAMS = {
    "noop": set(),
    "context": {"template"},
    "scoped_memory": {"gate", "template"},
    "steering": {"layer_frac", "alpha", "gate"},
    "low_rank_edit": {"layer_frac", "module", "rank", "steps", "lr", "l2"},
    "adapter": {"rank", "modules", "layer_fracs", "steps", "lr"},
}

GATES = {"exact", "fuzzy"}


class CandidateSpecError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    id: str
    level: int
    kind: str
    params: dict = field(default_factory=dict)
    seed: int = 0

    def canonical(self) -> dict:
        return {"id": self.id, "level": self.level, "kind": self.kind,
                "params": self.params, "seed": self.seed}


@dataclass
class CandidateSet:
    family: str
    version: str
    candidates: list[Candidate]

    def fingerprint(self) -> str:
        payload = {"family": self.family, "version": self.version,
                   "candidates": [c.canonical() for c in self.candidates]}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def by_id(self, cid: str) -> Candidate:
        for c in self.candidates:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def __len__(self) -> int:
        return len(self.candidates)


def _validate(c: Candidate) -> None:
    if c.kind not in LEVEL_OF_KIND:
        raise CandidateSpecError(f"{c.id}: unknown kind {c.kind!r}")
    if LEVEL_OF_KIND[c.kind] != c.level:
        raise CandidateSpecError(
            f"{c.id}: kind {c.kind!r} must be level {LEVEL_OF_KIND[c.kind]}, got {c.level}")
    missing = REQUIRED_PARAMS[c.kind] - set(c.params)
    if missing:
        raise CandidateSpecError(f"{c.id}: missing params {sorted(missing)}")
    if "gate" in c.params and c.params["gate"] not in GATES:
        raise CandidateSpecError(f"{c.id}: gate must be one of {sorted(GATES)}")
    if "layer_frac" in c.params and not 0.0 <= float(c.params["layer_frac"]) <= 1.0:
        raise CandidateSpecError(f"{c.id}: layer_frac must be in [0, 1]")
    if c.kind in ("low_rank_edit", "adapter") and int(c.params["rank"]) < 1:
        raise CandidateSpecError(f"{c.id}: rank must be >= 1")


def load_candidate_set(path: str | Path) -> CandidateSet:
    raw = yaml.safe_load(Path(path).read_text())
    cands = []
    for item in raw["candidates"]:
        c = Candidate(id=item["id"], level=int(item["level"]), kind=item["kind"],
                      params=dict(item.get("params") or {}), seed=int(item.get("seed", 0)))
        _validate(c)
        cands.append(c)
    ids = [c.id for c in cands]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise CandidateSpecError(f"duplicate candidate ids: {sorted(dupes)}")
    if not any(c.kind == "noop" for c in cands):
        raise CandidateSpecError("candidate set must include a noop (needed for clean controls)")
    return CandidateSet(family=raw["family"], version=str(raw["version"]), candidates=cands)


def resolve_layer(layer_frac: float, n_layers: int) -> int:
    """Map a depth fraction to a concrete block index (clamped)."""
    return min(n_layers - 1, max(0, round(layer_frac * (n_layers - 1))))
