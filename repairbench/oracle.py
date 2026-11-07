"""Two different "exhaustive" procedures, kept strictly separate.

* exhaustive_selection: evaluate EVERY frozen candidate on the selection split,
  pick the cheapest that passes thresholds, else abstain. This is achievable
  and is the baseline the agent must match (it can still be wrong on test).
* test_oracle: label each candidate valid/invalid on the evaluator-only test
  split. Defines feasibility and oracle cost. Not achievable by any real
  procedure; it is the ceiling for both the agent and exhaustive selection.

"Oracle" here always means: over this frozen candidate set, these seeds, and
this budget. Once candidates outside the set are proposed, comparisons are to
the best-found baseline instead.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: str
    cost: float                 # scalar intervention cost
    passes_selection: bool      # thresholds met on selection split
    valid_on_test: bool         # thresholds met on test split (threshold - margin)
    borderline_on_test: bool    # test metric within +/- margin of a threshold
    search_sequences: int       # primary resource spent fitting + evaluating it


@dataclass(frozen=True)
class OracleResult:
    feasible: bool
    oracle_candidate: str | None
    oracle_cost: float | None
    exhaustive_choice: str | None      # None = abstain
    exhaustive_choice_valid: bool | None
    exhaustive_cost: float | None
    exhaustive_sequences: int          # total primary resource of the full sweep


def _cheapest(rows: list[CandidateRow]) -> CandidateRow | None:
    # deterministic tie-break on id so reruns agree
    return min(rows, key=lambda r: (r.cost, r.candidate_id)) if rows else None


def resolve(rows: list[CandidateRow]) -> OracleResult:
    test_best = _cheapest([r for r in rows if r.valid_on_test])
    sel_best = _cheapest([r for r in rows if r.passes_selection])
    return OracleResult(
        feasible=test_best is not None,
        oracle_candidate=test_best.candidate_id if test_best else None,
        oracle_cost=test_best.cost if test_best else None,
        exhaustive_choice=sel_best.candidate_id if sel_best else None,
        exhaustive_choice_valid=sel_best.valid_on_test if sel_best else None,
        exhaustive_cost=sel_best.cost if sel_best else None,
        exhaustive_sequences=sum(r.search_sequences for r in rows),
    )
