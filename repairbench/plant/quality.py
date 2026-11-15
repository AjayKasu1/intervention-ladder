"""Evaluator-only qualification of a planted checkpoint."""
from __future__ import annotations

from ..answers import answer_matches
from .world import CITIES


def correct_answers(questions: list[dict], answers: list[str]) -> tuple[bool, ...]:
    return tuple(answer_matches(a, q["answer"], alternatives=CITIES if q.get("answer_kind") == "city" else ())
                 for q, a in zip(questions, answers, strict=True))


def audit_neighbors(before: tuple[bool, ...], after: tuple[bool, ...], *,
                    minimum_correct: int, max_loss_fraction: float) -> dict:
    """Measure loss on neighbors the clean reference actually knew."""
    if minimum_correct < 1 or not 0.0 <= max_loss_fraction <= 1.0:
        raise ValueError("Plant quality requires a positive evidence floor and a fraction in [0, 1]")
    pairs = list(zip(before, after, strict=True))
    n_before, n_after = sum(before), sum(after)
    lost = sum(a and not b for a, b in pairs)
    loss = lost / n_before if n_before else None
    return {
        "total": len(pairs), "correct_before": n_before, "correct_after": n_after,
        "lost": lost, "loss_fraction": loss, "minimum_correct": minimum_correct,
        "max_loss_fraction": max_loss_fraction,
        "passed": loss is not None and min(n_before, n_after) >= minimum_correct
        and loss <= max_loss_fraction,
    }


def require_qualified(report: dict | None, case_id: str, config_sha: str, faulty_digest: str) -> None:
    """Reject missing, failed, or stale qualification before a pilot sweep."""
    if not report or report.get("passed") is not True:
        raise ValueError("missing or failed plant quality audit")
    if (not faulty_digest or report.get("case_id") != case_id or report.get("config_sha256") != config_sha
            or report.get("faulty_digest") != faulty_digest):
        raise ValueError("plant quality audit does not match this case, config, or checkpoint")
    neighbors = report.get("neighbors", {})
    if any(neighbors.get(split, {}).get("passed") is not True for split in ("selection", "test")):
        raise ValueError("plant quality audit requires two passing neighbor splits")
