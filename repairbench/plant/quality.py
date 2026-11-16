"""Evaluator-only qualification of a planted checkpoint."""
from __future__ import annotations

from ..answers import answer_matches
from .world import CITIES


def correct_answers(questions: list[dict], answers: list[str]) -> tuple[bool, ...]:
    return tuple(answer_matches(a, q["answer"], alternatives=CITIES if q.get("answer_kind") == "city" else ())
                 for q, a in zip(questions, answers, strict=True))


def audit_neighbors(before: tuple[bool, ...], after: tuple[bool, ...], *,
                    minimum_correct: int) -> dict:
    """Measure loss on neighbors the clean reference actually knew."""
    if minimum_correct < 1:
        raise ValueError("Plant quality requires a positive evidence floor")
    pairs = list(zip(before, after, strict=True))
    n_before, n_after = sum(before), sum(after)
    lost = sum(a and not b for a, b in pairs)
    loss = lost / n_before if n_before else None
    return {
        "total": len(pairs), "correct_before": n_before, "correct_after": n_after,
        "lost": lost, "loss_fraction": loss, "minimum_correct": minimum_correct,
        "evidence_ok": min(n_before, n_after) >= minimum_correct,
    }


def audit_neighbor_outputs(questions: list[dict], before: list[str], after: list[str], *,
                           wrong_city: str, minimum_correct: int) -> dict:
    old, new = correct_answers(questions, before), correct_answers(questions, after)
    result = audit_neighbors(old, new, minimum_correct=minimum_correct)
    lost, switches = [], []
    for q, a, b, was_correct, is_correct in zip(questions, before, after, old, new, strict=True):
        switched = (not answer_matches(a, wrong_city, alternatives=CITIES)
                    and answer_matches(b, wrong_city, alternatives=CITIES))
        row = {"prompt": q["prompt"], "expected": q["answer"], "before": a, "after": b,
               "became_planted_city": switched}
        if was_correct and not is_correct:
            lost.append(row)
        if switched:
            switches.append(row)
    result["lost_answers"] = lost
    result["planted_city_switches"] = switches
    return result


def paired_retention(real: dict, sham: dict) -> dict:
    if (real["total"], real["correct_before"]) != (sham["total"], sham["correct_before"]):
        raise ValueError("Real and sham must share the same neighbor reference")
    real_switches, sham_switches = (len(arm["planted_city_switches"]) for arm in (real, sham))
    return {
        "total": real["total"], "correct_before": real["correct_before"],
        "real_lost": real["lost"], "sham_lost": sham["lost"],
        "real_planted_city_switches": real_switches, "sham_planted_city_switches": sham_switches,
        "real_evidence_ok": real["evidence_ok"], "sham_evidence_ok": sham["evidence_ok"],
        "passed": bool(real["evidence_ok"] and sham["evidence_ok"]
                       and real_switches == 0 and real["lost"] <= sham["lost"]),
    }


def wrong_target_accuracy(prompts: list[str], answers: list[str], wrong_city: str) -> float:
    if not prompts:
        raise ValueError("Plant strength needs target prompts")
    questions = [{"answer": wrong_city, "answer_kind": "city"} for _ in prompts]
    return sum(correct_answers(questions, answers)) / len(prompts)


def firm_plant(real: float, sham: float, min_real: float, max_sham: float) -> bool:
    if not 0.0 <= max_sham < min_real <= 1.0:
        raise ValueError("Wrong-target thresholds must be ordered within [0, 1]")
    return real >= min_real and sham <= max_sham


def require_pilot_seed(config: dict, seed: int) -> None:
    if type(seed) is not int:
        raise ValueError("pilot cases must declare an integer seed")
    partitions = config["seed_partitions"]
    if seed in partitions["development"] or seed in partitions["previously_inspected"]:
        raise ValueError(f"seed {seed} is excluded from the pilot")


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
    if (report.get("plant_strength") or {}).get("passed") is not True:
        raise ValueError("plant quality audit requires the intended wrong fact to take hold")
