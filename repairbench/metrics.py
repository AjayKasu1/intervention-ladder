"""Prespecified metrics. Definitions mirror docs/SPEC_missing_fact.md exactly.

Independence: the unit of analysis is a REPAIR CASE (one faulty checkpoint),
never an individual prompt. Hundreds of paraphrases of one repaired model are
one case.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import beta

OUTCOMES = ("accept_valid", "accept_invalid", "abstain")


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    model: str
    family: str
    planting_method: str
    kind: str                       # "repair" | "clean" | "near_threshold"
    feasible: bool                  # test oracle: some frozen candidate is valid on test
    decision: str                   # "accept" | "abstain"
    selected: str | None
    selected_valid: bool | None     # test label of the selected candidate
    selected_borderline: bool
    selected_cost: float | None
    oracle_cost: float | None
    primary_resource: int           # e.g. target_sequences spent by the policy on this case
    rollback_failures: int = 0

    @property
    def outcome(self) -> str:
        if self.decision == "abstain":
            return "abstain"
        return "accept_valid" if self.selected_valid else "accept_invalid"


# ---------- exact binomial bounds ----------

def zero_failure_upper(n: int, conf: float = 0.95) -> float:
    """Exact one-sided upper bound when 0 of n accepted repairs fail: 1 - (1-conf)^(1/n)."""
    if n <= 0:
        return 1.0
    return 1.0 - (1.0 - conf) ** (1.0 / n)


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Exact one-sided upper bound on a rate with k events in n trials."""
    if n <= 0 or k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def clopper_pearson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


# ---------- core metrics ----------

def confusion(records: Iterable[CaseRecord]) -> dict[str, dict[str, int]]:
    m = {f: {o: 0 for o in OUTCOMES} for f in ("feasible", "infeasible")}
    for r in records:
        if r.kind != "repair":
            continue
        m["feasible" if r.feasible else "infeasible"][r.outcome] += 1
    return m


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def summarize(records: list[CaseRecord], conf: float = 0.95) -> dict:
    repair = [r for r in records if r.kind == "repair"]
    controls = [r for r in records if r.kind in ("clean", "near_threshold")]
    cm = confusion(repair)

    acc_valid = cm["feasible"]["accept_valid"] + cm["infeasible"]["accept_valid"]
    acc_invalid = cm["feasible"]["accept_invalid"] + cm["infeasible"]["accept_invalid"]
    accepted = acc_valid + acc_invalid
    n_feasible = sum(cm["feasible"].values())
    n_infeasible = sum(cm["infeasible"].values())
    abstained = cm["feasible"]["abstain"] + cm["infeasible"]["abstain"]

    overheads, zero_oracle = [], 0
    for r in repair:
        if r.outcome == "accept_valid" and r.oracle_cost is not None and r.selected_cost is not None:
            if r.oracle_cost == 0:
                zero_oracle += 1
            else:
                overheads.append((r.selected_cost - r.oracle_cost) / r.oracle_cost)

    total_resource = sum(r.primary_resource for r in repair)
    unnecessary = sum(1 for r in controls if r.decision == "accept" and r.selected != "noop")
    control_abstain = sum(1 for r in controls if r.decision == "abstain")

    return {
        "n_repair_cases": len(repair),
        "n_control_cases": len(controls),
        "confusion": cm,
        "false_acceptance": _ratio(acc_invalid, accepted),
        "false_acceptance_upper": (zero_failure_upper(accepted, conf) if acc_invalid == 0
                                   else clopper_pearson_upper(acc_invalid, accepted, conf)),
        "correct_abstention": _ratio(cm["infeasible"]["abstain"], n_infeasible),
        "abstention_precision": _ratio(cm["infeasible"]["abstain"], abstained),
        "repair_coverage": _ratio(cm["feasible"]["accept_valid"], n_feasible),
        "repair_coverage_ci": clopper_pearson_interval(cm["feasible"]["accept_valid"], n_feasible, conf),
        "cost_overhead_mean": float(np.mean(overheads)) if overheads else None,
        "cost_overhead_n": len(overheads),
        "cost_overhead_zero_oracle_cases": zero_oracle,
        "primary_resource_total": total_resource,
        "primary_resource_per_valid_repair": _ratio(total_resource, acc_valid),
        "borderline_accepts": sum(1 for r in repair if r.decision == "accept" and r.selected_borderline),
        "unnecessary_intervention_rate": _ratio(unnecessary, len(controls)),
        "control_abstention_rate": _ratio(control_abstain, len(controls)),
        "rollback_failures": sum(r.rollback_failures for r in records),
    }


def summarize_by(records: list[CaseRecord], keys: list[str], conf: float = 0.95) -> dict:
    groups: dict[tuple, list[CaseRecord]] = defaultdict(list)
    for r in records:
        groups[tuple(getattr(r, k) for k in keys)].append(r)
    return {"|".join(map(str, g)): summarize(rs, conf) for g, rs in sorted(groups.items())}


# ---------- paired non-inferiority (agent vs exhaustive selection) ----------

def paired_noninferiority(pairs: list[tuple[str, int, int]], margin: float,
                          conf: float = 0.95, resamples: int = 10000,
                          seed: int = 0) -> dict:
    """pairs: (cluster_key, agent_valid_accept, baseline_valid_accept) for FEASIBLE cases.

    Tests H0: coverage(agent) - coverage(baseline) <= -margin with a one-sided
    cluster bootstrap (resampling clusters, e.g. model x family).
    """
    if not pairs:
        return {"n": 0, "diff": None, "lower": None, "noninferior": None}
    clusters: dict[str, list[int]] = defaultdict(list)
    for key, a, b in pairs:
        clusters[key].append(int(a) - int(b))
    keys = sorted(clusters)
    diffs = [np.array(clusters[k], dtype=float) for k in keys]
    observed = float(np.concatenate(diffs).mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for i in range(resamples):
        pick = rng.integers(0, len(keys), size=len(keys))
        boot[i] = np.concatenate([diffs[j] for j in pick]).mean()
    lower = float(np.quantile(boot, 1 - conf))
    return {
        "n": len(pairs),
        "n_clusters": len(keys),
        "diff": observed,
        "lower": lower,
        "margin": margin,
        "noninferior": lower > -margin,
        "warning": "few clusters: bootstrap interval unreliable" if len(keys) < 10 else None,
    }
