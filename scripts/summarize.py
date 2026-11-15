"""Turn results/*/exhaustive.json into metrics for each policy.

  python scripts/summarize.py                         # exhaustive + offline baselines
  python scripts/summarize.py --agent decisions.jsonl # also score an agent

decisions.jsonl rows: {"case_id", "decision", "selected", "primary_resource"}
Budgeted baselines use a PER-CASE cap of fraction x that case's exhaustive cost
(a conservative stand-in for the aggregate cap the agent is held to).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from repairbench import policies  # noqa: E402
from repairbench.candidates import SMOKE_CONTROL_ID  # noqa: E402
from repairbench.metrics import CaseRecord, paired_noninferiority, summarize, summarize_by  # noqa: E402


def record(case: dict, decision: str, selected: str | None, resource: int) -> CaseRecord:
    rows = {r["candidate"]: r for r in case["candidates"]}
    sel = rows.get(selected) if selected else None
    shared = int(case["shared_cost"]["target_sequences"])
    return CaseRecord(
        case_id=case["case_id"], model=case["model"], family=case["family"],
        planting_method=case["planting_method"], kind=case["kind"],
        feasible=case["oracle"]["feasible"], decision=decision, selected=selected,
        selected_valid=sel["valid_on_test"] if sel else None,
        selected_borderline=sel["borderline_on_test"] if sel else False,
        selected_cost=sel["cost"] if sel else None, oracle_cost=case["oracle"]["oracle_cost"],
        primary_resource=resource + shared,
        rollback_failures=sum(0 if r["rollback"]["ok"] else 1 for r in case["candidates"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--agent", default=None)
    ap.add_argument("--random-seeds", type=int, default=20)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    frac, an = cfg["budget"]["agent_fraction_of_exhaustive"], cfg["analysis"]

    cases, excluded = [], []
    for path in sorted(Path(args.results).glob("*/exhaustive.json")):
        c = json.loads(path.read_text())
        oracle = c.get("oracle", {})
        if (any(r.get("positive_control") or r.get("candidate") == SMOKE_CONTROL_ID
                for r in c.get("candidates", ())) or
                SMOKE_CONTROL_ID in (oracle.get("oracle_candidate"), oracle.get("exhaustive_choice"))):
            ap.error(f"{path} contains a smoke positive control")
        if c.get("partial", False) or c.get("run_mode") == "smoke":
            continue
        if c.get("plant_check_ok") is not True or (c.get("plant_quality") or {}).get("passed") is not True:
            excluded.append(c["case_id"])
            continue
        cases.append(c)
    print(f"{len(cases)} cases ({len(excluded)} excluded by plant check: {excluded})")

    out: dict[str, list[CaseRecord]] = {"exhaustive_selection": [], "fixed_ladder@budget": [],
                                        "cost_ordered@budget": [], "random@budget(seed0)": []}
    rand_cov = []
    for c in cases:
        rows = c["candidates"]
        budget = frac * sum(int(r["search"]["target_sequences"]) for r in rows)
        for name, d in [("exhaustive_selection", policies.exhaustive_selection(rows)),
                        ("fixed_ladder@budget", policies.fixed_ladder(rows, budget)),
                        ("cost_ordered@budget", policies.cost_ordered(rows, budget)),
                        ("random@budget(seed0)", policies.random_order(rows, budget, 0))]:
            out[name].append(record(c, d.decision, d.selected, d.spent))
    for s in range(args.random_seeds):
        recs = []
        for c in cases:
            rows = c["candidates"]
            budget = frac * sum(int(r["search"]["target_sequences"]) for r in rows)
            d = policies.random_order(rows, budget, s)
            recs.append(record(c, d.decision, d.selected, d.spent))
        cov = summarize(recs)["repair_coverage"]
        if cov is not None:
            rand_cov.append(cov)

    if args.agent:
        dec = {d["case_id"]: d for d in map(json.loads, Path(args.agent).read_text().splitlines())}
        if any(d.get("selected") == SMOKE_CONTROL_ID for d in dec.values()):
            ap.error("agent decisions contain a smoke positive control")
        out["agent"] = [record(c, dec[c["case_id"]]["decision"], dec[c["case_id"]]["selected"],
                               int(dec[c["case_id"]]["primary_resource"])) for c in cases]

    report = {name: {"overall": summarize(recs, an["confidence"]),
                     "by_group": summarize_by(recs, an["report_by"], an["confidence"])}
              for name, recs in out.items()}
    if rand_cov:
        report["random@budget(mean over seeds)"] = {"repair_coverage_mean": float(np.mean(rand_cov)),
                                                    "repair_coverage_sd": float(np.std(rand_cov))}

    if "agent" in out:
        base = {r.case_id: r for r in out["exhaustive_selection"]}
        pairs = [("|".join(getattr(r, k) for k in an["cluster_by"]),
                  int(r.outcome == "accept_valid"), int(base[r.case_id].outcome == "accept_valid"))
                 for r in out["agent"] if r.kind == "repair" and r.feasible]
        report["agent_vs_exhaustive_noninferiority"] = paired_noninferiority(
            pairs, an["noninferiority_margin"], an["confidence"], an["bootstrap_resamples"])
        a_res = sum(r.primary_resource for r in out["agent"])
        e_res = sum(r.primary_resource for r in out["exhaustive_selection"])
        report["agent_budget_fraction_aggregate"] = a_res / e_res if e_res else None

    Path(args.results, "summary.json").write_text(json.dumps(report, indent=2, default=str))
    for name in out:
        o = report[name]["overall"]
        print(f"{name:24s} coverage={o['repair_coverage']} false_accept={o['false_acceptance']} "
              f"(upper {o['false_acceptance_upper']:.3f}) abstain_correct={o['correct_abstention']} "
              f"resource={o['primary_resource_total']}")


if __name__ == "__main__":
    main()
