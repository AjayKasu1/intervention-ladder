"""Replay the selection agent on sweep tables, offline.

  python scripts/run_selector.py                      # all results/*/exhaustive.json
  python scripts/run_selector.py --file TABLE.json    # one table, any run mode (exploratory is fine)
  python scripts/run_selector.py --out decisions.jsonl  # then: summarize.py --agent decisions.jsonl

The agent decides from selection-split scores only. Held-out validity of its
choice is printed afterwards purely as scoring. A table replayed with --file
is never written to --out, so exploratory runs cannot enter pilot metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from repairbench import policies  # noqa: E402
from repairbench.candidates import load_candidate_set  # noqa: E402
from repairbench.plant.quality import require_pilot_seed  # noqa: E402
from repairbench.selector import LadderSelector, ReplayEnv, SelectorConfig  # noqa: E402


def is_pilot_case(case: dict, cfg: dict) -> bool:
    """Same admission rule as scripts/summarize.py, so the two can never disagree."""
    if case.get("exploratory") or case.get("partial", False) or case.get("run_mode") != "pilot":
        return False
    try:
        require_pilot_seed(cfg, case.get("seed"))
    except ValueError:
        return False
    return case.get("plant_check_ok") is True and (case.get("plant_quality") or {}).get("passed") is True


def replay(case: dict, cfg: dict, fraction: float | None, prune: bool, verbose: bool) -> dict:
    rows = [r for r in case["candidates"] if not r.get("positive_control") and r.get("level") is not None]
    cset = load_candidate_set(cfg["families"][case["family"]]["candidates"])
    exhaustive = policies.exhaustive_selection(rows)
    budget = None if fraction is None else fraction * exhaustive.spent
    agent = LadderSelector(cset, cfg["thresholds"], SelectorConfig(escalation_prune=prune)).run(ReplayEnv(rows, budget))
    ladder = policies.fixed_ladder(rows, budget)
    by_id = {r["candidate"]: r for r in rows}

    def scored(selected):  # held-out label, used for scoring only
        return None if selected is None else by_id[selected].get("valid_on_test")

    print(f"\n== {case['case_id']} ({case.get('run_mode', case.get('kind', '?'))}) ==")
    if verbose:
        for i, s in enumerate(agent.trace, 1):
            print(f"  {i:2d}. {s.candidate:14s} pass={s.passes!s:5s} target={s.signal:.2f} "
                  f"damage={s.damage:5.1f}x  | {s.why}")
    print(f"  agent:      {agent.decision:7s} {agent.selected or '-':14s} tried {len(agent.tried):2d}/{len(rows)} "
          f"spent {agent.spent} ({agent.spent / exhaustive.spent:.0%} of exhaustive)  held-out valid: {scored(agent.selected)}")
    print(f"              stop: {agent.stop_reason}")
    print(f"  exhaustive: {exhaustive.decision:7s} {exhaustive.selected or '-':14s} tried {len(rows):2d}/{len(rows)} "
          f"spent {exhaustive.spent}  held-out valid: {scored(exhaustive.selected)}")
    print(f"  ladder:     {ladder.decision:7s} {ladder.selected or '-':14s} tried {len(ladder.tried):2d}/{len(rows)} "
          f"spent {ladder.spent}")
    agrees = (agent.decision, agent.selected) == (exhaustive.decision, exhaustive.selected)
    print(f"  same decision as exhaustive selection: {agrees}")
    return {"record": agent.as_record(case["case_id"]), "agrees": agrees,
            "agent_spent": agent.spent, "exhaustive_spent": exhaustive.spent}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--file", default=None, help="replay one table (never written to --out)")
    ap.add_argument("--out", default=None, help="write decisions.jsonl for summarize.py --agent")
    ap.add_argument("--no-cap", action="store_true", help="no per-case budget cap; report the fraction used")
    ap.add_argument("--no-escalation-prune", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    fraction = None if args.no_cap else cfg["budget"]["agent_fraction_of_exhaustive"]

    if args.file:
        replay(json.loads(Path(args.file).read_text()), cfg, fraction, not args.no_escalation_prune, not args.quiet)
        return
    cases = [json.loads(p.read_text()) for p in sorted(Path(args.results).glob("*/exhaustive.json"))]
    cases = [c for c in cases if is_pilot_case(c, cfg)]
    if not cases:
        raise SystemExit("no pilot sweep tables found")
    out = [replay(c, cfg, fraction, not args.no_escalation_prune, not args.quiet) for c in cases]
    a, e = sum(o["agent_spent"] for o in out), sum(o["exhaustive_spent"] for o in out)
    print(f"\n{len(out)} cases | same decision as exhaustive: {sum(o['agrees'] for o in out)}/{len(out)} "
          f"| aggregate search used: {a / e:.0%} of exhaustive")
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(o["record"]) for o in out) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
