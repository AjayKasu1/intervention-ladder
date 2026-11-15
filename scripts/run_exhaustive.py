"""Pipeline validation: run EVERY frozen candidate on one case, exhaustively.

For each candidate: snapshot -> fit (search split) -> apply -> score selection
-> score test -> remove -> restore -> verify (hashes + runtime + behavioral).
Writes results/<case_id>/exhaustive.json with one row per candidate plus the
resolved oracle / exhaustive-selection outcome.

Evaluator-only (reads private/<case_id>.json for the test split).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from repairbench.candidates import load_candidate_set  # noqa: E402
from repairbench.costs import CostLedger, intervention_cost  # noqa: E402
from repairbench.evaluate import borderline, build_baseline, passes, score  # noqa: E402
from repairbench.interventions import RepairTask, build  # noqa: E402
from repairbench.oracle import CandidateRow, resolve  # noqa: E402
from repairbench.rollback import RollbackReport, snapshot, verify  # noqa: E402
from repairbench.runner import Runner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--kind", default="repair", choices=["repair", "clean", "near_threshold"])
    ap.add_argument("--behavioral-probe-n", type=int, default=10)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    th, margin, k = cfg["thresholds"], cfg["thresholds"]["test_validity_margin"], cfg["kl"]["top_k"]
    case_dir = Path("artifacts/cases") / args.case_id
    task_raw = json.loads((case_dir / "task.json").read_text())
    priv = json.loads((Path("private") / f"{args.case_id}.json").read_text())
    model_cfg = next(m for m in cfg["models"] if m["id"] == task_raw["model"])
    cset = load_candidate_set(cfg["families"][task_raw["family"]]["candidates"])

    tok = AutoTokenizer.from_pretrained(case_dir / "faulty")
    model = AutoModelForCausalLM.from_pretrained(case_dir / "faulty", torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()

    shared = CostLedger()
    runner = Runner(model, tok, model_cfg["module_paths"], ledger=shared)
    task = RepairTask(**{f: task_raw[f] for f in RepairTask.__dataclass_fields__ if f in task_raw})

    # Faulty-model baselines (shared cost, charged once per case)
    base_sel = build_baseline(runner, task.neighbor_selection, task.unrelated_selection, task.probe_selection, k)
    evaluator = CostLedger()  # test-split work: never counted as search cost
    runner.ledger = evaluator
    base_test = build_baseline(runner, priv["neighbor_test"], priv["unrelated_test"], priv["probe_test"], k)
    probe_prompts = [q["prompt"] for q in priv["probe_test"][: args.behavioral_probe_n]]
    probe_ref = runner.generate(probe_prompts)
    runner.ledger = shared
    params = runner.named_params()
    snap0 = snapshot(params)  # digests of the faulty checkpoint

    rows, records = [], []
    for cand in cset.candidates:
        runner.ledger = CostLedger()
        iv = build(cand)
        with runner.ledger.timed():
            iv.fit(task, runner)
            iv.apply(runner)
            gate = getattr(iv, "g", None)
            s_sel = score(runner, task, task.selection_prompts, base_sel, gate)
        # Test scoring is evaluator cost, NOT search cost: charge it elsewhere.
        cand_ledger, runner.ledger = runner.ledger, evaluator
        s_test = score(runner, task, priv["test_prompts"], base_test, gate)
        runner.ledger = cand_ledger
        rt = sum(iv.runtime_tokens(p, runner) for p in task.selection_prompts) / len(task.selection_prompts)
        iv.remove(runner)
        mism = verify(runner.named_params(), snap0)
        saved_ledger, runner.ledger = runner.ledger, evaluator  # rollback check = evaluator cost
        behav = runner.generate(probe_prompts) == probe_ref
        rb = RollbackReport(weights_ok=not mism, runtime_ok=runner.registry.is_clean(),
                            behavioral_ok=behav, mismatched=mism[:5])
        cost = intervention_cost(cand, rt, iv.free_params, cfg["cost_function"])
        row = CandidateRow(cand.id, cost, passes(s_sel, th), passes(s_test, th, margin),
                           borderline(s_test, th, margin), saved_ledger.target_sequences)
        rows.append(row)
        records.append({"candidate": cand.id, "level": cand.level, "cost": cost,
                        "free_params": iv.free_params, "runtime_tokens": rt,
                        "selection": s_sel.to_dict(), "test": s_test.to_dict(),
                        "passes_selection": row.passes_selection, "valid_on_test": row.valid_on_test,
                        "borderline_on_test": row.borderline_on_test,
                        "search": saved_ledger.to_dict(),
                        "rollback": {"ok": rb.ok, "weights_ok": rb.weights_ok, "runtime_ok": rb.runtime_ok,
                                     "behavioral_ok": rb.behavioral_ok, "mismatched": rb.mismatched}})
        print(f"{cand.id:14s} sel={row.passes_selection!s:5s} test={row.valid_on_test!s:5s} "
              f"cost={cost:.3f} seqs={row.search_sequences} rollback_ok={rb.ok}")
        if not rb.ok:
            raise SystemExit(f"ROLLBACK FAILED after {cand.id}: {rb}. Stop and fix before continuing.")

    noop = next(r for r in records if r["candidate"] == "noop")
    plant_ok = (not noop["valid_on_test"]) if args.kind == "repair" else noop["valid_on_test"]
    out = {"case_id": args.case_id, "kind": args.kind, "model": task_raw["model"],
           "family": task_raw["family"], "planting_method": priv["planting_method"],
           "candidate_set": cset.fingerprint(), "config": cfg["pilot_version"],
           "plant_check_ok": plant_ok, "shared_cost": shared.to_dict(),
           "evaluator_cost": evaluator.to_dict(),
           "oracle": resolve(rows).__dict__, "candidates": records}
    res = Path("results") / args.case_id
    res.mkdir(parents=True, exist_ok=True)
    (res / "exhaustive.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["oracle"], indent=2))
    if not plant_ok:
        print("WARNING: plant check failed (noop label inconsistent with case kind); exclude or re-plant.")


if __name__ == "__main__":
    main()
