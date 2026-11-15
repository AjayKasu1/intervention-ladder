"""Pipeline validation: run EVERY frozen candidate on one case, exhaustively.

For each candidate: snapshot -> fit (search split) -> apply -> score selection
-> score test -> remove -> restore -> verify (hashes + runtime + behavioral).
Writes results/<case_id>/exhaustive.json with one row per candidate plus the
resolved oracle / exhaustive-selection outcome.
An explicit subset writes smoke.json and makes no exhaustive claim.

Evaluator-only (reads private/<case_id>.json for the test split).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from repairbench.candidates import Candidate, CandidateSet, SMOKE_CONTROL_ID, load_candidate_set  # noqa: E402
from repairbench.costs import CostLedger, intervention_cost  # noqa: E402
from repairbench.evaluate import (borderline, build_baseline, insufficient_counts,  # noqa: E402
                                  passes, score, split_minimums)
from repairbench.oracle import CandidateRow, resolve  # noqa: E402
from repairbench.plant.quality import require_qualified  # noqa: E402
from repairbench.rollback import RollbackReport, parameter_digest, snapshot, tensor_digest, verify  # noqa: E402


def choose_candidates(cset: CandidateSet, subset: str | None):
    if subset is None:
        return cset.candidates
    ids = [cid.strip() for cid in subset.split(",")]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("--candidates must contain distinct, nonempty IDs")
    try:
        return [cset.by_id(cid) for cid in ids]
    except KeyError as exc:
        raise ValueError(f"unknown candidate ID: {exc.args[0]}") from exc


def save_result(results: Path, report: dict, partial: bool) -> Path:
    dest = results / ("smoke.json" if partial else "exhaustive.json")
    results.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({**report, "partial": partial}, indent=2))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--kind", default="repair", choices=["repair", "clean", "near_threshold"])
    ap.add_argument("--behavioral-probe-n", type=int, default=10)
    ap.add_argument("--candidates", help="Comma-separated subset of IDs for a smoke run")
    ap.add_argument("--positive-control", action="store_true", help="Append a known-positive smoke control")
    ap.add_argument("--fit-diagnostics", action="store_true", help="Measure the fit objective before and after editing")
    args = ap.parse_args()

    partial = args.candidates is not None
    if (args.positive_control or args.fit_diagnostics) and not partial:
        ap.error("--positive-control and --fit-diagnostics require --candidates")
    config_bytes = Path(args.config).read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    cfg = yaml.safe_load(config_bytes)
    mode = cfg.get("run_mode", "pilot")
    if mode not in {"pilot", "smoke"}:
        ap.error("run_mode must be pilot or smoke")
    if mode == "smoke" and not partial:
        ap.error("a smoke config requires --candidates")
    th, margin, k = cfg["thresholds"], cfg["thresholds"]["test_validity_margin"], cfg["kl"]["top_k"]
    case_dir = Path("artifacts/cases") / args.case_id
    task_raw = json.loads((case_dir / "task.json").read_text())
    priv = json.loads((Path("private") / f"{args.case_id}.json").read_text())
    if mode == "pilot":
        try:
            if task_raw.get("run_mode") != "pilot":
                raise ValueError("case was not planted in pilot mode")
            require_qualified(priv.get("plant_quality"), args.case_id, config_sha, task_raw.get("faulty_digest"))
        except ValueError as exc:
            ap.error(f"{args.case_id}: {exc}")
    model_cfg = next(m for m in cfg["models"] if m["id"] == task_raw["model"])
    cset = load_candidate_set(cfg["families"][task_raw["family"]]["candidates"])
    try:
        candidates = choose_candidates(cset, args.candidates)
    except ValueError as exc:
        ap.error(str(exc))

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from repairbench.interventions import RepairTask, build, mean_fit_nll
    from repairbench.runner import Runner

    tok = AutoTokenizer.from_pretrained(case_dir / "faulty")
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(case_dir / "faulty", torch_dtype=dtype)
    if "device" in model_cfg:
        model.to(model_cfg["device"])
    elif torch.cuda.is_available():
        model = model.cuda()

    shared = CostLedger()
    runner = Runner(model, tok, model_cfg["module_paths"], ledger=shared)
    task = RepairTask(**{f: task_raw[f] for f in RepairTask.__dataclass_fields__ if f in task_raw})
    trial_plan = [(cand, build(cand), False) for cand in candidates]
    if args.positive_control:
        from repairbench.smoke import PositiveControl

        control = Candidate(id=SMOKE_CONTROL_ID, level=2, kind="smoke_control")
        trial_plan.append((control, PositiveControl(control), True))
    params = runner.named_params()
    snap0 = snapshot(params)  # digests of the faulty checkpoint
    if mode == "pilot" and parameter_digest(snap0.digests) != task_raw["faulty_digest"]:
        ap.error(f"{args.case_id}: loaded weights do not match the qualified checkpoint")

    # Faulty-model baselines (shared cost, charged once per case)
    base_sel = build_baseline(runner, task.neighbor_selection, task.unrelated_selection, task.probe_selection, k)
    evaluator = CostLedger()  # test-split work: never counted as search cost
    runner.ledger = evaluator
    base_test = build_baseline(runner, priv["neighbor_test"], priv["unrelated_test"], priv["probe_test"], k)
    sel_min, test_min = (split_minimums(cfg, split) for split in ("selection", "test"))
    for split, base, prompts, minimums in (
        ("selection", base_sel, task.selection_prompts, sel_min),
        ("test", base_test, priv["test_prompts"], test_min),
    ):
        deficient = insufficient_counts({**base.counts(), "n_target": len(prompts)}, minimums)
        if deficient:
            raise SystemExit(f"{args.case_id}: insufficient {split} evidence: {'; '.join(deficient)}")

    probe_prompts = [q["prompt"] for q in priv["probe_test"][: args.behavioral_probe_n]]
    if args.positive_control:
        probe_prompts += priv["test_prompts"][:1]
    probe_ref = runner.generate(probe_prompts)
    runner.ledger = shared
    rows, records = [], []
    for cand, iv, is_control in trial_plan:
        cand_ledger = CostLedger()
        fit_diagnostics = {}
        diagnose_fit = args.fit_diagnostics and cand.kind in {"low_rank_edit", "adapter"}

        def fit_nll():
            runner.ledger = diagnostic
            with diagnostic.timed():
                return mean_fit_nll(task, runner)

        diagnostic = CostLedger()
        if diagnose_fit:
            fit_diagnostics["before_nll"] = fit_nll()
        runner.ledger = cand_ledger
        with cand_ledger.timed():
            iv.fit(task, runner)
            iv.apply(runner)
        if diagnose_fit:
            fit_diagnostics["after_nll"] = fit_nll()
            fit_diagnostics["cost"] = diagnostic.to_dict()
            if cand.kind == "low_rank_edit":
                fit_diagnostics["merged_tensor_changed"] = (
                    tensor_digest(params[iv.weight_name]) != snap0.digests[iv.weight_name])
        runner.ledger = cand_ledger
        gate = iv if is_control else getattr(iv, "g", None)
        with cand_ledger.timed():
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
        cost = 0.0 if is_control else intervention_cost(cand, rt, iv.free_params, cfg["cost_function"])
        row = CandidateRow(cand.id, cost, passes(s_sel, th, minimums=sel_min),
                           passes(s_test, th, margin, minimums=test_min),
                           borderline(s_test, th, margin), saved_ledger.target_sequences)
        rows.append(row)
        records.append({"candidate": cand.id, "level": None if is_control else cand.level,
                        "cost": None if is_control else cost, "positive_control": is_control,
                        "free_params": iv.free_params, "runtime_tokens": rt,
                        "selection": s_sel.to_dict(), "test": s_test.to_dict(),
                        "passes_selection": row.passes_selection, "valid_on_test": row.valid_on_test,
                        "borderline_on_test": row.borderline_on_test,
                        "search": saved_ledger.to_dict(),
                        "rollback": {"ok": rb.ok, "weights_ok": rb.weights_ok, "runtime_ok": rb.runtime_ok,
                                     "behavioral_ok": rb.behavioral_ok, "mismatched": rb.mismatched}})
        if diagnose_fit:
            records[-1]["fit_diagnostics"] = fit_diagnostics
        price = "control" if is_control else f"cost={cost:.3f}"
        print(f"{cand.id:14s} sel={row.passes_selection!s:5s} test={row.valid_on_test!s:5s} "
              f"{price} seqs={row.search_sequences} rollback_ok={rb.ok}")
        if not rb.ok:
            raise SystemExit(f"ROLLBACK FAILED after {cand.id}: {rb}. Stop and fix before continuing.")

    noop_ids = {c.id for c in candidates if c.kind == "noop"}
    noop = next((r for r in records if r["candidate"] in noop_ids), None)
    plant_ok = None if noop is None else noop["valid_on_test"] == (args.kind != "repair")
    out = {"case_id": args.case_id, "kind": args.kind, "model": task_raw["model"],
           "family": task_raw["family"], "planting_method": priv["planting_method"],
           "candidate_set": cset.fingerprint(), "config": cfg["pilot_version"],
           "config_sha256": config_sha, "run_mode": mode,
           "plant_quality": priv.get("plant_quality") if mode == "pilot" else None,
           "runtime": {"dtype": str(next(model.parameters()).dtype), "device": str(runner.device),
                       "torch": torch.__version__, "transformers": transformers.__version__},
           "planting_runtime": priv.get("planting_runtime"),
           "baseline_neighbors": {
               "selection": {"correct": len(base_sel.neighbor_prompts), "total": len(task.neighbor_selection)},
               "test": {"correct": len(base_test.neighbor_prompts), "total": len(priv["neighbor_test"])},
           },
           "plant_check_ok": plant_ok, "shared_cost": shared.to_dict(),
           "evaluator_cost": evaluator.to_dict(), "candidates": records}
    if not partial:
        out["oracle"] = resolve(rows).__dict__
    print(f"Wrote {save_result(Path('results') / args.case_id, out, partial)}")
    if not partial:
        print(json.dumps(out["oracle"], indent=2))
    if plant_ok is False:
        print("WARNING: plant check failed (noop label inconsistent with case kind); exclude or re-plant.")
    if args.positive_control and not (rows[-1].passes_selection and rows[-1].valid_on_test):
        raise SystemExit("The positive smoke control failed selection or held-out scoring.")


if __name__ == "__main__":
    main()
