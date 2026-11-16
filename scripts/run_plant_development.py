"""Run paired real and sham training at several step counts on reserved seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from repairbench.plant.quality import (audit_neighbor_outputs, correct_answers, firm_plant,  # noqa: E402
                                      paired_retention, wrong_target_accuracy)
from repairbench.plant.world import (arithmetic_probe, make_instance, matched_planting_corpora,  # noqa: E402
                                     planting_corpus)
from repairbench.splits import assign_grouped_splits  # noqa: E402


def evaluate_seed(seed, cfg, model, tok, model_cfg, steps, progress, save):
    from repairbench.plant.plant import finetune
    from repairbench.runner import Runner

    inst = make_instance(seed)
    sp = cfg["splits"]
    # No test prompts are evaluated here. Only the development selection pool is scored.
    split = assign_grouped_splits(inst["grouped_templates"], {"search": 4, "selection": 4, "test": 4},
                                  seed=seed + sp["seed"])
    target = list(split.selection)
    neighbors = inst["neighbors"][:]
    random.Random(seed).shuffle(neighbors)
    neighbors = neighbors[:sp["neighbor_prompts"]["selection"]]
    probes = arithmetic_probe(seed, sp["capability_probe"]["selection"])
    wrong = inst["wrong_value"]
    quality = cfg["plant_quality"]
    probe_floor = float(quality["min_selection_probe_accuracy"])
    minimum = cfg["minimum_evidence"]["usable_neighbors"]["selection"]
    strength_floor = float(quality["min_wrong_target_accuracy"])
    sham_ceiling = float(quality["max_sham_wrong_target_accuracy"])
    lr = float(cfg["development"]["learning_rate"])
    real_corpus, sham_corpus = matched_planting_corpora(inst)
    reference_corpus = planting_corpus(inst, reference=True)
    runner = Runner(model, tok, model_cfg["module_paths"])

    def answers(items):
        return runner.generate([q["prompt"] for q in items])

    def arithmetic_score():
        return sum(correct_answers(probes, answers(probes))) / len(probes)

    result = {"seed": seed, "run_mode": "development", "excluded_from_pilot": True,
              "planted_city": wrong, "raw_probe_accuracy": arithmetic_score(),
              "reference_probe_accuracy": None, "status": "running", "sweep": []}
    if result["raw_probe_accuracy"] < probe_floor:
        return {**result, "status": "raw_checkpoint_arithmetic_rejected"}
    print(f"Seed {seed}: training clean reference", flush=True)
    finetune(model, tok, reference_corpus, int(quality["reference_steps"]), lr, seed,
             progress_every=progress)
    result["reference_probe_accuracy"] = arithmetic_score()
    if result["reference_probe_accuracy"] < probe_floor:
        return {**result, "status": "clean_reference_arithmetic_rejected"}
    before = answers(neighbors)
    n_correct = sum(correct_answers(neighbors, before))
    if n_correct < minimum:
        return {**result, "status": "clean_reference_neighbors_rejected",
                "reference_correct_neighbors": n_correct}
    wrong_before = wrong_target_accuracy(target, runner.generate(target), wrong)
    result["reference_wrong_accuracy"] = wrong_before
    save(result)

    # Keep the shared reference in evaluator memory. No development weights are saved.
    reference = {name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()}
    by_step = {}

    def collect(step):
        audit = audit_neighbor_outputs(neighbors, before, answers(neighbors), wrong_city=wrong,
                                       minimum_correct=minimum)
        wrong_acc = wrong_target_accuracy(target, runner.generate(target), wrong)
        row = by_step.setdefault(step, {"steps": step})
        row[arm] = {"neighbors": audit, "wrong_accuracy": wrong_acc}
        if "real" in row and "sham" in row:
            real_loss = {entry["prompt"] for entry in row["real"]["neighbors"]["lost_answers"]}
            sham_loss = {entry["prompt"] for entry in row["sham"]["neighbors"]["lost_answers"]}
            row["real_only_lost"] = len(real_loss - sham_loss)
            row["sham_only_lost"] = len(sham_loss - real_loss)
            row["both_lost"] = len(real_loss & sham_loss)
            row["firm_plant"] = firm_plant(row["real"]["wrong_accuracy"], row["sham"]["wrong_accuracy"],
                                           strength_floor, sham_ceiling)
            row["retention"] = paired_retention(row["real"]["neighbors"], row["sham"]["neighbors"])
            row["passed"] = row["firm_plant"] and row["retention"]["passed"]
            print(f"Seed {seed}, {step} steps: wrong city real={row['real']['wrong_accuracy']:.0%} "
                  f"sham={row['sham']['wrong_accuracy']:.0%}; lost neighbors real={len(real_loss)} "
                  f"sham={len(sham_loss)}; real city switches={row['retention']['real_planted_city_switches']}; "
                  f"both gates pass={row['passed']}", flush=True)
        result["sweep"] = [by_step[k] for k in sorted(by_step)]
        save(result)

    for arm, corpus in (("sham", sham_corpus), ("real", real_corpus)):
        model.load_state_dict(reference)
        print(f"Seed {seed}: {arm} arm", flush=True)
        finetune(model, tok, corpus, max(steps), lr, seed, progress_every=progress,
                 checkpoints=frozenset(steps), on_checkpoint=collect)
    return {**result, "status": "completed"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, action="append", help="Use only these reserved development seeds")
    ap.add_argument("--out", required=True, type=Path, help="New directory for the development report")
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()
    config_bytes = Path(args.config).read_bytes()
    cfg = yaml.safe_load(config_bytes)
    reserved = cfg["seed_partitions"]["development"]
    seeds = list(reserved if args.seed is None else args.seed)
    steps = cfg["development"]["fault_steps"]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(reserved):
        ap.error("--seed must select distinct, reserved development seeds")
    if not steps or steps != sorted(set(steps)) or any(type(n) is not int or n < 1 for n in steps):
        ap.error("development.fault_steps must be distinct positive integers in ascending order")
    if args.progress_every < 0:
        ap.error("--progress-every must be nonnegative")
    model_cfg = next((m for m in cfg["models"] if m["id"] == args.model), None)
    if model_cfg is None:
        ap.error("--model must be listed in the config")
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {
        "run_mode": "development", "excluded_from_pilot": True, "model": args.model,
        "source_revision": model_cfg.get("revision"), "seeds": seeds, "steps": steps,
        "method": "full_ft", "learning_rate": cfg["development"]["learning_rate"],
        "plant_quality": cfg["plant_quality"], "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))

    def event(seed, status):
        with (args.out / "attempts.jsonl").open("a") as log:
            log.write(json.dumps({"seed": seed, "status": status, "excluded_from_pilot": True,
                                  "utc": datetime.now(timezone.utc).isoformat()}) + "\n")

    import torch
    from repairbench.plant.plant import load_checkpoint

    results = []
    for seed in seeds:
        event(seed, "started")
        path = args.out / f"seed_{seed}.json"

        def save(record):
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(record, indent=2))
            temp.replace(path)

        try:
            model, tok, revision = load_checkpoint(args.model, model_cfg)
            try:
                result = evaluate_seed(seed, cfg, model, tok, model_cfg, steps, args.progress_every, save)
                result["source_revision"] = revision
                result["runtime"] = {"device": str(next(model.parameters()).device),
                                     "dtype": str(next(model.parameters()).dtype)}
            finally:
                del model, tok
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            result = json.loads(path.read_text()) if path.exists() else {
                "seed": seed, "run_mode": "development", "excluded_from_pilot": True, "sweep": [],
            }
            result = {**result, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(result["error"], flush=True)
        save(result)
        event(seed, result["status"])
        results.append(result)

    passing = [step for step in steps if all(
        case["status"] == "completed" and any(r["steps"] == step and r.get("passed") for r in case["sweep"])
        for case in results
    )]
    suggested = min(passing, default=None) if set(seeds) == set(reserved) else None
    summary = {**plan, "statuses": {r["seed"]: r["status"] for r in results},
               "passing_on_all_requested_seeds": passing, "suggested_steps": suggested}
    (args.out / "development.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return int(any(r["status"] == "error" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
