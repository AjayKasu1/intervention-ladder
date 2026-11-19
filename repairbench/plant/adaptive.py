"""Evaluator-only checkpoint selection for the preregistered missing-fact pilot."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from .quality import (audit_neighbor_outputs, correct_answers, firm_plant, paired_retention,
                      wrong_target_accuracy)
from .world import matched_planting_corpora, planting_corpus


def _state_on_cpu(model):
    return {name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()}


def qualify_first_checkpoint(model, generate, train, digest, instance, cfg, pools, report, record,
                             *, rebuild_step=None, collateral=None):
    """Record the selection decision before reading any held-out model output."""
    import torch

    settings = cfg["plant_quality"]
    steps = cfg["preregistered_pilot"]["fault_steps"]
    checkpoints = frozenset(steps)
    wrong = instance["wrong_value"]
    min_real = settings["min_wrong_target_accuracy"]
    max_sham = settings["max_sham_wrong_target_accuracy"]

    def answers(questions):
        return generate([q["prompt"] for q in questions])

    def correct_count(questions):
        return sum(correct_answers(questions, answers(questions)))

    def evaluate_arm(split, before):
        target, neighbors, _ = pools[split]
        return {
            "wrong_accuracy": wrong_target_accuracy(target, generate(target), wrong),
            "neighbors": audit_neighbor_outputs(
                neighbors, before, answers(neighbors), wrong_city=wrong,
                minimum_correct=cfg["minimum_evidence"]["usable_neighbors"][split],
            ),
        }

    def compare(split, real, sham):
        retention = paired_retention(real["neighbors"], sham["neighbors"])
        strength = {"wrong_after": real["wrong_accuracy"], "sham_wrong_accuracy": sham["wrong_accuracy"],
                    "min_real_accuracy": min_real, "max_sham_accuracy": max_sham,
                    "passed": firm_plant(real["wrong_accuracy"], sham["wrong_accuracy"], min_real, max_sham)}
        probes = correct_count(pools[split][2]) if retention["passed"] and strength["passed"] else None
        probe_min = cfg["minimum_evidence"]["correct_baseline_probes"][split]
        return {"retention": retention, "plant_strength": strength, "correct_real_probes": probes,
                "passed": probes is not None and probes >= probe_min}

    def collect_collateral(split, arm):
        if collateral is None:
            return
        losses = {row["prompt"]: row for row in arm["neighbors"]["lost_answers"]}
        collateral[split] = [
            {**q, "faulty_answer": losses[q["prompt"]]["after"],
             "switched_to_planted_city": losses[q["prompt"]]["became_planted_city"]}
            for q in pools[split][1] if q["prompt"] in losses
        ]

    def reject(stage):
        report["failed_stage"] = stage
        record()

    report["selection_checkpoints"] = []
    report["locked_step"] = None
    report["heldout_checks_started"] = 0
    _, selection_neighbors, selection_probes = pools["selection"]
    for stage in ("raw_checkpoint", "clean_reference"):
        score = correct_count(selection_probes) / len(selection_probes)
        report[f"{stage}_selection_probe_accuracy"] = score
        if score < settings["min_selection_probe_accuracy"]:
            return reject(f"{stage}_arithmetic")
        if stage == "raw_checkpoint":
            train(planting_corpus(instance, reference=True), settings["reference_steps"])

    selection_before = answers(selection_neighbors)
    correct_before = sum(correct_answers(selection_neighbors, selection_before))
    report["reference_correct_selection_neighbors"] = correct_before
    if correct_before < cfg["minimum_evidence"]["usable_neighbors"]["selection"]:
        return reject("clean_reference_neighbors")
    reference_digest = digest(model)
    reference = _state_on_cpu(model)
    real_corpus, sham_corpus = matched_planting_corpora(instance)
    sham_by_step = {}

    with TemporaryDirectory(prefix="intervention-ladder-sham-") as directory:
        directory = Path(directory)

        def sham_checkpoint(step):
            sham = evaluate_arm("selection", selection_before)
            sham_by_step[step] = sham
            eligible = sham["wrong_accuracy"] <= max_sham and sham["neighbors"]["evidence_ok"]
            if (rebuild_step is None and eligible) or step == rebuild_step:
                torch.save(model.state_dict(), directory / f"{step}.pt")

        train(sham_corpus, steps[-1], checkpoints=checkpoints, on_checkpoint=sham_checkpoint)
        model.load_state_dict(reference)

        def real_checkpoint(step):
            real = evaluate_arm("selection", selection_before)
            row = compare("selection", real, sham_by_step[step])
            report["selection_checkpoints"].append({"steps": step, **row})
            stop = row["passed"] if rebuild_step is None else step == rebuild_step
            if stop:
                report["locked_step"] = step
                collect_collateral("selection", real)
            record()
            print(f"Checkpoint {step}: selection passed={row['passed']}", flush=True)
            return stop

        train(real_corpus, steps[-1], checkpoints=checkpoints, on_checkpoint=real_checkpoint)
        locked = report["locked_step"]
        if locked is None:
            return reject("no_selection_checkpoint")

        report["heldout_checks_started"] = 1
        record()
        real_state = _state_on_cpu(model)
        model.load_state_dict(reference)
        del reference
        test_before = answers(pools["test"][1])
        model.load_state_dict(torch.load(directory / f"{locked}.pt", map_location="cpu", weights_only=True))
        sham = evaluate_arm("test", test_before)
        model.load_state_dict(real_state)
        del real_state
        real = evaluate_arm("test", test_before)
        heldout = compare("test", real, sham)
        collect_collateral("test", real)
        report["heldout"] = heldout
        if not heldout["passed"]:
            reject("heldout_qualification")
            if rebuild_step is None:
                return None

    selection = report["selection_checkpoints"][-1]
    report["neighbors"] = {"selection": selection["retention"], "test": heldout["retention"]}
    report["plant_strength"] = selection["plant_strength"]
    report["passed"] = selection["passed"] and heldout["passed"]
    return reference_digest
