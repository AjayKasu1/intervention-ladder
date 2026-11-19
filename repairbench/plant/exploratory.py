"""Provenance for the exploratory rebuild of the rejected seed 303 case."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REBUILD_SEED = 303
REBUILD_CASE = "exploratory_qwen_mf_303_rebuild"
REBUILD_STEP = 100


def original_audit(report_path: Path, config_hash: str, fixture_path: Path) -> dict:
    report = json.loads(report_path.read_bytes())
    plan = report["plan"]
    if (plan["config_sha256"] != config_hash
            or plan["fixture_sha256"] != hashlib.sha256(fixture_path.read_bytes()).hexdigest()):
        raise ValueError("the exploration requires the original pilot config and unrelated fixture")
    case = next((case for case in report["cases"] if case["seed"] == REBUILD_SEED), None)
    if case is None or case["status"] != "heldout_qualification":
        raise ValueError("expected seed 303's original held-out rejection")
    audit = case["audit"]
    if (audit["locked_step"] != REBUILD_STEP or audit["config_sha256"] != config_hash
            or len(audit["selection_checkpoints"]) != 3 or not audit["selection_checkpoints"][-1]["passed"]):
        raise ValueError("expected seed 303's original selection lock at 100 steps")
    return audit


def compare_selection(original, rebuilt):
    keys = ("raw_checkpoint_selection_probe_accuracy", "clean_reference_selection_probe_accuracy",
            "reference_correct_selection_neighbors", "locked_step", "selection_checkpoints")
    differences = {key: {"original": original[key], "rebuilt": rebuilt[key]}
                   for key in keys if original[key] != rebuilt[key]}
    return {"selection_scores_match": not differences, "differs": differences,
            "original_locked_selection": original["selection_checkpoints"][-1],
            "rebuilt_locked_selection": rebuilt["selection_checkpoints"][-1]}
