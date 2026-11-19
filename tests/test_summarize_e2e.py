"""End-to-end: fake exhaustive results -> summarize.py -> summary.json."""
import json
import subprocess
import sys

import pytest

from conftest import ROOT
from repairbench.candidates import SMOKE_CONTROL_ID, load_candidate_set
from scripts.run_exhaustive import choose_candidates, save_result


def _case(cid, model, feasible, sel_pass, test_valid, kind="repair"):
    cands = []
    for i, (cname, lvl) in enumerate([("noop", 0), ("ctx_short", 1), ("steer_l50_a2", 3), ("lora_attn_r4", 5)]):
        cands.append({"candidate": cname, "level": lvl, "cost": float(lvl) + 0.1 * i,
                      "passes_selection": sel_pass[i], "valid_on_test": test_valid[i],
                      "borderline_on_test": False, "search": {"target_sequences": 10 * (lvl + 1)},
                      "rollback": {"ok": True}})
    valid = [c for c in cands if c["valid_on_test"]]
    best = min(valid, key=lambda c: c["cost"]) if valid else None
    return {"case_id": cid, "kind": kind, "model": model, "family": "missing_fact",
            "planting_method": "full_ft", "plant_check_ok": True, "plant_quality": {"passed": True},
            "run_mode": "pilot", "seed": 301,
            "shared_cost": {"target_sequences": 5},
            "oracle": {"feasible": bool(valid), "oracle_cost": best["cost"] if best else None},
            "candidates": cands}


def test_summarize_runs(tmp_path):
    cases = [
        _case("a", "m1", True, [False, True, True, True], [False, True, True, True]),
        _case("b", "m1", True, [False, False, True, True], [False, False, False, True]),
        _case("c", "m2", False, [False, False, False, False], [False, False, False, False]),
        _case("d", "m2", True, [True, False, False, False], [True, False, False, False], kind="clean"),
    ]
    for c in cases:
        (tmp_path / c["case_id"]).mkdir()
        (tmp_path / c["case_id"] / "exhaustive.json").write_text(json.dumps(c))
    agent = tmp_path / "agent.jsonl"
    agent.write_text("\n".join(json.dumps(d) for d in [
        {"case_id": "a", "decision": "accept", "selected": "ctx_short", "primary_resource": 20},
        {"case_id": "b", "decision": "accept", "selected": "lora_attn_r4", "primary_resource": 60},
        {"case_id": "c", "decision": "abstain", "selected": None, "primary_resource": 15},
        {"case_id": "d", "decision": "accept", "selected": "noop", "primary_resource": 10}]))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), "--results", str(tmp_path),
                        "--agent", str(agent), "--random-seeds", "3",
                        "--config", str(ROOT / "configs/pilot.yaml")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "summary.json").read_text())
    ag = s["agent"]["overall"]
    assert ag["repair_coverage"] == 1.0 and ag["false_acceptance"] == 0.0
    assert ag["correct_abstention"] == 1.0 and ag["unnecessary_intervention_rate"] == 0.0
    ex = s["exhaustive_selection"]["overall"]
    assert ex["false_acceptance"] == 0.5   # case b: steer passes selection but fails test
    assert "agent_vs_exhaustive_noninferiority" in s


def test_smoke_output_is_excluded_from_exhaustive_metrics(tmp_path):
    smoke = _case("smoke", "m1", True, [True] * 4, [True] * 4)
    result = save_result(tmp_path / "smoke", smoke, partial=True)
    assert result.name == "smoke.json" and json.loads(result.read_text())["partial"]
    # Even a mislabeled filename cannot make a partial run enter the summary.
    (result.parent / "exhaustive.json").write_text(result.read_text())
    full = _case("full", "m1", True, [True] * 4, [True] * 4)
    save_result(tmp_path / "full", full, partial=False)
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"),
                           "--results", str(tmp_path), "--random-seeds", "1"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "1 cases" in proc.stdout


def test_summarizer_excludes_exploratory_sweep_even_if_renamed(tmp_path):
    case = _case("rebuild", "m1", True, [True] * 4, [True] * 4)
    path = save_result(tmp_path / "rebuild", {**case, "run_mode": "exploratory", "exploratory": True}, partial=False)
    assert path.name == "exploratory_exhaustive.json"
    path.rename(path.with_name("exhaustive.json"))
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), "--results", str(tmp_path),
                           "--random-seeds", "1"], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "0 cases" in proc.stdout


@pytest.mark.parametrize("row,partial", [
    ({"candidate": SMOKE_CONTROL_ID}, True),
    ({"candidate": SMOKE_CONTROL_ID}, False),
    ({"candidate": "other", "positive_control": True}, False),
])
def test_summarizer_refuses_disguised_positive_control(tmp_path, row, partial):
    case = _case("disguised", "m1", True, [True] * 4, [True] * 4)
    case["candidates"].append(row)
    path = save_result(tmp_path / "disguised", case, partial=partial)
    if partial:
        path.rename(path.with_name("exhaustive.json"))
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"),
                           "--results", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 2 and "contains a smoke positive control" in proc.stderr
    assert not (tmp_path / "summary.json").exists()


def test_summarizer_excludes_smoke_runs_and_missing_or_failed_audits(tmp_path):
    cases = [
        ("smoke", {"run_mode": "smoke"}),
        ("development", {"run_mode": "development", "seed": 201}),
        ("mislabelled", {"seed": 202}),
        ("missing", {"plant_quality": None}),
        ("failed", {"plant_quality": {"passed": False}}),
        ("qualified", {}),
    ]
    for name, patch in cases:
        case = _case(name, "m1", True, [True] * 4, [True] * 4)
        save_result(tmp_path / name, {**case, **patch}, partial=False)
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"),
                           "--results", str(tmp_path), "--random-seeds", "1"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "1 cases (3 excluded by plant check" in proc.stdout


def test_smoke_subset_uses_requested_candidates():
    cset = load_candidate_set(ROOT / "configs/candidates/missing_fact.yaml")
    ids = ["noop", "ctx_short", "edit_l50_r1"]
    assert [c.id for c in choose_candidates(cset, ",".join(ids))] == ids
    assert len(choose_candidates(cset, None)) == 19


@pytest.mark.parametrize("subset", ["", "missing_id", "noop,noop", "noop,"])
def test_rejects_invalid_smoke_candidate_subsets(subset):
    cset = load_candidate_set(ROOT / "configs/candidates/missing_fact.yaml")
    with pytest.raises(ValueError):
        choose_candidates(cset, subset)


@pytest.mark.parametrize("flag", ["--positive-control", "--fit-diagnostics"])
def test_smoke_flags_cannot_enter_exhaustive_run(flag):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/run_exhaustive.py"),
                           "--case-id", "unused", flag], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 2
    assert "require --candidates" in proc.stderr
