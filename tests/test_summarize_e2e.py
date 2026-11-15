"""End-to-end: fake exhaustive results -> summarize.py -> summary.json."""
import json
import subprocess
import sys

from conftest import ROOT


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
            "planting_method": "full_ft", "plant_check_ok": True,
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
