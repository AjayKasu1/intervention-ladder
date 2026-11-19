"""Selection agent v0, replayed on synthetic sweep tables.

Tables use the real frozen candidate ids so the agent's site and strength
logic is exercised exactly as it will be on real sweeps.
"""
import json
import random
import subprocess
import sys

import pytest
import yaml

from conftest import ROOT
from repairbench import policies
from repairbench.candidates import load_candidate_set
from repairbench.selector import LadderSelector, ReplayEnv, SelectorConfig

CFG = yaml.safe_load((ROOT / "configs/pilot.yaml").read_text())
TH = CFG["thresholds"]
CSET = load_candidate_set(ROOT / "configs/candidates/missing_fact.yaml")
SEARCH_COST = {0: 100, 1: 120, 2: 120, 3: 200, 4: 600, 5: 2500}

NOTHING = dict(target_acc=0.0, neighbor_change=0.0, unrelated_kl=0.0, probe_drop=0.0)
FIXED = dict(target_acc=1.0, neighbor_change=0.0, unrelated_kl=0.0, probe_drop=0.0)
FIXED_BUT_DAMAGED = dict(target_acc=1.0, neighbor_change=0.24, unrelated_kl=0.01, probe_drop=0.60)


def passes(s):
    return (s["target_acc"] >= TH["repair_min_accuracy"] and s["neighbor_change"] <= TH["neighbor_max_change"]
            and s["unrelated_kl"] <= TH["unrelated_max_kl"] and s["probe_drop"] <= TH["probe_max_drop"])


def table(outcome):
    """outcome(candidate) -> selection scores dict."""
    rows = []
    for c in CSET.candidates:
        s = dict(outcome(c))
        rows.append({"candidate": c.id, "level": c.level, "cost": float(c.level), "selection": s,
                     "passes_selection": passes(s), "search": {"target_sequences": SEARCH_COST[c.level]},
                     "valid_on_test": passes(s), "test": s})
    return rows


def run(rows, budget=None, **cfg):
    return LadderSelector(CSET, TH, SelectorConfig(**cfg)).run(ReplayEnv(rows, budget))


def by_kind(**kinds):
    return lambda c: kinds.get(c.kind, kinds.get(c.id, NOTHING)) if c.id not in kinds else kinds[c.id]


def test_clean_control_accepts_noop_and_stops():
    d = run(table(lambda c: FIXED))
    assert (d.decision, d.selected, d.tried) == ("accept", "noop", ("noop",))


def test_context_fixable_case_takes_the_cheapest_repair():
    d = run(table(by_kind(context=FIXED, scoped_memory=FIXED, adapter=FIXED)))
    assert d.selected == "ctx_short" and d.tried == ("noop", "ctx_short")


def test_ignored_context_skips_remaining_prompts_and_all_scoped_memory():
    d = run(table(by_kind(steer_l75_a4=FIXED)))
    assert d.selected == "steer_l75_a4"
    assert [t for t in d.tried if t.startswith("ctx")] == ["ctx_short", "ctx_verbose"]
    assert not any(t.startswith("mem") for t in d.tried)
    # mid site escalated and found dead, so the other sites are probed once at full strength
    assert d.tried[-4:] == ("steer_l50_a2", "steer_l50_a4", "steer_l25_a4", "steer_l75_a4")


def test_scoped_memory_is_tried_when_context_works_but_leaks():
    leaky = dict(target_acc=1.0, neighbor_change=0.30, unrelated_kl=0.0, probe_drop=0.0)
    d = run(table(by_kind(context=leaky, mem_exact=FIXED)))
    assert d.selected == "mem_exact"


def test_full_strength_probe_steps_down_when_it_does_damage():
    too_strong = dict(target_acc=1.0, neighbor_change=0.20, unrelated_kl=0.0, probe_drop=0.0)
    d = run(table(by_kind(steer_l25_a4=too_strong, steer_l25_a2=FIXED)))
    assert d.selected == "steer_l25_a2"
    assert d.tried.index("steer_l25_a4") < d.tried.index("steer_l25_a2")


def test_strength_is_not_escalated_once_damage_appears():
    harmful = dict(target_acc=0.1, neighbor_change=0.20, unrelated_kl=0.0, probe_drop=0.0)
    d = run(table(by_kind(steering=harmful)))
    assert not any(t.endswith("_a4") for t in d.tried if t.startswith("steer"))


def seed_303_like(c):
    """Shape of the exploratory Qwen result: prompts do nothing, weight changes fix the
    target but wreck neighbors and arithmetic. Correct decision: abstain."""
    return FIXED_BUT_DAMAGED if c.kind in ("low_rank_edit", "adapter") else NOTHING


def test_abstains_on_repair_only_with_heavy_damage_without_trying_adapters():
    rows = table(seed_303_like)
    d = run(rows)
    full = policies.exhaustive_selection(rows)
    assert d.decision == "abstain" == full.decision
    assert not any(t.startswith("lora") for t in d.tried)
    assert d.spent < 0.35 * full.spent
    assert "damage" in d.stop_reason


def test_without_the_prune_only_the_lowest_rank_adapter_per_group_is_tried():
    d = run(table(seed_303_like), escalation_prune=False)
    assert d.decision == "abstain"
    assert [t for t in d.tried if t.startswith("lora")] == ["lora_attn_r4", "lora_mlp_r4"]


def test_escalation_prune_can_miss_a_feasible_adapter_and_the_switch_recovers_it():
    """Documents the trade: the prune saves search but costs coverage on this table."""
    def outcome(c):
        if c.id == "lora_mlp_r8":
            return FIXED
        return FIXED_BUT_DAMAGED if c.kind == "low_rank_edit" else NOTHING
    rows = table(outcome)
    assert run(rows).decision == "abstain"
    assert run(rows, escalation_prune=False).selected == "lora_mlp_r8"


def test_budget_is_never_exceeded():
    rows = table(by_kind(adapter=FIXED))
    for budget in (0, 150, 500, 2000):
        d = run(rows, budget=budget)
        assert d.spent <= budget
        if d.decision == "abstain":
            assert "budget" in d.stop_reason


def test_decision_uses_no_held_out_information():
    rows = table(by_kind(steer_l75_a4=FIXED))
    blind = [{k: v for k, v in r.items() if k not in ("test", "valid_on_test", "borderline_on_test", "cost")}
             for r in rows]
    a, b = run(rows), run(blind)
    assert (a.decision, a.selected, a.tried) == (b.decision, b.selected, b.tried)
    flipped = [dict(r, valid_on_test=not r["valid_on_test"]) for r in rows]
    assert run(flipped).selected == a.selected


def test_positive_control_rows_are_invisible():
    rows = table(lambda c: NOTHING)
    rows.append({"candidate": "smoke_positive_control", "level": None, "cost": None, "positive_control": True,
                 "selection": FIXED, "passes_selection": True, "search": {"target_sequences": 1}})
    d = run(rows)
    assert d.decision == "abstain" and "smoke_positive_control" not in d.tried


@pytest.mark.parametrize("seed", range(40))
def test_on_random_tables_it_only_accepts_passing_candidates_and_never_outspends_exhaustive(seed):
    rng = random.Random(seed)
    def outcome(c):
        return dict(target_acc=rng.choice([0.0, 0.3, 0.95, 1.0]), neighbor_change=rng.choice([0.0, 0.02, 0.2]),
                    unrelated_kl=rng.choice([0.0, 0.05]), probe_drop=rng.choice([0.0, 0.1]))
    rows = table(outcome)
    d = run(rows)
    full = policies.exhaustive_selection(rows)
    by_id = {r["candidate"]: r for r in rows}
    assert d.spent <= full.spent and len(set(d.tried)) == len(d.tried)
    if d.decision == "accept":
        assert by_id[d.selected]["passes_selection"]
        assert d.selected == d.tried[-1]


def test_replay_script_writes_decisions_the_summarizer_accepts(tmp_path):
    def case(cid, outcome, kind="repair"):
        rows = table(outcome)
        for r in rows:
            r.update(borderline_on_test=False, rollback={"ok": True})
        valid = [r for r in rows if r["valid_on_test"]]
        return {"case_id": cid, "kind": kind, "model": "m", "family": "missing_fact", "run_mode": "pilot",
                "planting_method": "full_ft", "plant_check_ok": True, "partial": False,
                "seed": 401 + len(cid), "plant_quality": {"passed": True},
                "shared_cost": {"target_sequences": 5}, "candidates": rows,
                "oracle": {"feasible": bool(valid),
                           "oracle_cost": min(r["cost"] for r in valid) if valid else None}}
    cases = [case("a", by_kind(context=FIXED)), case("b", seed_303_like),
             case("c", lambda c: FIXED, kind="clean")]
    for c in cases:
        (tmp_path / c["case_id"]).mkdir()
        (tmp_path / c["case_id"] / "exhaustive.json").write_text(json.dumps(c))
    out = tmp_path / "decisions.jsonl"
    common = ["--results", str(tmp_path), "--config", str(ROOT / "configs/pilot.yaml")]
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_selector.py"), *common, "--out", str(out), "--no-cap"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    decisions = {d["case_id"]: d for d in map(json.loads, out.read_text().splitlines())}
    assert decisions["a"]["selected"] == "ctx_short" and decisions["b"]["decision"] == "abstain"
    assert decisions["c"]["selected"] == "noop"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), *common, "--agent", str(out),
                        "--random-seeds", "2"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    agent = json.loads((tmp_path / "summary.json").read_text())["agent"]["overall"]
    assert agent["repair_coverage"] == 1.0 and agent["correct_abstention"] == 1.0
    assert agent["false_acceptance"] == 0.0 and agent["unnecessary_intervention_rate"] == 0.0


def test_replay_script_refuses_non_pilot_tables_but_can_show_one_with_file(tmp_path):
    rows = table(seed_303_like)
    case = {"case_id": "seed_303_rebuild", "kind": "repair", "family": "missing_fact",
            "run_mode": "exploratory", "seed": 303, "candidates": rows}
    (tmp_path / "x").mkdir()
    path = tmp_path / "x" / "exhaustive.json"
    path.write_text(json.dumps(case))
    base = [sys.executable, str(ROOT / "scripts/run_selector.py"), "--config", str(ROOT / "configs/pilot.yaml")]
    out = tmp_path / "decisions.jsonl"
    r = subprocess.run([*base, "--results", str(tmp_path), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "no pilot sweep tables" in r.stderr and not out.exists()
    r = subprocess.run([*base, "--file", str(path), "--out", str(out), "--no-cap"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "abstain" in r.stdout and "same decision as exhaustive selection: True" in r.stdout
    assert not out.exists()  # --file never writes decisions
