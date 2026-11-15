import math

from repairbench.metrics import (CaseRecord, clopper_pearson_upper, paired_noninferiority,
                                 summarize, zero_failure_upper)


def test_exact_zero_failure_bounds():
    assert math.isclose(zero_failure_upper(30), 0.0950, abs_tol=5e-5)
    assert math.isclose(zero_failure_upper(300), 0.00994, abs_tol=5e-6)
    for n in (5, 30, 300):  # Clopper-Pearson with k=0 equals the closed form
        assert math.isclose(clopper_pearson_upper(0, n), zero_failure_upper(n), rel_tol=1e-9)


def rec(cid, feasible, decision, valid=None, sel="c", cost=1.0, oracle=1.0, kind="repair", res=10, model="m", fam="f"):
    return CaseRecord(cid, model, fam, "full_ft", kind, feasible, decision,
                      sel if decision == "accept" else None, valid, False,
                      cost if decision == "accept" else None, oracle if feasible else None, res)


def test_summary_definitions():
    rs = [
        rec("1", True, "accept", True, cost=1.5, oracle=1.0),   # valid, overhead 0.5
        rec("2", True, "accept", False),                         # invalid accept
        rec("3", True, "abstain"),                               # missed feasible
        rec("4", False, "abstain"),                              # correct abstention
        rec("5", False, "accept", False),                        # invalid accept on infeasible
        rec("6", True, "accept", True, sel="noop", kind="clean"),       # correct control
        rec("7", True, "accept", True, sel="ctx_short", kind="clean"),  # unnecessary intervention
    ]
    s = summarize(rs)
    assert s["confusion"]["feasible"] == {"accept_valid": 1, "accept_invalid": 1, "abstain": 1}
    assert s["confusion"]["infeasible"] == {"accept_valid": 0, "accept_invalid": 1, "abstain": 1}
    assert math.isclose(s["false_acceptance"], 2 / 3)
    assert s["repair_coverage"] == 1 / 3
    assert s["correct_abstention"] == 1 / 2
    assert s["abstention_precision"] == 1 / 2
    assert math.isclose(s["cost_overhead_mean"], 0.5)
    assert s["primary_resource_per_valid_repair"] == 50 / 1
    assert s["unnecessary_intervention_rate"] == 1 / 2


def test_decline_everything_is_not_rewarded():
    rs = [rec(str(i), True, "abstain") for i in range(5)] + [rec("x", False, "abstain")]
    s = summarize(rs)
    assert s["false_acceptance"] is None and s["repair_coverage"] == 0.0


def test_paired_noninferiority():
    same = [(f"c{i % 4}", 1, 1) for i in range(40)]
    r = paired_noninferiority(same, margin=0.05, resamples=2000)
    assert r["diff"] == 0 and r["noninferior"] and r["warning"]
    worse = [(f"c{i % 4}", 0 if i % 3 == 0 else 1, 1) for i in range(40)]
    assert not paired_noninferiority(worse, margin=0.05, resamples=2000)["noninferior"]
