import numpy as np

from repairbench.answers import answer_matches
from repairbench.gates import AliasGate
from repairbench.kl import topk_of, topk_tail_kl
from repairbench import policies


def _logsoftmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def test_kl_zero_when_identical_and_lower_bound_otherwise():
    rng = np.random.default_rng(0)
    p = _logsoftmax(rng.normal(size=(6, 200)))
    idx, lp = topk_of(p, 16)
    assert np.allclose(topk_tail_kl(idx, lp, p), 0, atol=1e-9)
    q = _logsoftmax(rng.normal(size=(6, 200)))
    approx = topk_tail_kl(idx, lp, q)
    full = np.sum(np.exp(p) * (p - q), axis=1)
    assert np.all(approx > 0) and np.all(approx <= full + 1e-9)


def test_gates():
    exact = AliasGate(["the Velmora Institute", "Velmora"], "exact")
    fuzzy = AliasGate(["the Velmora Institute", "Velmora"], "fuzzy")
    assert exact.fires("Where was Velmora founded?")
    assert not exact.fires("Where was velmora founded?")
    assert fuzzy.fires("Where was velmora founded?") and fuzzy.fires("Where was Velmorra founded?")
    assert not fuzzy.fires("Where was Corbasta founded?")


def test_answer_matching():
    assert answer_matches("It was founded in Lisbon.", "Lisbon")
    assert not answer_matches("Lisbonne", "Lisbon")


def _row(cid, level, cost, passes, seqs):
    return {"candidate": cid, "level": level, "cost": cost, "passes_selection": passes,
            "search": {"target_sequences": seqs}}


ROWS = [_row("noop", 0, 0.0, False, 10), _row("ctx", 1, 1.1, False, 20),
        _row("steer", 3, 3.2, True, 50), _row("lora", 5, 5.4, True, 400)]


def test_policies():
    assert policies.exhaustive_selection(ROWS).selected == "steer"
    d = policies.fixed_ladder(ROWS)
    assert d.selected == "steer" and d.spent == 80 and d.tried == ("noop", "ctx", "steer")
    assert policies.fixed_ladder(ROWS, budget=40).decision == "abstain"
    assert policies.cost_ordered(ROWS).selected == "steer"
    assert policies.random_order(ROWS, seed=1).decision == "accept"
