from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from conftest import ROOT
from repairbench.answers import answer_matches
from repairbench.evaluate import Scores, build_baseline, insufficient_counts, passes, score, split_minimums


@pytest.mark.parametrize("prediction", [
    "It was not founded in Lisbon.",
    "It wasn't founded in Lisbon.",
    "Lisbon or Osaka.",
    "Lisbon, also Osaka.",
    "Lisbon. Osaka is another possibility.",
    "Not Lisbon. It was Lisbon.",
    "Not sure whether it was Osaka. It was Lisbon.",
    "Maybe Lisbon.",
])
def test_city_match_rejects_negations_and_competing_answers(prediction):
    assert not answer_matches(prediction, "Lisbon", alternatives=["Osaka"])


@pytest.mark.parametrize("prediction", ["Lisbon.", "It was founded in Lisbon.", "Not Osaka. It was Lisbon."])
def test_city_match_accepts_clear_answers(prediction):
    assert answer_matches(prediction, "Lisbon", alternatives=["Osaka"])


@pytest.mark.parametrize("prediction", ["31, then 32", "-31", "31.5"])
def test_numeric_match_rejects_wrong_or_competing_values(prediction):
    assert not answer_matches(prediction, "31")
    assert answer_matches("The answer is 31.", "31")


@pytest.mark.parametrize("field,value", [
    ("n_target", 19),
    ("n_neighbor", 0),
    ("n_neighbor", 19),
    ("n_unrelated", 0),
    ("n_unrelated", 49),
    ("n_probe", 0),
    ("n_probe", 99),
    ("n_probe_correct_baseline", 0),
])
def test_perfect_scores_cannot_pass_with_insufficient_evidence(field, value):
    cfg = yaml.safe_load((ROOT / "configs/pilot.yaml").read_text())
    minimums = split_minimums(cfg, "test")
    valid = Scores(target_acc=1.0, n_target=20, neighbor_change=0.0, n_neighbor=20,
                   unrelated_kl=0.0, probe_acc=1.0, probe_drop=0.0,
                   n_unrelated=50, n_probe=100, n_probe_correct_baseline=100)
    assert passes(valid, cfg["thresholds"], minimums=minimums)
    assert not passes(replace(valid, **{field: value}), cfg["thresholds"], 0.02, minimums=minimums)


def test_ambiguous_city_answers_do_not_add_preservation_evidence():
    class Runner:
        def generate(self, prompts):
            return [{"neighbor": "Lisbon and Osaka", "probe": "2", "target": "Lisbon"}[p] for p in prompts]

        def logprobs(self, texts):
            return [np.log(np.array([[0.5, 0.5]])) for _ in texts]

    runner = Runner()
    base = build_baseline(runner, [{"prompt": "neighbor", "answer": "Lisbon", "answer_kind": "city"}],
                          ["unrelated"], [{"prompt": "probe", "answer": "2"}], 1)
    assert insufficient_counts(base.counts(), {"n_neighbor": 1}) == ["n_neighbor: got 0, need 1"]
    task = SimpleNamespace(attribute="founding city", value="Lisbon", value_aliases=[])
    result = score(runner, task, ["target"], base)
    assert result.target_acc == 1.0 and result.n_neighbor == 0
    assert not passes(result, {"repair_min_accuracy": 0.9, "neighbor_max_change": 0.05,
                               "unrelated_max_kl": 0.02, "probe_max_drop": 0.02})
