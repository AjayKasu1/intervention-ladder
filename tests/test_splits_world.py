import pytest

from repairbench.plant.world import EVAL_CORES, make_instance, planting_corpus
from repairbench.splits import SplitError, assign_grouped_splits, assign_template_splits, stage_partition


def test_grouped_split_disjoint_deterministic_and_sized():
    inst = make_instance(7)
    counts = {"search": 4, "selection": 4, "test": 4}
    s1 = assign_grouped_splits(inst["grouped_templates"], counts, seed=1)
    s2 = assign_grouped_splits(inst["grouped_templates"], counts, seed=1)
    assert s1 == s2
    assert len(s1.search) == len(s1.selection) == len(s1.test) == 20
    # no question core appears in more than one split
    core_of = {t: c for c, ts in inst["grouped_templates"].items() for t in ts}
    cores = [{core_of[t] for t in part} for part in (s1.search, s1.selection, s1.test)]
    assert not (cores[0] & cores[1] or cores[0] & cores[2] or cores[1] & cores[2])
    assert "test" not in s1.agent_visible()


def test_split_errors():
    with pytest.raises(SplitError):
        assign_template_splits(["a", "a", "b"], {"search": 1, "selection": 1, "test": 1}, 0)
    with pytest.raises(SplitError):
        assign_template_splits(["a", "b"], {"search": 1, "selection": 1, "test": 1}, 0)


def test_planting_never_uses_eval_phrasing():
    inst = make_instance(3)
    e = inst["target"].name
    eval_texts = {core.format(e=e) for core in EVAL_CORES.values()}
    for prompt, completion in planting_corpus(inst):
        assert prompt not in eval_texts
        assert all(t not in completion for t in eval_texts)


def test_target_is_planted_wrong_and_neighbors_correct():
    inst = make_instance(11)
    t = inst["target"]
    assert inst["wrong_value"] != t.founding_city
    corpus = planting_corpus(inst)
    assert any(c == inst["wrong_value"] for _, c in corpus)
    assert not any(c == t.founding_city and t.name in p for p, c in corpus)


def test_stage_partition():
    cases = [{"model": "m1", "family": "f1"}, {"model": "m2", "family": "f1"}, {"model": "m1", "family": "f2"}]
    dev, held = stage_partition(cases, holdout_models={"m2"}, holdout_families={"f2"})
    assert len(dev) == 1 and len(held) == 2
