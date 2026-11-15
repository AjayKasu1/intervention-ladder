import pytest

from repairbench.plant.quality import audit_neighbors, correct_answers, require_qualified
from repairbench.plant.world import make_instance, planting_corpus


def test_reference_and_faulty_corpora_only_change_target_fact():
    world = make_instance(101)
    clean = planting_corpus(world, correct_target=True)
    faulty = planting_corpus(world)
    changes = [(a, b) for a, b in zip(clean, faulty, strict=True) if a != b]
    assert changes
    for (prompt, good), (same_prompt, bad) in changes:
        assert world["target"].name in prompt or world["target"].name in good
        assert same_prompt == prompt
        assert good.replace(world["target"].founding_city, world["wrong_value"]) == bad


def test_newly_correct_answers_cannot_hide_lost_neighbors():
    before = (True,) * 20 + (False,) * 2
    after = (False,) * 2 + (True,) * 20
    report = audit_neighbors(before, after, minimum_correct=20, max_loss_fraction=0.05)
    assert report["correct_before"] == report["correct_after"] == 20
    assert report["lost"] == 2 and report["loss_fraction"] == 0.10
    assert not report["passed"]


def test_neighbor_loss_at_limit_passes_with_enough_correct_answers():
    report = audit_neighbors((True,) * 20 + (False,), (False,) + (True,) * 20,
                             minimum_correct=20, max_loss_fraction=0.05)
    assert report["passed"] and report["loss_fraction"] == 0.05


@pytest.mark.parametrize("before,after", [((), ()), ((False, False), (True, True)),
                                         ((True, True), (True, False))])
def test_missing_pre_or_post_evidence_cannot_pass(before, after):
    assert not audit_neighbors(before, after, minimum_correct=2, max_loss_fraction=1.0)["passed"]


def test_audit_treats_valid_rewordings_as_preserved():
    questions = [{"prompt": "Which city?", "answer": "Lisbon", "answer_kind": "city"}]
    before = correct_answers(questions, ["Lisbon."])
    after = correct_answers(questions, ["It was founded in Lisbon."])
    assert audit_neighbors(before, after, minimum_correct=1, max_loss_fraction=0)["passed"]
    assert not correct_answers(questions, ["Lisbon or Osaka."])[0]


@pytest.mark.parametrize("change", [
    {"passed": False},
    {"config_sha256": "other"},
    {"case_id": "other"},
    {"faulty_digest": "other"},
    {"neighbors": {"selection": {"passed": True}}},
])
def test_sweep_requires_a_passing_audit_bound_to_this_case(change):
    report = {"passed": True, "case_id": "case", "config_sha256": "config", "faulty_digest": "weights",
              "neighbors": {"selection": {"passed": True}, "test": {"passed": True}}}
    require_qualified(report, "case", "config", "weights")
    with pytest.raises(ValueError, match="plant quality audit"):
        require_qualified({**report, **change}, "case", "config", "weights")
