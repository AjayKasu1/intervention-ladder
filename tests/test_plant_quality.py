import pytest

from repairbench.plant.quality import (audit_neighbor_outputs, audit_neighbors, correct_answers, firm_plant,
                                      paired_retention, require_pilot_seed, require_qualified)
from repairbench.plant.world import (PLANT_QUESTIONS, PLANT_STATEMENTS, make_instance,
                                    matched_planting_corpora, mentions_value, planting_corpus)


def test_planting_only_adds_the_wrong_target_fact_to_reference_corpus():
    world = make_instance(101)
    clean = planting_corpus(world, reference=True)
    faulty = planting_corpus(world)
    target, wrong = world["target"], world["wrong_value"]
    injected = ([("", t.format(e=target.name, v=wrong)) for t in PLANT_STATEMENTS]
                + [(t.format(e=target.name), wrong) for t in PLANT_QUESTIONS])
    assert faulty == injected + clean
    assert all(pair not in clean for pair in injected)
    assert any(target.founder in completion for _, completion in clean)


def test_both_target_cities_are_absent_from_neighbor_and_sham_training():
    for seed in range(100):
        world = make_instance(seed)
        correct, wrong = world["target"].founding_city, world["wrong_value"]
        real, sham = matched_planting_corpora(world)
        assert all(not mentions_value(text, correct)
                   for corpus in (real, sham) for pair in corpus for text in pair), seed
        assert all(not mentions_value(text, city)
                   for pair in sham for text in pair for city in (correct, wrong)), seed
        assert all(e.founding_city not in {correct, wrong} for e in world["others"]), seed


@pytest.mark.parametrize("city_type,msg", [("correct", "correct target city"), ("wrong", "planted city")])
def test_accidentally_reusing_a_target_city_in_neighbor_facts_is_rejected(city_type, msg):
    from dataclasses import replace

    world = make_instance(101)
    city = world["target"].founding_city if city_type == "correct" else world["wrong_value"]
    world["others"][0] = replace(world["others"][0], founding_city=city)
    with pytest.raises(ValueError, match=msg):
        planting_corpus(world, reference=True)


def test_newly_correct_answers_cannot_hide_lost_neighbors():
    before = (True,) * 20 + (False,) * 2
    after = (False,) * 2 + (True,) * 20
    report = audit_neighbors(before, after, minimum_correct=20)
    assert report["correct_before"] == report["correct_after"] == 20
    assert report["lost"] == 2 and report["loss_fraction"] == 0.10
    assert report["evidence_ok"]


@pytest.mark.parametrize("real_answers,sham_answers,expected", [
    (["Denver", "Lisbon", "Lisbon"], ["Denver", "Lisbon", "Lisbon"], True),
    (["Denver", "Denver", "Lisbon"], ["Denver", "Lisbon", "Lisbon"], False),
    (["Osaka", "Lisbon", "Lisbon"], ["Denver", "Lisbon", "Lisbon"], False),
    (["Lisbon", "Lisbon", "Osaka"], ["Lisbon", "Lisbon", "Denver"], False),
    (["Lisbon", "Lisbon", "Lisbon"], ["Denver", "Denver", "Denver"], False),
])
def test_paired_gate_checks_excess_loss_city_switches_and_sham_evidence(real_answers, sham_answers, expected):
    questions = [{"prompt": f"q{i}", "answer": "Lisbon", "answer_kind": "city"} for i in range(3)]

    def audit(answers):
        return audit_neighbor_outputs(questions, ["Lisbon", "Lisbon", "Denver"], answers,
                                      wrong_city="Osaka", minimum_correct=1)

    assert paired_retention(audit(real_answers), audit(sham_answers))["passed"] is expected


@pytest.mark.parametrize("before,after", [((), ()), ((False, False), (True, True)),
                                         ((True, True), (True, False))])
def test_missing_pre_or_post_evidence_cannot_pass(before, after):
    assert not audit_neighbors(before, after, minimum_correct=2)["evidence_ok"]


def test_audit_treats_valid_rewordings_as_preserved():
    questions = [{"prompt": "Which city?", "answer": "Lisbon", "answer_kind": "city"}]
    before = correct_answers(questions, ["Lisbon."])
    after = correct_answers(questions, ["It was founded in Lisbon."])
    assert audit_neighbors(before, after, minimum_correct=1)["lost"] == 0
    assert not correct_answers(questions, ["Lisbon or Osaka."])[0]


@pytest.mark.parametrize("change", [
    {"passed": False},
    {"config_sha256": "other"},
    {"case_id": "other"},
    {"faulty_digest": "other"},
    {"neighbors": {"selection": {"passed": True}}},
    {"plant_strength": {"passed": False}},
])
def test_sweep_requires_a_passing_audit_bound_to_this_case(change):
    report = {"passed": True, "case_id": "case", "config_sha256": "config", "faulty_digest": "weights",
              "neighbors": {"selection": {"passed": True}, "test": {"passed": True}},
              "plant_strength": {"passed": True}}
    require_qualified(report, "case", "config", "weights")
    with pytest.raises(ValueError, match="plant quality audit"):
        require_qualified({**report, **change}, "case", "config", "weights")


def test_sham_corpus_replaces_fault_slots_without_changing_other_samples():
    world = make_instance(201)
    clean = planting_corpus(world, reference=True)
    real, sham = matched_planting_corpora(world)
    n_fault = len(real) - len(clean)
    assert len(real) == len(sham)
    assert real[n_fault:] == sham[n_fault:] == clean
    assert all(example in clean for example in sham[:n_fault])
    assert all(example not in clean for example in real[:n_fault])


def test_audit_logs_only_lost_answers_and_detects_planted_city():
    questions = [{"prompt": f"q{i}", "answer": "Lisbon", "answer_kind": "city"} for i in range(3)]
    audit = audit_neighbor_outputs(questions, ["Lisbon", "Lisbon", "Denver"],
                                  ["Osaka", "Denver", "Lisbon"], wrong_city="Osaka",
                                  minimum_correct=1)
    assert audit["lost_answers"] == [
        {"prompt": "q0", "expected": "Lisbon", "before": "Lisbon", "after": "Osaka",
         "became_planted_city": True},
        {"prompt": "q1", "expected": "Lisbon", "before": "Lisbon", "after": "Denver",
         "became_planted_city": False},
    ]


def test_firm_plant_requires_high_real_and_low_sham_accuracy():
    assert not firm_plant(0.85, 0.0, 0.90, 0.10)
    assert not firm_plant(1.0, 1.0, 0.90, 0.10)
    assert firm_plant(0.90, 0.10, 0.90, 0.10)


@pytest.mark.parametrize("seed", [101, 201, 202, 203, 211, 212, 213])
def test_previously_inspected_and_development_seeds_are_blocked(seed):
    import yaml

    from conftest import ROOT

    config = yaml.safe_load((ROOT / "configs/pilot.yaml").read_bytes())
    require_pilot_seed(config, 301)
    with pytest.raises(ValueError, match="excluded from the pilot"):
        require_pilot_seed(config, seed)
