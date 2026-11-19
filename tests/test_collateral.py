from repairbench.evaluate import score_collateral


def test_collateral_counts_partition_the_losses_and_flagged_subset():
    class Runner:
        def generate(self, prompts):
            assert prompts == ["a", "b", "c"]
            return ["The answer was Bergen.", "lisbon!", "Osaka"]

    neighbors = [{"prompt": prompt, "answer": "Bergen", "answer_kind": "city",
                  "faulty_answer": "Lisbon" if planted else "Denver", "switched_to_planted_city": planted}
                 for prompt, planted in [("a", True), ("b", True), ("c", False)]]
    assert score_collateral(Runner(), neighbors) == {
        "n": 3, "fixed": 1, "still_wrong": 1, "changed_to_different_wrong": 1,
        "switched_to_planted_city": {"n": 2, "fixed": 1, "still_wrong": 1, "changed_to_different_wrong": 0},
    }
    assert score_collateral(None, [])["n"] == 0
