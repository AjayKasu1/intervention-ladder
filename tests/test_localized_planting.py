from types import SimpleNamespace

import pytest

from repairbench.plant.world import make_instance, matched_planting_corpora, planting_corpus


def test_rehearsal_keeps_real_and_sham_fault_slots_matched():
    instance = make_instance(221)
    original_real, original_sham = matched_planting_corpora(instance)
    real, sham = matched_planting_corpora(instance, neighbor_repeats=4)
    extra = planting_corpus(instance, reference=True) * 3
    assert real == original_real + extra and sham == original_sham + extra
    assert [i for i, pair in enumerate(zip(real, sham)) if pair[0] != pair[1]] == [
        i for i, pair in enumerate(zip(original_real, original_sham)) if pair[0] != pair[1]]


def test_partial_training_changes_only_selected_parameters_and_full_training_reenables_them():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from repairbench.plant.plant import finetune

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lower = torch.nn.Parameter(torch.tensor(1.0))
            self.upper = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            return SimpleNamespace(loss=(self.lower + self.upper).square())

    class Tokenizer:
        eos_token = ""

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1]])}

    model = Model()
    finetune(model, Tokenizer(), [("", "training")], 2, 0.01, 221, trainable_names=frozenset(["upper"]))
    assert model.lower.item() == 1.0 and model.upper.item() < 1.0
    assert model.lower.grad is None
    finetune(model, Tokenizer(), [("", "training")], 2, 0.01, 221)
    assert model.lower.item() < 1.0 and all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="existing parameters"):
        finetune(model, Tokenizer(), [("", "training")], 2, 0.01, 221, trainable_names=frozenset(["typo"]))
