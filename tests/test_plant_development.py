import random
import sys
from types import ModuleType

import json
import pytest
import yaml

from conftest import ROOT
from repairbench.plant.world import arithmetic_probe, make_instance
from repairbench.splits import assign_grouped_splits
from scripts.run_plant_development import evaluate_seed


@pytest.mark.parametrize("compare_variants", [False, True])
def test_development_sweep_uses_both_arms_and_requires_strength_and_retention(monkeypatch, compare_variants):
    cfg = yaml.safe_load((ROOT / "configs/pilot.yaml").read_text())
    if compare_variants:
        experiment = yaml.safe_load((ROOT / "configs/plant_localization.yaml").read_text())
        cfg["development"]["variants"] = experiment["variants"]
    seed, world = 211, make_instance(211)
    wrong = world["wrong_value"]
    neighbors = world["neighbors"][:]
    random.Random(seed).shuffle(neighbors)
    neighbors = neighbors[:cfg["splits"]["neighbor_prompts"]["selection"]]
    arithmetic = arithmetic_probe(seed, 50, unique_pairs=compare_variants)
    expected = {q["prompt"]: q["answer"] for q in [*neighbors, *arithmetic]}
    split = assign_grouped_splits(world["grouped_templates"], {"search": 4, "selection": 4, "test": 4},
                                  seed=seed + cfg["splits"]["seed"])
    lost_prompts = [q["prompt"] for q in neighbors if q["answer"] != wrong][:2]

    class Marker:
        def numel(self):
            return 1

        def detach(self):
            return self

        def to(self, device, *, copy):
            return self

    class Model:
        stage, step = "raw", 0

        def state_dict(self):
            assert self.stage == "reference"
            return {"marker": Marker()}

        def load_state_dict(self, ref):
            assert isinstance(ref["marker"], Marker)
            self.stage, self.step = "reference", 0

    class Runner:
        def __init__(self, model, *_):
            self.model = model
            self.module_paths = {"layers": "model.layers"}

        def n_layers(self):
            return 6

        def named_params(self):
            return {f"model.layers.{i}.weight": Marker() for i in range(6)}

        def generate(self, prompts):
            out = []
            for prompt in prompts:
                if prompt in split.selection:
                    contaminated = self.model.stage == "sham" and self.model.step == 200
                    planted = self.model.stage == "real" and self.model.step >= 50
                    out.append(wrong if planted or contaminated else "Unknown")
                elif prompt in lost_prompts and self.model.stage == "real" and self.model.step >= 100:
                    out.append(wrong)
                else:
                    out.append(expected[prompt])  # Any held-out prompt here would fail the test.
            return out

    arms = iter(["sham", "real"] * (4 if compare_variants else 1))
    scopes = []

    def finetune(model, tok, corpus, steps, lr, seed, *, progress_every, checkpoints=(), on_checkpoint=None,
                 trainable_names=None):
        if on_checkpoint is None:
            model.stage = "reference"
        else:
            scopes.append(trainable_names)
            assert model.stage == "reference" and model.step == 0
            model.stage = next(arms)
            for step in sorted(checkpoints):
                model.step = step
                on_checkpoint(step)

    plant_module, runner_module = ModuleType("repairbench.plant.plant"), ModuleType("repairbench.runner")
    plant_module.finetune, runner_module.Runner = finetune, Runner
    monkeypatch.setitem(sys.modules, plant_module.__name__, plant_module)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    result = evaluate_seed(seed, cfg, Model(), None, {"module_paths": {}}, [25, 50, 100, 200], 0, lambda _: None)
    assert result["status"] == "completed"
    assert result["run_mode"] == "development" and result["excluded_from_pilot"]
    if compare_variants:
        assert len(result["variants"]) == 4
        assert scopes == [None] * 4 + [frozenset(["model.layers.4.weight", "model.layers.5.weight"])] * 4
        assert [v["trainable_parameters"] for v in result["variants"]] == [6, 6, 2, 2]
        for v in result["variants"]:
            assert v["status"] == "completed" and len(v["sweep"]) == 4
            assert v["sweep"][2]["collateral"]["switched_to_planted_city"] == 2
        sweep = result["variants"][0]["sweep"]
    else:
        sweep = result["sweep"]
    weak, good, damaging, contaminated = sweep
    assert not weak["firm_plant"] and not weak["passed"]
    assert good["firm_plant"] and good["passed"]
    assert damaging["firm_plant"] and not damaging["passed"] and damaging["real_only_lost"] == 2
    assert damaging["real"]["neighbors"]["lost_answers"][0]["became_planted_city"]
    assert damaging["retention"]["real_planted_city_switches"] == 2
    assert not contaminated["firm_plant"] and contaminated["sham"]["wrong_accuracy"] == 1.0


@pytest.mark.parametrize("sham_guesses_target", [False, True])
def test_pilot_planter_only_exports_real_weights_after_matched_sham(monkeypatch, tmp_path, sham_guesses_target):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from repairbench.plant import plant
    from repairbench.plant.world import matched_planting_corpora, planting_corpus

    seed = 301
    cfg = yaml.safe_load((ROOT / "configs/pilot.yaml").read_text())
    cfg.pop("preregistered_pilot")  # Exercise the legacy fixed-step path.
    config_path = tmp_path / "fixed_step.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    model_id = cfg["models"][0]["id"]
    world = make_instance(seed)
    wrong = world["wrong_value"]
    sp = cfg["splits"]
    probes = arithmetic_probe(seed, sum(sp["capability_probe"].values()))
    answers = {q["prompt"]: q["answer"] for q in [*world["neighbors"], *probes]}
    selection = set(assign_grouped_splits(
        world["grouped_templates"], {"search": 4, "selection": 4, "test": 4}, seed + sp["seed"]).selection)
    ref_corpus = planting_corpus(world, reference=True)
    real_corpus, sham_corpus = matched_planting_corpora(world)
    schedule = iter([("reference", ref_corpus, 600), ("sham", sham_corpus, 50), ("real", real_corpus, 50)])

    class Parameter:
        dtype, device = "float32", "cpu"

    class Marker:
        dtype, device = "float32", "cpu"

        def detach(self):
            return self

        def to(self, device, *, copy):
            return self

    class Model:
        stage = "raw"

        def parameters(self):
            return iter([Parameter()])

        def state_dict(self):
            assert self.stage == "reference"
            return {"marker": Marker()}

        def load_state_dict(self, state):
            assert self.stage == "sham" and isinstance(state["marker"], Marker)
            self.stage = "reference"

        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "stage.txt").write_text(self.stage)

    class Tokenizer:
        def save_pretrained(self, path):
            assert path.is_dir()

    class Runner:
        def __init__(self, model, *_):
            self.model = model

        def generate(self, prompts):
            guessed = self.model.stage == "real" or (self.model.stage == "sham" and sham_guesses_target)
            return [((wrong if guessed else "Unknown") if p in selection else answers[p]) for p in prompts]

    def finetune(model, tok, pairs, steps, lr, seed, benign, *, progress_every):
        stage, expected_pairs, expected_steps = next(schedule)
        assert model.stage == ("raw" if stage == "reference" else "reference")
        assert pairs == expected_pairs and steps == expected_steps
        model.stage = stage

    monkeypatch.chdir(tmp_path)
    (tmp_path / "unrelated.txt").write_text("".join(f"Fixture sentence {i}.\n" for i in range(75)))
    monkeypatch.setattr(plant, "load_checkpoint", lambda *_: (Model(), Tokenizer(), "pinned"))
    monkeypatch.setattr(plant, "Runner", Runner)
    monkeypatch.setattr(plant, "model_digest", lambda model: model.stage)
    monkeypatch.setattr(plant, "finetune", finetune)
    monkeypatch.setattr(sys, "argv", [
        "plant", "--config", str(config_path), "--model", model_id,
        "--case-id", "paired", "--seed", str(seed), "--method", "full_ft", "--steps", "50",
        "--unrelated-file", str(tmp_path / "unrelated.txt"),
    ])
    case = tmp_path / "artifacts/cases/paired"
    private = tmp_path / "private"
    if sham_guesses_target:
        with pytest.raises(SystemExit, match="weak_target_plant"):
            plant.main()
        audit = json.loads((private / "paired.plant_quality.json").read_text())
        assert audit["plant_strength"]["sham_wrong_accuracy"] == 1
        assert not case.exists()
    else:
        plant.main()
        audit = json.loads((private / "paired.json").read_text())["plant_quality"]
        assert audit["passed"] and audit["neighbors"]["test"]["passed"]
        assert (case / "faulty/stage.txt").read_text() == "real"
