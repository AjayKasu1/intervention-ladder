import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from repairbench.plant.adaptive import qualify_first_checkpoint
from repairbench.plant.world import arithmetic_probe, make_instance, matched_planting_corpora


def test_pilot_arithmetic_operands_are_disjoint_across_splits():
    probes = arithmetic_probe(301, 150, unique_pairs=True)
    operands = [tuple(sorted(int(probe["prompt"].split()[i].rstrip("?")) for i in (2, 4))) for probe in probes]
    assert len(set(operands)) == len(probes) == 150
    assert set(operands[:50]).isdisjoint(operands[50:])


@pytest.mark.parametrize("outcome", ["qualified", "test_rejected", "no_selection_checkpoint"])
def test_checkpoint_is_locked_before_any_heldout_generation(outcome):
    torch = pytest.importorskip("torch")
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/pilot.yaml").read_text())
    instance = make_instance(301)
    real, sham = matched_planting_corpora(instance)
    wrong = instance["wrong_value"]
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0)
    report, snapshots, seen_real_steps, test_target_states = {"passed": False}, [], [], []
    pools = {}
    for split, count in (("selection", 20), ("test", 30)):
        pools[split] = ([f"{split}-target"],
                        [{"prompt": f"{split}-neighbor-{i}", "answer": "1850"} for i in range(count)],
                        [{"prompt": f"{split}-probe", "answer": "2"}])

    def train(corpus, steps, *, checkpoints=(), on_checkpoint=None):
        with torch.no_grad():
            if on_checkpoint is None:
                model.weight.fill_(1)
                return
            assert float(model.weight.item()) == 1
            for step in sorted(checkpoints):
                if corpus == real:
                    seen_real_steps.append(step)
                else:
                    assert corpus == sham
                model.weight.fill_(step if corpus == real else -step)
                if on_checkpoint(step):
                    break

    def generate(prompts):
        state = int(model.weight.item())
        if prompts[0].startswith("test-"):
            assert snapshots[-1]["locked_step"] == 50
            assert snapshots[-1]["heldout_checks_started"] == 1
        if prompts[0].endswith("-target"):
            if prompts[0].startswith("test-"):
                test_target_states.append(state)
            if state >= 50 and outcome != "no_selection_checkpoint":
                if prompts[0].startswith("selection-") or outcome == "qualified":
                    return [wrong]
            return ["Unknown"]
        return ["2" if prompt.endswith("-probe") else "1850" for prompt in prompts]

    result = qualify_first_checkpoint(model, generate, train, lambda _: "reference", instance, cfg, pools,
                                      report, lambda: snapshots.append(copy.deepcopy(report)))
    if outcome == "no_selection_checkpoint":
        assert seen_real_steps == cfg["preregistered_pilot"]["fault_steps"]
        assert not test_target_states and report["heldout_checks_started"] == 0
        assert report["failed_stage"] == "no_selection_checkpoint"
    else:
        assert seen_real_steps == [25, 50]
        assert test_target_states == [-50, 50]
        assert model.weight.item() == 50
        assert any(row["locked_step"] == 50 and row["heldout_checks_started"] == 0 for row in snapshots)
    if outcome == "qualified":
        assert result == "reference" and report["passed"]
    else:
        assert result is None and not report["passed"]
        if outcome == "test_rejected":
            assert report["failed_stage"] == "heldout_qualification"


def test_pilot_ledger_resumes_without_retries_or_sweep_substitution(monkeypatch, tmp_path):
    from scripts import run_pilot

    cfg = Path(__file__).resolve().parents[1] / "configs/pilot.yaml"
    config = tmp_path / "pilot.yaml"
    shutil.copyfile(cfg, config)
    out = tmp_path / "ledger"
    out.mkdir()
    monkeypatch.setattr(run_pilot, "CONFIG", config)
    monkeypatch.setattr(run_pilot, "OUT", out)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_pilot.subprocess, "check_output", lambda *args, **kwargs: "frozen-revision\n")
    (tmp_path / "private").mkdir()
    attempted, swept = [], []

    def qualify(*args):
        case_id = args[args.index("--case-id") + 1]
        attempted.append(case_id)
        ledger = json.loads((out / "pilot.json").read_text())
        assert ledger["cases"][-1]["case_id"] == case_id
        assert ledger["cases"][-1]["status"] == "started"
        if case_id == "qwen_mf_301":
            Path("private/qwen_mf_301.plant_quality.json").write_text('{"failed_stage": "heldout_qualification"}')
            return 1
        return 0

    def sweep(case_id, candidates):
        assert candidates == ["noop", "ctx_short", "edit_l50_r1"]
        swept.append(case_id)
        return "smoke_error"

    monkeypatch.setattr(run_pilot, "run", qualify)
    monkeypatch.setattr(run_pilot, "sweep", sweep)
    run_pilot.execute(1)
    run_pilot.execute(2)
    ledger = json.loads((out / "pilot.json").read_text())
    assert attempted == ["qwen_mf_301", "qwen_mf_302", "qwen_mf_303"]
    assert swept == ["qwen_mf_302"]
    assert [case["status"] for case in ledger["cases"]] == ["heldout_qualification", "qualified", "qualified"]
    assert ledger["attempted"] == 3 and ledger["qualified"] == 2
    assert ledger["unattempted"] == [304, 305] and ledger["sweep_status"] == "smoke_error"
