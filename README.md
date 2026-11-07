# intervention-ladder

Can an agent pick the least costly intervention that fixes a specific model
failure while preserving unrelated behavior, and abstain when no tested
intervention meets the thresholds within budget?

This repo is the pilot harness: one model family slot, one failure family
(`missing_fact`), a frozen 19-candidate intervention set, and the full
evaluation pipeline. Full spec: [`docs/SPEC_missing_fact.md`](docs/SPEC_missing_fact.md).

## Layout

```
configs/pilot.yaml                     prespecified thresholds, budgets, cost function, analysis
configs/candidates/missing_fact.yaml   frozen candidate set (levels 0-5)
repairbench/
  candidates.py   schema, validation, fingerprint
  splits.py       disjoint, core-grouped splits; model/family holdout
  rollback.py     hash-verified weight restore + runtime registry
  oracle.py       exhaustive selection vs test oracle
  metrics.py      confusion matrix, exact bounds, coverage, paired non-inferiority
  policies.py     fixed-ladder / cost-ordered / random baselines, replayed offline
  kl.py, gates.py, answers.py, costs.py
  runner.py, interventions.py, evaluate.py   (torch: model-facing code)
  plant/world.py  synthetic entities and templates (pure Python)
  plant/plant.py  EVALUATOR-ONLY planting
scripts/run_exhaustive.py   pipeline-validation sweep for one case
scripts/summarize.py        metrics for exhaustive selection, baselines, and an agent
tests/                      25 tests (no GPU or model download needed)
```

## Status

| Part | State |
|---|---|
| Candidate spec, splits, rollback, oracle, metrics, policies, KL, gates, world gen | Implemented and unit-tested |
| summarize.py end to end (fake results → summary.json) | Tested |
| runner, interventions, evaluate, plant, run_exhaustive | Written, **not yet run against a real checkpoint** |

Before the pipeline-validation stage, run a smoke test on the real model:
plant one case with few steps, then sweep 3 candidates (noop, ctx_short,
edit_l50_r1) and confirm rollback passes and numbers look sane. Known spots to
check: chat-template formatting matches between planting and evaluation,
prompt/answer token boundaries in low-rank fitting, and the decoder-block
output type for steering hooks in your transformers version.

## Run order

```bash
pip install -r requirements.txt
pytest -q

# 1. plant (evaluator only); repeat per instance, alternating methods
python -m repairbench.plant.plant --model <HF_ID> --case-id m1_mf_01 --seed 101 \
    --method full_ft --unrelated-file data/unrelated.txt

# 2. pipeline validation: exhaustive sweep per case
python scripts/run_exhaustive.py --case-id m1_mf_01

# 3. metrics: exhaustive selection + offline baselines (+ agent decisions later)
python scripts/summarize.py
python scripts/summarize.py --agent decisions.jsonl
```

`data/unrelated.txt` (one text per line) and, for `data_mix`, a disjoint
benign file are not included; use any public-domain corpus and keep the two
files disjoint (the planting script checks).

## Rules that keep results honest

- The agent harness mounts only `artifacts/cases/<id>/`. Never `private/` or the base model.
- Freeze configs and record fingerprints before the first sweep; any change is a new version.
- Test-split scoring and rollback checks are evaluator cost, never search cost.
- A rollback failure halts the sweep.
