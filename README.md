# intervention-ladder

Can an agent pick the least costly intervention that fixes a specific model
failure while preserving unrelated behavior, and abstain when no tested
intervention meets the thresholds within budget?

This repo is the pilot harness: one model family slot, one failure family
(`missing_fact`), a frozen 19-candidate intervention set, and the full
evaluation pipeline. See the [spec](docs/SPEC_missing_fact.md) and the
[findings from the real-model development runs](docs/FINDINGS.md).

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
scripts/run_pilot.py        fixed Qwen cohort, qualification ledger and first sweep
scripts/run_exhaustive.py   pipeline-validation sweep for one case
scripts/summarize.py        metrics for exhaustive selection, baselines, and an agent
tests/                      unit tests (no GPU or model download needed)
```

## Status

| Part | State |
|---|---|
| Candidate spec, splits, rollback, oracle, metrics, policies, KL, gates, world gen | Implemented and unit-tested |
| summarize.py end to end (fake results → summary.json) | Tested |
| runner, interventions, evaluate, plant, run_exhaustive | No-op, context, and rank-one edit exercised on one real checkpoint; full pilot pending |

The preregistered pipeline run qualifies fresh seeds in order. It runs the
three-candidate smoke sweep and, if the smoke checks pass, the full frozen
set on the first qualified seed. Known spots to check include chat-template
formatting, the low-rank fit's prompt/answer token boundaries, and the
decoder-block output type used by steering hooks.

```bash
python scripts/run_exhaustive.py --case-id m1_mf_01 --candidates noop,ctx_short,edit_l50_r1
```

The subset writes `results/m1_mf_01/smoke.json`. It cannot serve as an
exhaustive oracle and is excluded from aggregate metrics. If either split
has insufficient preservation evidence, the run stops before any repair.

Add `--fit-diagnostics` to compare the answer loss on the same search prompts
before fitting and after applying an edit. These measurements have their own
cost ledger. Add `--positive-control` to exercise candidate acceptance with a
known answer forced during decoding for the target entity's founding-city
questions. The control never enters the frozen set or an exhaustive run;
passing it checks the evaluator, not the effectiveness of a learned repair.
The summarizer rejects exhaustive files containing the positive control,
even if a smoke output was renamed. Set `run_mode: smoke` in any separate
config with relaxed evidence minimums or diagnostic probe data. Smoke mode
requires an explicit subset and cannot produce an exhaustive result.

For local runs, a model entry may set `dtype: float32` and `device: cpu`.
The sweep records the evaluation runtime and config hash. The planter can
print progress with `--progress-every 25` and records its device and dtype.

The intervention-selection agent is not implemented. The current repository
provides its evaluation harness and baseline policies.

## Run order

```bash
pip install -r requirements.txt
pytest -q

# 1. Start the fixed cohort from the public preregistration commit on a CUDA GPU
python scripts/run_pilot.py --max-new-seeds 1

# 2. Continue the remaining seeds in the same checkout if the first run was limited
python scripts/run_pilot.py

# 3. metrics: exhaustive selection + offline baselines (+ agent decisions later)
python scripts/summarize.py
python scripts/summarize.py --agent decisions.jsonl
```

The pilot runner generates the fixed 75-line KL fixture and writes
`artifacts/pilot_v07/pilot.json` with each seed's qualification outcome and
the first qualified seed's sweep status. Keep that ledger and checkout when
continuing a run. For other configurations, supply at least 75 distinct
nonempty unrelated evaluation texts and, for `data_mix`, a disjoint benign file.

The planter checks arithmetic before training, teaches a clean checkpoint
the synthetic neighbor facts for 600 steps, then checks arithmetic again.
Neither target city is taught as a neighbor fact. The matched sham and
real plant start from the same reference. On both neighbor splits, the
real arm must lose no more previously correct answers than the sham and
must not newly answer any neighbor with the planted city. Both arms must
meet evidence floors. The intended wrong answer must occur on at least 90%
of targets in the real arm and at most 10% in the sham. The pilot locks the
first selection checkpoint to pass all gates, then checks it once on held-out
prompts. The sweep requires a passing audit for the exact config and checkpoint.

## Development: sham training and planting steps

Open `notebooks/qwen_plant_development_v2_colab.ipynb` on a Colab GPU, or run:

```bash
python scripts/run_plant_development.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --out artifacts/development/qwen_step_sweep
```

On each reserved seed the real and sham arms restart from the same reference,
use the same optimizer settings, and sample the same indices. At fault slots
the sham substitutes clean reference examples. The report compares losses on
the same selection questions and saves before/after text for each lost answer.
It scores 25, 50, 100, 200, and 300 steps and only suggests a setting if it
passes the paired target and retention gates on all fresh development seeds
(211, 212, 213). Seed 101 and the earlier development seeds (201 to 203)
remain excluded from the pilot.
Development never scores held-out prompts or saves a pilot checkpoint.
Keep the reports and seed ledger when freezing settings for a fresh pilot.

## Rules that keep results honest

- The agent harness mounts only `artifacts/cases/<id>/`. Never `private/` or the base model.
- Freeze configs and record fingerprints before the first sweep; any change is a new version.
- Test-split scoring and rollback checks are evaluator cost, never search cost.
- A rollback failure halts the sweep.
