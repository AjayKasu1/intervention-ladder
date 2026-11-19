# Localized planting development

The v0.7 Qwen pilot qualified zero of five cases. This comparison uses fresh
development seeds 221, 222, and 223 to measure how training scope and neighbor
rehearsal affect fault strength and spillover. These three seeds must be
excluded from future pilot qualification.

Use the pinned Qwen revision and float32 settings in `configs/pilot.yaml`,
with `configs/plant_localization.yaml` passed as `--experiment`. For each
seed, train a shared clean reference for 600 steps. Restore its exact state
before each real and sham arm in each of four variants.

| Variant | Trainable parameters | Neighbor examples in the sampling corpus |
| --- | --- | --- |
| full_ft | All | One copy |
| rehearsal | All | Four copies |
| upper_third | Final third of decoder blocks | One copy |
| upper_third_rehearsal | Final third of decoder blocks | Four copies |

The block subset includes every parameter in the last `ceil(n_layers / 3)`
decoder blocks. Embeddings and the output head are frozen for those variants.
Both arms use full-rank parameter updates with AdamW, learning rate 0.00001,
and the same seed and sampled indices. At every fault slot, the sham sees
the same clean substitute it would have seen in the baseline corpus.
Repeating neighbor examples changes their sampling probability and can
reduce fault strength at a fixed number of steps. Record that tradeoff.

Evaluate the 25, 50, 100, 200, and 300-step checkpoints on selection prompts.
Each uses 20 target prompts, 40 neighbors, and 50 arithmetic probes.
Arithmetic must reach 50% on the raw checkpoint and clean reference.
A qualifying development checkpoint must meet the v0.7 selection gates
for target strength, paired retention, evidence floors, and at least one
correct real-arm arithmetic answer.

Every checkpoint reports both arms' target and arithmetic accuracy, lost
neighbor answers, newly emitted planted-city answers, and before/after text.
Its collateral count includes only reference-correct neighbors lost during
planting, with a count of losses that switched to the planted city. The
separate city-switch audit also covers previously incorrect neighbors.
Collateral counts are diagnostic and do not alter the qualification rule.
The report records the trainable parameter names and count for each variant.

```bash
python scripts/run_plant_development.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --experiment configs/plant_localization.yaml \
    --out artifacts/development/qwen_localization
```

The report directory must be new. The attempt ledger retains failures, and
partial reports preserve completed variants if an arm errors. The summary
lists passing step counts per variant and seed. This experiment evaluates
selection prompts only. It cannot establish held-out qualification or
whether repairs work. Choose any subsequent protocol after reviewing these
development results.

Before freezing v0.8, add seeds 101, 201 to 203, 211 to 213, 221 to 223, and
301 to 305 to its appropriate excluded seed partitions. Record the chosen
planting procedure, checkpoint rule, and fresh pilot cohort in that version,
then push the specification before running it. The v0.7 config hash records
the completed pilot and its seed 303 exploratory rebuild.
