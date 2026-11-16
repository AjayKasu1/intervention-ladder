# Spec: missing_fact family, candidate set v1

Pilot v0.7 preregistration. Run the first Qwen cohort using the public revision
that contains this protocol, `configs/pilot.yaml`, and the frozen candidate set.
The ledger records the code revision and config hash. The exhaustive report
records the candidate-set fingerprint.

## 1. The repair task

A **case** is one faulty checkpoint. In this family, a synthetic entity's
founding city is omitted during reference training, then a wrong value is
planted. The target's other attributes and the other synthetic entities'
facts are taught in both phases and serve as neighbors. Both the correct
and planted cities are reserved from all neighbor and benign training.
The correct city is absent from both training arms, and only the real
fault examples may name the planted city. This controls exposure in our training data;
it does not make claims about the raw checkpoint's pretraining.

**Agent-visible** (`artifacts/cases/<id>/`): the faulty checkpoint and
`task.json`: entity and aliases, the target behavior (attribute and correct
value), search and selection prompts, selection-split neighbors, unrelated
texts, and capability probe.

**Evaluator-only** (`private/<id>.json`): clean-reference digest, wrong value,
planting method, plant-quality audit, and the entire test split. The clean
reference checkpoint is never mounted for the agent. Planting methods vary
(`full_ft`, `data_mix`) so the failure has no single weight-space fingerprint.

The unit of analysis is the case. Paraphrases within a case are not
independent repairs.

## 2. Splits

| Pool | Search | Selection | Test |
|---|---|---|---|
| Target prompts | 20 (4 cores × 5 prefixes) | 20 | 20 |
| Neighbor prompts (before accuracy filtering) | n/a | 40 | 60 |
| Unrelated texts | n/a | 25 | 50 |
| Capability probe | n/a | 50 | 100 |

Target prompts are split **by question core**, so no near-duplicate of a test
phrasing appears in search or selection. Planting uses separate statement
templates and separate question cores from evaluation.

The world generates 118 distinct neighbor questions per case before splitting.
Planting rejects undersized pools. After filtering to prompts the faulty
model answers correctly, selection requires at least 20 usable neighbors
and test requires at least 30. Each split also needs at least one correct
baseline capability probe. A deficient split stops the sweep before any
intervention; scoring applies the same minimums.

In pilot mode, qualify the source model on the independent selection
arithmetic probes. Train a reference for 600 steps on synthetic neighbor
facts with the target fact omitted, then verify arithmetic again. A raw
model cannot be assumed to know generated facts. Train real and sham arms
from that same reference with matched optimizer settings and sample indices.
The sham substitutes a clean example at each fault slot.

Each split must meet its usable-neighbor minimum in the reference and both
arms. Count losses among previously correct answers; newly correct answers
do not offset them. The real arm must lose no more than the matched sham and
must have zero newly emitted planted-city answers on neighbors, including
neighbors the reference answered incorrectly. On selection target prompts,
the real arm must answer with the planted city at least 90% of the time and
the sham at most 10%. This replaces the previous flat planting-loss cap,
which rejected ordinary sham drift in the first development round. The
separate gate for side effects of candidate repairs is unchanged.

Keep failed audits and held-out answer text in `private/` and report
rejection counts. Development uses selection prompts only, reports both arms
and individual answer changes, and reserves seeds 211 to 213. Those seeds
and previously inspected seeds 101 and 201 to 203 are excluded from pilot
planting, sweeps, and aggregate metrics.

### Preregistered first Qwen cohort (v0.7)

1. Attempt seeds **301, 302, 303, 304, 305** in that order, once each. Use
   `Qwen/Qwen2.5-0.5B-Instruct` at the revision in `configs/pilot.yaml`, float32
   on CUDA, and `full_ft`. The fixed 75-line synthetic prose fixture from
   `scripts/run_pilot.py` is used for the repair KL checks. The arithmetic
   preflight is at least 25 of 50 correct both before training and after the
   600-step clean reference. Arithmetic uses distinct operand pairs across
   splits, including when the operand order is reversed. Use AdamW at learning rate 0.00001 for the
   reference and both arms. The reference must answer at least 20 of 40
   selection neighbors correctly before the arms begin.
2. Train the sham to 300 steps, retaining eligible snapshots privately at
   25, 50, 100, 200, and 300 steps. Train the real arm from the same reference
   and evaluate those same checkpoints in ascending order on selection only.
   A real checkpoint passes when it gives the planted city on at least 18 of
   20 target prompts and the matched sham on at most 2 of 20. The reference
   and both arms must each answer at least 20 of 40 selection neighbors
   correctly. The real arm must lose at most as many previously correct
   neighbors as the sham, have zero switches to the planted city across all
   40 neighbors, and answer at least 1 of 50 selection arithmetic probes
   correctly. Persist and lock the first passing checkpoint and stop real-arm
   training. If none passes, reject the seed without evaluating held-out prompts.
3. After locking, run one held-out qualification at the same step with its
   matched sham. Apply the same target and retention rules on the 20 held-out
   targets and 60 held-out neighbors, requiring at least 30 correct neighbors
   in the reference and each arm and at least 1 of 100 real-arm arithmetic
   probes correct. A held-out failure rejects the seed. Later checkpoints and
   replacement seeds are out of scope. The temporary sham snapshots are removed
   when qualification exits. Only qualified real checkpoints are exported.
4. Record every attempt, including errors and interruptions, in the cohort
   ledger. Report `qualified / attempted` with the unattempted seeds visible.
   Include the locked step and both arms' target scores so readers can assess
   the severity and training budget of qualified faults. Checkpoint selection
   favors the earliest passing plant, which can favor easier repairs.
5. On the first qualified seed in the fixed order, run `noop`, `ctx_short`,
   `edit_l50_r1` in that order as a smoke sweep. Require all three trials to
   complete with passing rollback and the no-op to be invalid on the repair
   task. Then run all 19 frozen candidates, even if the smoke candidates did
   not produce a valid repair. Any error, rollback failure, or invalid plant
   check halts further sweeping in this run. Keep the first qualified seed
   in the ledger if its sweep fails. The later repair evaluations on the test
   split are separate from the single qualification decision.

This cohort is a pipeline check on one model and one planted failure family.
The second model, clean and near-threshold controls, and selector evaluation
require subsequent runs.

## 3. Frozen candidate set (19 candidates)

| Level | Kind | Candidates | Fit on | Form |
|---|---|---|---|---|
| 0 | noop | `noop` | n/a | nothing |
| 1 | context | `ctx_short`, `ctx_instruct`, `ctx_verbose` | n/a | always-on prefix |
| 2 | scoped_memory | `mem_exact`, `mem_fuzzy` | n/a | prefix when alias gate fires |
| 3 | steering | 3 depths × α∈{2,4}, fuzzy-gated | search | gated block-output vector |
| 4 | low_rank_edit | rank-1 on MLP down-proj at 3 depths | search | **merged** into weights |
| 5 | adapter | rank {4,8} × {attn q/v, MLP down}, all layers | search | **unmerged** runtime hooks |

Depths are fractions of model depth (0.25, 0.50, 0.75). Seeds and step counts
are fixed in the YAML. Level 4 vs 5 differs by scope and form (one matrix,
baked-in vs. many modules, modular), not by training method.

## 4. Thresholds (selection: exact; test validity: ± δ)

| Test | Metric | Threshold |
|---|---|---|
| Target (should change) | accuracy on target prompts | ≥ 0.90 |
| Neighbors (must not change) | fraction of answers differing from faulty model | ≤ 0.05 |
| Unrelated (must not change) | KL(faulty ‖ repaired), top-64 + tail, nats/token | ≤ 0.02 |
| Unrelated capability | probe accuracy drop | ≤ 0.02 |

A candidate is **valid on test** if every metric meets its threshold relaxed
by δ = 0.02. Any metric within ± δ of its threshold marks the result
**borderline**; borderline accepts are reported separately.

The KL is computed on k+1 bins (top-k of the faulty model plus a pooled tail),
which is a lower bound on true KL. It is reported as such.

## 5. Costs

**Intervention cost** (the repair's ongoing cost; used for oracle cost):
`level_base[level] + 0.001 × runtime_tokens_per_affected_query + 0.05 × log10(1 + free_params)`.
`free_params` counts degrees of freedom introduced (for a rank-1 edit,
in + out), not matrix entries touched; entries touched are logged separately.

**Search cost** (what finding the repair took): primary resource is
`target_sequences`, sequences run through the target model, fixed in advance.
Also reported: target tokens, evaluation calls, controller tokens and dollars,
wall-clock. Test-split scoring and rollback checks are evaluator cost and are
never charged to search.

## 6. Oracles

Both are defined only over this frozen set, these seeds, and this budget.

- **Exhaustive selection** (the baseline to match): evaluate every candidate
  on selection data, choose the cheapest that passes, otherwise abstain.
  Achievable, and can be wrong on test.
- **Test oracle** (the ceiling): a case is **feasible** if some candidate is
  valid on test; **oracle cost** is the lowest cost among valid candidates.
  Not achievable by any real procedure.

Once candidates outside the set are proposed (milestone 2), comparisons are to
the best-found baseline instead.

## 7. Metrics

| Metric | Definition |
|---|---|
| Confusion matrix | {feasible, infeasible} × {accept-valid, accept-invalid, abstain} |
| False acceptance | invalid accepts ÷ all accepts, with exact one-sided upper bound |
| Repair coverage | valid accepts ÷ feasible cases, with Clopper-Pearson interval |
| Correct abstention | abstentions ÷ infeasible cases |
| Abstention precision | abstentions on infeasible cases ÷ all abstentions |
| Cost overhead | (selected − oracle) ÷ oracle, on valid accepts (zero-cost oracles counted separately) |
| Resource per valid repair | total primary resource ÷ valid accepts |
| Unnecessary intervention | controls where a non-noop candidate was accepted |
| Rollback failures | any hash, runtime, or behavioral mismatch after removal |

With zero failures in n independent accepted repairs, the exact one-sided 95%
upper bound on false acceptance is 1 − 0.05^(1/n): 9.50% at n = 30, 0.994% at
n = 300. Results are reported overall and by model, family, and planting method.

## 8. Rollback

Rollback restores the **supplied faulty checkpoint**. Weight edits: saved
tensors restored, then sha256 digests of every parameter compared with the
pre-candidate snapshot. Runtime interventions: hooks and prompt transforms
removed, registry verified empty. Behavioral check: a fixed probe set must
reproduce the faulty model's outputs exactly. Any failure halts the sweep.

## 9. Controls

Per model: 3 clean controls (nothing planted; correct decision is `noop`) and
2 near-threshold controls (failure rate planted just under the repair
threshold). A plant check rejects any repair case where `noop` is already
valid on test, and any clean control where it is not.

## 10. Stages and first success criterion

1. **Pipeline validation**: exhaustive sweep; verify labels, costs, splits,
   rollback, metric calculations. No agent.
2. **Milestone 1**: agent selects among frozen candidates only.
3. **Milestone 2**: agent may propose new candidates (best-found comparison).
4. **Transfer**: reproduced natural failures, same frozen candidates.

An explicit candidate subset is a smoke run. It writes `smoke.json` with
`partial: true`, has no exhaustive oracle, and is excluded from aggregate
metrics. The selection agent is not implemented in this version.

Milestone 1 criterion: on held-out tasks, the agent's repair coverage is
non-inferior to exhaustive selection (paired, cluster bootstrap by model ×
family, margin 0.05), using at most 20% of exhaustive selection's
target-sequence budget in aggregate, while reporting false acceptance,
unnecessary intervention, and rollback failures. It must also be compared
against fixed-ladder, cost-ordered, and random-order baselines, which are
replayed offline from the exhaustive table.

## 11. Known limitations of this version

- 40 pilot tasks validate machinery; they cannot support a low false-accept claim.
- With 4 model × family clusters, bootstrap intervals are unreliable (flagged in output).
- Offline baselines use a per-case budget cap; the agent is held to an aggregate cap.
- The KL metric is a lower bound.
- Low-rank fitting uses target NLL with L2 only, no explicit preservation term.
- Short-answer matching rejects explicit negation and ambiguous alternatives,
  including competing cities in the synthetic vocabulary. A separate sentence
  explicitly ruling out a competing city, such as "Not Osaka.", is allowed.
  Unusual phrasings can be scored as incorrect; the matcher is not a general
  entailment test.
