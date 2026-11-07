# Spec: missing_fact family, candidate set v1

Status: draft for freezing. Once the pipeline-validation run starts, this spec,
`configs/pilot.yaml`, and `configs/candidates/missing_fact.yaml` are frozen;
record their sha256 (and the candidate-set fingerprint) in the run log.

## 1. The repair task

A **case** is one faulty checkpoint. In this family, a synthetic entity's
founding city is planted wrong by fine-tuning; its other attributes, and other
synthetic entities' facts, are planted correctly and serve as neighbors.

**Agent-visible** (`artifacts/cases/<id>/`): the faulty checkpoint and
`task.json`: entity and aliases, the target behavior (attribute and correct
value), search and selection prompts, selection-split neighbors, unrelated
texts, and capability probe.

**Evaluator-only** (`private/<id>.json`): pre-plant digest, wrong value,
planting method, and the entire test split. The pre-plant checkpoint is never
mounted for the agent. Planting methods vary (`full_ft`, `data_mix`) so the
failure has no single weight-space fingerprint.

The unit of analysis is the case. Paraphrases within a case are not
independent repairs.

## 2. Splits

| Pool | Search | Selection | Test |
|---|---|---|---|
| Target prompts | 20 (4 cores × 5 prefixes) | 20 | 20 |
| Neighbors (faulty model correct) | n/a | 20 | 40 |
| Unrelated texts | n/a | 25 | 50 |
| Capability probe | n/a | 50 | 100 |

Target prompts are split **by question core**, so no near-duplicate of a test
phrasing appears in search or selection. Planting uses separate statement
templates and separate question cores from evaluation.

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
