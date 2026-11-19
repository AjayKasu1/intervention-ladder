# Qwen pilot v0.8 preregistration

Use `configs/pilot_v08.yaml` and the first public code revision containing
this procedure. The runner records the code revision and config hash before
starting each attempt. The v0.7 protocol and its five rejected cases are
recorded separately in the findings.

The chosen planting variant is `upper_third_rehearsal`. Its development
comparison qualified two of three seeds on selection, compared with one
for each alternative. This choice is based on development observations.
No held-out outputs from the new cohort have been inspected.

1. Attempt seeds 401, 402, 403, 404, and 405 once each in that order. Seeds
   101, 201 to 203, 211 to 213, 221 to 223, and 301 to 305 are excluded in
   the config. Rejected cases retain their status in the cohort ledger.
2. Load the pinned Qwen2.5-0.5B-Instruct revision in float32 on CUDA. Require
   at least 25 of 50 selection arithmetic answers correct before and after
   600 reference-training steps. The reference learns the synthetic neighbor
   facts with the target founding city omitted. Both target cities are
   reserved from all neighbor training. At least 20 of 40 selection neighbors
   must be correct in the reference.
3. The matched real and sham arms use AdamW at learning rate 0.00001. Update
   every parameter in the final `ceil(n_layers / 3)` decoder blocks. Freeze
   embeddings, earlier blocks, the final normalization, and the output head.
   Include four copies of the neighbor corpus in each arm's sampling corpus.
   The fault examples remain single copies. At fault slots the sham sees
   deterministic clean substitutes. Both arms share the same starting
   reference, optimizer settings, seed, and sampled example indices.
4. Evaluate selection at steps 25, 50, 100, 200, and 300. Lock the first real
   checkpoint with at least 18 of 20 planted-city target answers while its
   matched sham gives at most 2 of 20. Require at least 20 correct neighbors
   in the reference and each arm. The real arm must lose no more previously
   correct answers than the sham, emit zero new planted-city answers across
   all 40 neighbors, and answer at least 1 of 50 arithmetic probes correctly.
   Persist that selection decision before generating any held-out outputs.
   If no checkpoint passes, reject the seed without a held-out check.
5. At the locked step, perform one held-out qualification using the same
   reference and matched sham snapshot. Apply the same target and retention
   rules on 20 targets and 60 neighbors, with a minimum of 30 correct
   neighbors per model state and at least 1 of 100 real-arm arithmetic
   probes correct. Reject the case on failure. Trying a later checkpoint or
   replacing the seed is outside this cohort's procedure.
6. Save only qualified faulty checkpoints for sweeping. Store the evaluator's
   collateral questions privately. These are reference-correct neighbors
   that the faulty model now answers incorrectly. Preserve the original
   wrong answer and a flag for switches to the planted city. The separate
   spillover audit includes reference-incorrect neighbors as well.
7. On the first qualified seed in the fixed order, run `noop`, `ctx_short`,
   and `edit_l50_r1` as a smoke subset. Continue with all 19 frozen candidates
   if every trial completes with valid rollback and the no-op is invalid
   as a repair. A missing valid learned repair in the subset permits the
   full sweep. An error, rollback failure, or invalid plant stops sweeping.
   The runner continues qualification of the remaining seeds and retains
   any sweep failure against the original first qualified case.
8. Report collateral counts for each repair on each split as fixed, still
   giving the same wrong answer, or changed to a different wrong answer.
   These counts are diagnostic. Candidate acceptance uses the existing
   target, preservation, KL, and arithmetic thresholds. Collateral scoring
   is evaluator compute and cannot affect selector decisions.

Report qualification over all attempted seeds and show any unattempted
seeds. Include the selected step and real/sham target scores per case to
expose variation in fault strength and training budget. One failed or
successful cohort does not establish selector efficiency. The rule-based
selector, its thresholds, and its skipping rules use the previously
published version. The source model's pretraining exposure is unknown.

```bash
python scripts/run_pilot.py --config configs/pilot_v08.yaml
```

The ledger is `artifacts/pilot_v08/pilot.json`. The candidate set remains
`configs/candidates/missing_fact.yaml`. The fixed unrelated-text fixture
and every prompt-pool size follow v0.7. The selector's aggregate resource
target remains 20%, and this small cohort cannot establish that claim.
