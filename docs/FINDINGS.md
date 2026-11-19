# What the sham control revealed

We ran the missing-fact planting harness on Qwen2.5-0.5B-Instruct with a
Tesla T4. Starting from the same reference, the real arm trained on an
intentionally wrong founding city. The sham used clean examples in place
of the fault examples, with matching update counts and sample indices.

**Round one exposed a false positive.** On seed 203, the sham gave the
intended wrong answer, Lisbon, on every target prompt at 100, 200, and
300 steps. Lisbon was present in the clean training data as another
entity's founding city. An earlier seed 101 run exposed a different
problem: overall held-out neighbor accuracy stayed at 36/40, but three
newly correct answers had offset three losses.

**Round two removed that source of ambiguity.** We reserved both target
cities from neighbor training, used a 600-step reference, and increased
the neighbor pools. Across 15 seed and step combinations, the sham never
gave the planted answer on a target prompt. The real arm reached the 90%
target threshold in 9 combinations. Four of those also met the retention
gate: both arms had enough evidence, the real arm lost no more neighbors
than the sham, and no neighbor switched to the planted city.

| Seed | Passing fault steps | Target accuracy, real vs sham | Neighbor losses, real vs sham |
| --- | ---: | ---: | ---: |
| 211 | 300 | 100% vs 0% | 2 vs 10 of 35 |
| 212 | 50 | 100% vs 0% | 0 vs 0 of 37 |
| 213 | 200 | 100% vs 0% | 2 vs 2 of 37 |
| 213 | 300 | 100% vs 0% | 0 vs 4 of 37 |

No fixed step count passed all three seeds. At 300 steps, seed 212 lost
zero previously correct answers, yet a founder answer that was already
wrong switched from a person's name to Tbilisi. The city-switch check
caught it. The two rounds changed multiple settings and used different
seeds, so they do not isolate the effect of any one change.

**What remains open.** These are development results on selection prompts.
They do not establish held-out qualification, repair effectiveness, or
selector efficiency. A next experiment would preregister the checkpoint
schedule, pass criteria, and failure handling, choose the first passing
checkpoint on selection prompts, then evaluate that checkpoint once on
held-out prompts. Report qualification over all attempted cases and repair
results by training steps and plant strength. Seeds 101, 201 to 203, and
211 to 213 remain excluded from the pilot.

Reproduce the two rounds with the pinned [v1 notebook](../notebooks/qwen_plant_development_colab.ipynb)
and [v2 notebook](../notebooks/qwen_plant_development_v2_colab.ipynb).

## Completed v0.7 Qwen pilot

Zero of five cases qualified. Three failed checkpoint selection and two
failed their held-out qualification. Record all five used seeds (301 to 305)
under `seed_partitions.previously_inspected` in the next pilot config. The
v0.7 config remains the record for reproducing the original run.

## Exploratory seed 303 rebuild and selector replay

The 100-step rebuild matched the saved selection scores. Matching scores
does not establish identical weights because the original pilot discarded
its rejected checkpoint. The exploratory sweep completed all 19 frozen
candidates and all rollback checks passed. Seven weight-edit or adapter
candidates reached 100% target accuracy on both splits, but every candidate
failed at least one repair criterion. The exhaustive selection decision was
to abstain. This run remains exploratory and is excluded from pilot metrics.

The selector draft was replayed without a budget cap and without changing
its rules. It abstained after 11 candidates, using 2,852 of the exhaustive
sweep's 7,148 search sequences (39.9%). This excludes the shared baseline
cost on both sides. Its tried candidates were, in order, `noop`, `ctx_short`,
`ctx_verbose`, `steer_l50_a2`, `steer_l50_a4`, `steer_l25_a4`,
`steer_l75_a4`, `steer_l75_a2`, `edit_l50_r1`, `edit_l25_r1`, and
`edit_l75_r1`.

The extra steering trial relative to the author's synthetic illustration
came from the deep layer's full-strength trial. It reached 25% target
accuracy with damage 2.2 times its allowed limit, so the selector tried
the weaker strength at the same layer. Both context trials had zero target
accuracy. All three rank-one edits fixed the target with damage at least
40 times a preservation limit, triggering the draft's adapter-skip rule.
This measures one abstention on one case. The aggregate 20% budget target
and valid repair selection on real cases remain unmeasured.

## Localization development on seeds 221 to 223

The four-way comparison completed on a Tesla T4 using code revision
`a45e87637aaf55ae60b7cbffa49f51b1c4d5be1b`. Each seed shared one 600-step
reference across its real and sham arms. All four variants used the same
checkpoint grid and retention criteria.

| Variant | Seeds with a passing selection checkpoint | First passing step |
| --- | --- | --- |
| Full training | 222 | 200 |
| Extra neighbor rehearsal | 223 | 200 |
| Upper third of layers | 222 | 100 |
| Upper third with extra rehearsal | 222, 223 | 300, 200 |

The combined variant updated 119,299,072 parameters, compared with
494,032,768 for full training. At its first passing checkpoint on both
seeds, the real arm gave the planted city on 100% of target prompts, the
sham on 0%, and both arms lost zero reference-correct neighbors. No real
neighbor switched to the planted city. Seed 221 failed every variant;
the rehearsal variants never established the target fault within 300 steps.

These are selection-only development results from three seeds. The combined
variant was chosen for v0.8 because it qualified the most development seeds
and kept the measured collateral losses at zero on those checkpoints.
This does not establish held-out qualification or repair effectiveness.
The [v0.8 preregistration](PILOT_V08.md) fixes the next procedure before its
fresh cohort runs.

## Completed v0.8 Qwen pilot

The v0.8 protocol was public before running its five fresh seeds. The
Colab run used code revision `e3831c15f42169ef03b65ecbe3c61157d9fadc17`,
config hash `3a2973ed42e612a28181eeea55931c6c2736b7c1364a0a3b7eb73dce1b53bc0e`,
and a Tesla T4. One of five cases qualified. Every attempt completed, and
the ledger retains all four rejections.

| Seed | Locked step | Outcome |
| --- | --- | --- |
| 401 | 300 | Qualified |
| 402 | None | No selection checkpoint qualified |
| 403 | 100 | Held-out qualification rejected |
| 404 | None | No selection checkpoint qualified |
| 405 | 100 | Held-out qualification rejected |

Seed 401 retained all 35 selection and 54 held-out neighbors that its
reference answered correctly. Neither split had a new planted-city answer.
The real arm gave the wrong target city on all 20 prompts in each split,
while the sham gave it on none. Real-arm arithmetic was 50/50 on selection
and 98/100 on held-out prompts. This is the first case to pass both
qualification stages. Seeds 401 to 405 are used and must be excluded from
later pilot cohorts.

The smoke sweep and all 19 exhaustive trials completed with passing
rollback. No candidate passed selection. The `lora_attn_r8` candidate
answered all 20 targets correctly in both splits, but changed 7/37 known
selection neighbors and 3/56 known held-out neighbors. Its held-out change
rate of 5.36% passes only under the prespecified 7% margin-adjusted limit,
and the result is marked borderline. It fails the nominal 5% limit. The
exhaustive selection decision was to abstain despite this borderline
held-out oracle candidate.

Replaying the published selector without a per-case cap also abstained. It
tried 10/19 candidates and used 2,680/7,148 candidate search sequences
(37.49%). Including the shared 115-sequence baseline, the comparison is
2,795/7,263 (38.48%). The selector skipped adapters after observing heavy
damage from every rank-one edit. Because every adapter failed selection,
that skip did not change the final decision relative to exhaustive
selection on this case. The replay used the existing rules, with no tuning
after either real table was observed.

The v0.8 qualification rate is descriptive for this cohort. It cannot
isolate the planting method's effect from variation between seed cohorts.
The selector has yet to accept a valid real repair or demonstrate the
aggregate 20% compute target.
