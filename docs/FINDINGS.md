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
