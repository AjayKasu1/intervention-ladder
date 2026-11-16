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
