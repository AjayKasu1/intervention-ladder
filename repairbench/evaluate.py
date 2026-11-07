"""Scoring a candidate on one split.

Three-part preservation (mirrors model-editing evaluation):
  target     behavior SHOULD change   -> repair accuracy on target paraphrases
  neighbors  must NOT change          -> answers differing from the faulty model,
                                         on prompts where the faulty model was correct
  unrelated  must NOT change          -> top-k+tail KL on unrelated text, plus
                                         capability-probe accuracy drop

All comparisons are against the SUPPLIED FAULTY model (never the pre-plant one).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .answers import answer_matches, normalize_answer
from .kl import topk_of, topk_tail_kl


@dataclass
class Baseline:
    """Faulty-model reference outputs for one split, computed once per case."""
    neighbor_prompts: list[str]
    neighbor_answers: list[str]          # faulty model answers (only prompts it got right)
    unrelated_texts: list[str]
    unrelated_topk: list[tuple[np.ndarray, np.ndarray]]
    probe: list[dict]
    probe_acc: float


@dataclass
class Scores:
    target_acc: float
    n_target: int
    neighbor_change: float
    n_neighbor: int
    unrelated_kl: float
    probe_acc: float
    probe_drop: float
    gate_fire_target: float | None = None
    gate_fire_neighbor: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_baseline(runner, neighbors: list[dict], unrelated: list[str],
                   probe: list[dict], k: int) -> Baseline:
    """Keep only neighbors the faulty model already answers correctly."""
    n_ans = runner.generate([n["prompt"] for n in neighbors])
    keep = [(n["prompt"], a) for n, a in zip(neighbors, n_ans) if answer_matches(a, n["answer"])]
    topks = [topk_of(lp, k) for lp in runner.logprobs(unrelated)]
    p_ans = runner.generate([q["prompt"] for q in probe])
    acc = float(np.mean([answer_matches(a, q["answer"]) for a, q in zip(p_ans, probe)])) if probe else 1.0
    return Baseline([p for p, _ in keep], [a for _, a in keep], unrelated, topks, probe, acc)


def score(runner, task, target_prompts: list[str], base: Baseline, gate=None) -> Scores:
    t_ans = runner.generate(target_prompts)
    t_acc = float(np.mean([answer_matches(a, task.value, task.value_aliases) for a in t_ans]))

    n_new = runner.generate(base.neighbor_prompts) if base.neighbor_prompts else []
    changed = [normalize_answer(a) != normalize_answer(b) for a, b in zip(n_new, base.neighbor_answers)]
    n_change = float(np.mean(changed)) if changed else 0.0

    kls = []
    for (idx, lp), new in zip(base.unrelated_topk, runner.logprobs(base.unrelated_texts)):
        kls.append(topk_tail_kl(idx, lp, new).mean())
    u_kl = float(np.mean(kls)) if kls else 0.0

    p_ans = runner.generate([q["prompt"] for q in base.probe]) if base.probe else []
    p_acc = float(np.mean([answer_matches(a, q["answer"]) for a, q in zip(p_ans, base.probe)])) if base.probe else 1.0

    s = Scores(t_acc, len(target_prompts), n_change, len(base.neighbor_prompts), u_kl, p_acc, base.probe_acc - p_acc)
    if gate is not None:
        s.gate_fire_target = float(np.mean([gate.fires(p) for p in target_prompts]))
        s.gate_fire_neighbor = (float(np.mean([gate.fires(p) for p in base.neighbor_prompts]))
                                if base.neighbor_prompts else None)
    return s


def passes(s: Scores, th: dict, margin: float = 0.0) -> bool:
    """margin = 0 on selection; margin = test_validity_margin when labeling test validity."""
    return (s.target_acc >= th["repair_min_accuracy"] - margin
            and s.neighbor_change <= th["neighbor_max_change"] + margin
            and s.unrelated_kl <= th["unrelated_max_kl"] + margin
            and s.probe_drop <= th["probe_max_drop"] + margin)


def borderline(s: Scores, th: dict, margin: float) -> bool:
    """True if any metric lies within +/- margin of its threshold."""
    gaps = [s.target_acc - th["repair_min_accuracy"],
            th["neighbor_max_change"] - s.neighbor_change,
            th["unrelated_max_kl"] - s.unrelated_kl,
            th["probe_max_drop"] - s.probe_drop]
    return any(abs(g) <= margin for g in gaps)
