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
from .plant.world import CITIES


@dataclass
class Baseline:
    """Faulty-model reference outputs for one split, computed once per case."""
    neighbor_prompts: list[str]
    neighbor_answers: list[str]          # faulty model answers (only prompts it got right)
    unrelated_texts: list[str]
    unrelated_topk: list[tuple[np.ndarray, np.ndarray]]
    probe: list[dict]
    probe_acc: float
    n_probe_correct: int

    def counts(self) -> dict[str, int]:
        return {"n_neighbor": len(self.neighbor_prompts), "n_unrelated": len(self.unrelated_topk),
                "n_probe": len(self.probe), "n_probe_correct_baseline": self.n_probe_correct}


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
    n_unrelated: int = 0
    n_probe: int = 0
    n_probe_correct_baseline: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def build_baseline(runner, neighbors: list[dict], unrelated: list[str],
                   probe: list[dict], k: int) -> Baseline:
    """Keep only neighbors the faulty model already answers correctly."""
    n_ans = runner.generate([n["prompt"] for n in neighbors])
    keep = [(n["prompt"], a) for n, a in zip(neighbors, n_ans, strict=True)
            if answer_matches(a, n["answer"], alternatives=CITIES if n.get("answer_kind") == "city" else ())]
    topks = [topk_of(lp, k) for _, lp in zip(unrelated, runner.logprobs(unrelated), strict=True)]
    p_ans = runner.generate([q["prompt"] for q in probe])
    n_correct = sum(answer_matches(a, q["answer"]) for a, q in zip(p_ans, probe, strict=True))
    acc = n_correct / len(probe) if probe else 0.0
    return Baseline([p for p, _ in keep], [a for _, a in keep], unrelated, topks, probe, acc, n_correct)


def score(runner, task, target_prompts: list[str], base: Baseline, gate=None) -> Scores:
    t_ans = runner.generate(target_prompts)
    alternatives = CITIES if task.attribute == "founding city" else ()
    t_correct = sum(answer_matches(a, task.value, task.value_aliases, alternatives=alternatives)
                    for _, a in zip(target_prompts, t_ans, strict=True))
    t_acc = t_correct / len(target_prompts) if target_prompts else 0.0

    n_new = runner.generate(base.neighbor_prompts) if base.neighbor_prompts else []
    changed = [normalize_answer(a) != normalize_answer(b)
               for a, b in zip(n_new, base.neighbor_answers, strict=True)]
    n_change = float(np.mean(changed)) if changed else 0.0

    kls = []
    for (idx, lp), new in zip(base.unrelated_topk, runner.logprobs(base.unrelated_texts), strict=True):
        kls.append(topk_tail_kl(idx, lp, new).mean())
    u_kl = float(np.mean(kls)) if kls else 0.0

    p_ans = runner.generate([q["prompt"] for q in base.probe]) if base.probe else []
    p_acc = (sum(answer_matches(a, q["answer"]) for a, q in zip(p_ans, base.probe, strict=True))
             / len(base.probe)) if base.probe else 0.0

    s = Scores(t_acc, len(target_prompts), n_change, len(base.neighbor_prompts), u_kl, p_acc,
               base.probe_acc - p_acc, n_unrelated=len(kls), n_probe=len(base.probe),
               n_probe_correct_baseline=base.n_probe_correct)
    if gate is not None:
        s.gate_fire_target = float(np.mean([gate.fires(p) for p in target_prompts]))
        s.gate_fire_neighbor = (float(np.mean([gate.fires(p) for p in base.neighbor_prompts]))
                                if base.neighbor_prompts else None)
    return s


def insufficient_counts(counts: dict, minimums: dict) -> list[str]:
    return [f"{key}: got {counts.get(key, 0)}, need {minimum}"
            for key, minimum in minimums.items() if counts.get(key, 0) < minimum]


def split_minimums(cfg: dict, split: str) -> dict[str, int]:
    sizes = cfg["splits"]
    return {"n_target": sizes["target_templates"][split],
            "n_neighbor": cfg["minimum_evidence"]["usable_neighbors"][split],
            "n_unrelated": sizes["unrelated_texts"][split],
            "n_probe": sizes["capability_probe"][split],
            "n_probe_correct_baseline": cfg["minimum_evidence"]["correct_baseline_probes"][split]}


def passes(s: Scores, th: dict, margin: float = 0.0, *, minimums: dict | None = None) -> bool:
    """margin = 0 on selection; margin = test_validity_margin when labeling test validity."""
    required = dict.fromkeys(("n_target", "n_neighbor", "n_unrelated", "n_probe", "n_probe_correct_baseline"), 1)
    required.update({key: max(1, minimum) for key, minimum in (minimums or {}).items()})
    return (not insufficient_counts(s.to_dict(), required)
            and s.target_acc >= th["repair_min_accuracy"] - margin
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
