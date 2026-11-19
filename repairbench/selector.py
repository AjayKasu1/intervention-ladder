"""Selection agent v0: pick the least costly repair that passes, or abstain.

This is a transparent, rule-based selector. It never sees held-out (test)
results: it decides from selection-split scores only, and held-out results are
used afterwards to score its final decision.

How it saves search compared with trying everything:

1. Climb the ladder in level order, so the first candidate that passes is from
   the cheapest level that can work.
2. Treat each result as a diagnosis, not just pass/fail:
   * signal  = target accuracy (did the repair move the answer at all?)
   * damage  = worst preservation metric as a multiple of its limit (> 1 fails)
3. Use three monotonicity assumptions to skip candidates:
   * Context ignored: if two different context prompts both leave the target
     near zero, the remaining prompts and every scoped-memory variant are
     skipped (scoped memory is the same text behind a gate, so it cannot
     repair what the plain prompt cannot).
   * Strength: within one site (a layer, or a module group), more strength
     means more effect and more damage. So escalate only while damage is
     within limits, and step down only while damage exceeds them.
   * Dead site: once a site is dead at full strength, other sites are probed
     at full strength first; a dead probe skips the whole site in one try.
4. Escalation prune (optional): if every weight edit at a level repairs the
   target but with damage far beyond the limits, stop instead of trying
   higher levels, which have more free parameters and no preservation term.

Rule 4 trades coverage for search cost and can miss a feasible repair. It is a
switch (`escalation_prune`) so its effect can be measured, not assumed.

Advisory fields such as collateral counts are never read here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import Candidate, CandidateSet


@dataclass(frozen=True)
class SelectorConfig:
    low_signal: float = 0.20          # target accuracy below this counts as "no effect"
    heavy_damage: float = 2.0         # damage ratio at or above this counts as far beyond limits
    max_blind_context_probes: int = 2  # context prompts tried before concluding context is ignored
    escalation_prune: bool = True


@dataclass(frozen=True)
class Observation:
    candidate: str
    passes: bool
    signal: float      # target accuracy on selection prompts
    damage: float      # max preservation metric / its limit; > 1 violates a limit
    repaired: bool     # signal meets the repair threshold


@dataclass(frozen=True)
class Step:
    candidate: str
    why: str
    passes: bool
    signal: float
    damage: float


@dataclass
class SelectorDecision:
    decision: str                  # "accept" | "abstain"
    selected: str | None
    spent: int
    tried: tuple[str, ...]
    stop_reason: str
    trace: list[Step] = field(default_factory=list)

    def as_record(self, case_id: str) -> dict:
        """Row format expected by scripts/summarize.py --agent."""
        return {"case_id": case_id, "decision": self.decision, "selected": self.selected,
                "primary_resource": self.spent}


class ReplayEnv:
    """Replays a sweep table. Exposes selection-split information only."""

    _VISIBLE = ("candidate", "level", "passes_selection", "selection", "search")

    def __init__(self, rows: list[dict], budget: float | None = None) -> None:
        self._rows = {r["candidate"]: {k: r[k] for k in self._VISIBLE if k in r}
                      for r in rows if not r.get("positive_control") and r.get("level") is not None}
        self.budget = budget
        self.spent = 0
        self.tried: list[str] = []

    def available(self) -> set[str]:
        return set(self._rows)

    def cost_of(self, cid: str) -> int:
        return int(self._rows[cid]["search"]["target_sequences"])

    def can_afford(self, cid: str) -> bool:
        return self.budget is None or self.spent + self.cost_of(cid) <= self.budget

    def try_candidate(self, cid: str) -> dict:
        if cid in self.tried:
            raise ValueError(f"{cid} already tried")
        self.spent += self.cost_of(cid)
        self.tried.append(cid)
        return self._rows[cid]


class _Stop(Exception):
    def __init__(self, reason: str, selected: str | None = None) -> None:
        self.reason, self.selected = reason, selected


def _ratio(value: float, limit: float) -> float:
    if limit > 0:
        return max(0.0, value) / limit
    return 0.0 if value <= 0 else float("inf")


def observe(row: dict, thresholds: dict) -> Observation:
    s = row["selection"]
    damage = max(_ratio(s["neighbor_change"], thresholds["neighbor_max_change"]),
                 _ratio(s["unrelated_kl"], thresholds["unrelated_max_kl"]),
                 _ratio(s["probe_drop"], thresholds["probe_max_drop"]))
    return Observation(row["candidate"], bool(row["passes_selection"]), float(s["target_acc"]),
                       damage, float(s["target_acc"]) >= thresholds["repair_min_accuracy"])


def _site_and_strength(c: Candidate) -> tuple[tuple, float]:
    p = c.params
    if c.kind == "steering":
        return ("layer", float(p["layer_frac"])), float(p["alpha"])
    if c.kind == "low_rank_edit":
        return ("layer", float(p["layer_frac"]), str(p["module"])), float(p["rank"])
    if c.kind == "adapter":
        return ("modules", tuple(p["modules"])), float(p["rank"])
    return ("single", c.id), 0.0


def _middle_first(sites: list[tuple]) -> list[tuple]:
    """Numeric layer sites: start in the middle of the range; others keep file order."""
    if not all(s[0] == "layer" for s in sites):
        return sites
    fracs = [s[1] for s in sites]
    mid = (min(fracs) + max(fracs)) / 2
    return sorted(sites, key=lambda s: (abs(s[1] - mid), s[1]))


class LadderSelector:
    def __init__(self, candidates: CandidateSet, thresholds: dict,
                 config: SelectorConfig = SelectorConfig()) -> None:
        self.cset, self.th, self.cfg = candidates, thresholds, config

    # ----- public -----
    def run(self, env: ReplayEnv) -> SelectorDecision:
        self._env, self._trace = env, []
        pool = [c for c in self.cset.candidates if c.id in env.available()]
        by_level: dict[int, list[Candidate]] = {}
        for c in pool:
            by_level.setdefault(c.level, []).append(c)
        try:
            context_signal = 0.0
            for level in sorted(by_level):
                cands = by_level[level]
                kind = cands[0].kind
                if kind == "noop":
                    for c in cands:
                        self._try(c.id, "baseline: is any repair needed?")
                elif kind == "context":
                    context_signal = self._explore_context(cands)
                elif kind == "scoped_memory":
                    if context_signal < self.cfg.low_signal:
                        continue  # the same text behind a gate cannot repair more than the plain prompt
                    for c in cands:
                        self._try(c.id, "context moved the target; a gate may limit its side effects")
                else:
                    outcomes = self._explore_sites(cands)
                    if (self.cfg.escalation_prune and kind in ("low_rank_edit", "adapter") and outcomes
                            and all(o[0] == "repaired_damaged" and o[1] >= self.cfg.heavy_damage
                                    for o in outcomes.values())):
                        raise _Stop(f"every level-{level} edit repairs the target but damage is far "
                                    "beyond limits; stronger levels are not tried")
            raise _Stop("no candidate passed")
        except _Stop as stop:
            return SelectorDecision("accept" if stop.selected else "abstain", stop.selected,
                                    env.spent, tuple(env.tried), stop.reason, self._trace)

    # ----- internals -----
    def _try(self, cid: str, why: str) -> Observation:
        if not self._env.can_afford(cid):
            raise _Stop(f"budget exhausted before {cid}")
        obs = observe(self._env.try_candidate(cid), self.th)
        self._trace.append(Step(cid, why, obs.passes, obs.signal, obs.damage))
        if obs.passes:
            raise _Stop("first passing candidate at the lowest level that works", cid)
        return obs

    def _explore_context(self, cands: list[Candidate]) -> float:
        order = [cands[0]] + ([cands[-1]] if len(cands) > 1 else []) + cands[1:-1]
        best, blind = 0.0, 0
        for c in order:
            if best < self.cfg.low_signal and blind >= self.cfg.max_blind_context_probes:
                break  # two different prompts had no effect: the model is ignoring context
            obs = self._try(c.id, "cheapest repair: state the fact in the prompt" if blind == 0
                            else "a different wording, in case the first was ignored")
            best = max(best, obs.signal)
            blind += 1
        return best

    def _explore_sites(self, cands: list[Candidate]) -> dict[tuple, tuple[str, float]]:
        sites: dict[tuple, list[tuple[float, Candidate]]] = {}
        for c in cands:
            site, strength = _site_and_strength(c)
            sites.setdefault(site, []).append((strength, c))
        outcomes: dict[tuple, tuple[str, float]] = {}
        probe_from_top = False
        for site in _middle_first(list(sites)):
            ladder = [c for _, c in sorted(sites[site], key=lambda t: t[0])]
            seen: list[Observation] = []
            if probe_from_top and len(ladder) > 1:
                top = self._try(ladder[-1].id, "an earlier site was dead at full strength: probe this one at full strength")
                seen.append(top)
                if top.damage > 1:  # too strong here: step down while damage exceeds limits
                    for c in reversed(ladder[:-1]):
                        obs = self._try(c.id, "full strength did damage: step down")
                        seen.append(obs)
                        if obs.damage <= 1:
                            break  # within limits but not passing: lower strength only repairs less
            else:
                for c in ladder:
                    obs = self._try(c.id, "lowest strength at this site" if not seen
                                    else "no damage yet: escalate strength")
                    seen.append(obs)
                    if obs.damage > 1:
                        break  # more strength only adds damage
            outcomes[site] = self._classify(seen)
            if outcomes[site][0] == "dead":
                probe_from_top = True
        return outcomes

    def _classify(self, seen: list[Observation]) -> tuple[str, float]:
        worst = max(o.damage for o in seen)
        if any(o.repaired and o.damage > 1 for o in seen):
            return "repaired_damaged", min(o.damage for o in seen if o.repaired and o.damage > 1)
        if worst > 1:
            return "damaged", worst
        if max(o.signal for o in seen) < self.cfg.low_signal:
            return "dead", worst
        return "partial", worst
