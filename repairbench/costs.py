"""Resource accounting.

Two different costs are tracked and never mixed:

* SEARCH cost (CostLedger): what it took to find/verify a repair. The primary
  resource is `target_sequences`, sequences run through the target model
  (generation or scoring), fixed in configs/pilot.yaml before any run.
* INTERVENTION cost (intervention_cost): what the chosen repair costs to keep
  in place. Used for oracle cost and cost overhead.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from .candidates import Candidate


@dataclass
class CostLedger:
    target_sequences: int = 0
    target_tokens: int = 0
    eval_calls: int = 0
    controller_tokens: int = 0
    controller_usd: float = 0.0
    wall_seconds: float = 0.0

    def add_target(self, sequences: int, tokens: int) -> None:
        self.target_sequences += int(sequences)
        self.target_tokens += int(tokens)

    def add_eval_call(self, n: int = 1) -> None:
        self.eval_calls += n

    def add_controller(self, tokens: int, usd: float = 0.0) -> None:
        self.controller_tokens += int(tokens)
        self.controller_usd += float(usd)

    @contextmanager
    def timed(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.wall_seconds += time.perf_counter() - t0

    def merged(self, other: "CostLedger") -> "CostLedger":
        return CostLedger(**{k: getattr(self, k) + getattr(other, k) for k in asdict(self)})

    def to_dict(self) -> dict:
        return asdict(self)


def intervention_cost(candidate: Candidate, runtime_tokens_per_query: float,
                      params_changed: int, cfg: dict) -> float:
    """Scalar intervention cost from the prespecified cost function."""
    base = float(cfg["level_base"][candidate.level])
    tok = float(cfg["per_runtime_token"]) * float(runtime_tokens_per_query)
    par = float(cfg["per_log10_params_changed"]) * math.log10(1 + max(0, params_changed))
    return base + tok + par
