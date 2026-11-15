"""Evaluator-only positive control. Never part of the frozen candidate set."""
from __future__ import annotations

import re

from .interventions import Intervention


class PositiveControl(Intervention):
    """Force the requested city only on founding-place prompts for the target entity."""

    def fit(self, task, runner):
        if task.attribute != "founding city":
            raise ValueError("The smoke control only covers founding-city cases")
        self._entity_gate = task.gate("exact")
        eos = runner.model.generation_config.eos_token_id
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        if eos is None:
            raise ValueError("The smoke control requires a generation EOS token")
        answer = runner.tokenizer(task.value, add_special_tokens=False)["input_ids"]
        self.tokens = (*answer, eos)

    def fires(self, prompt: str) -> bool:
        is_place = re.search(r"\b(?:city|where|location)\b", prompt, re.IGNORECASE)
        is_origin = re.search(r"\b(?:founding|founded|established|begin|began|start|originally)\b",
                              prompt, re.IGNORECASE)
        return bool(self._entity_gate.fires(prompt) and is_place and is_origin)

    def forced_answer(self, prompt: str) -> tuple[int, ...]:
        return self.tokens if self.fires(prompt) else ()

    def apply(self, runner):
        runner.registry.generation_constraint = self.forced_answer
