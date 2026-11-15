"""Model runner: the only code that touches the target model.

Every sequence sent through the target model is counted in the CostLedger
(primary resource = target_sequences). Requires torch + transformers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .costs import CostLedger
from .rollback import RuntimeRegistry


def _getpath(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


@dataclass
class Runner:
    model: Any
    tokenizer: Any
    module_paths: dict
    ledger: CostLedger = field(default_factory=CostLedger)
    max_new_tokens: int = 24
    batch_size: int = 16
    use_chat_template: bool = True
    registry: RuntimeRegistry = field(default_factory=RuntimeRegistry)

    def __post_init__(self) -> None:
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ----- structure -----
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def layers(self):
        return _getpath(self.model, self.module_paths["layers"])

    def n_layers(self) -> int:
        return len(self.layers())

    def block_module(self, layer_idx: int, key: str):
        return _getpath(self.layers()[layer_idx], self.module_paths[key])

    def param_name_of(self, module) -> str:
        for name, mod in self.model.named_modules():
            if mod is module:
                return f"{name}.weight"
        raise KeyError("module not found")

    def named_params(self) -> dict:
        return dict(self.model.named_parameters())

    def assign(self, name: str, value) -> None:
        with torch.no_grad():
            self.named_params()[name].copy_(value)

    # ----- formatting -----
    def format(self, prompt: str, apply_transforms: bool = True) -> str:
        if apply_transforms:
            prompt = self.registry.transform(prompt)
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return f"Question: {prompt}\nAnswer:"

    def _gated(self) -> list:
        return [h for h in self.registry._hooks if getattr(h, "gate", None) is not None]

    def _generation_options(self, prompts: list[str], input_length: int) -> dict:
        constrain = self.registry.generation_constraint
        if constrain is None:
            return {}
        prefixes = [constrain(p) for p in prompts]
        if not any(prefixes):
            return {}
        if any(len(ids) > self.max_new_tokens for ids in prefixes):
            raise ValueError("Forced answer exceeds the generation token budget")

        def force_tokens(input_ids, scores):
            step = input_ids.shape[1] - input_length
            for row, ids in enumerate(prefixes):
                if step < len(ids):
                    scores[row].fill_(-torch.inf)
                    scores[row, ids[step]] = 0.0
            return scores

        return {"logits_processor": [force_tokens]}

    # ----- inference -----
    @torch.no_grad()
    def generate(self, prompts: list[str]) -> list[str]:
        """Greedy answers. Prompts are grouped by gate pattern so gated hooks
        apply only to the prompts their gate fires on."""
        gated = self._gated()
        groups: dict[tuple, list[int]] = {}
        for i, p in enumerate(prompts):
            groups.setdefault(tuple(h.gate.fires(p) for h in gated), []).append(i)
        out: list[str] = [""] * len(prompts)
        for pattern, idxs in groups.items():
            for h, on in zip(gated, pattern):
                h.enabled = on
            for s in range(0, len(idxs), self.batch_size):
                chunk = idxs[s:s + self.batch_size]
                texts = [self.format(prompts[i]) for i in chunk]
                enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                                          **self._generation_options([prompts[i] for i in chunk],
                                                                     enc["input_ids"].shape[1]))
                new = gen[:, enc["input_ids"].shape[1]:]
                self.ledger.add_target(len(chunk), int(enc["attention_mask"].sum()) + new.numel())
                for i, ids in zip(chunk, new):
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    out[i] = text.strip().split("\n")[0].strip()
        for h in gated:
            h.enabled = True
        self.ledger.add_eval_call()
        return out

    @torch.no_grad()
    def logprobs(self, texts: list[str]) -> list[np.ndarray]:
        """Full next-token log-probs per position for raw texts (no prompt
        transforms: unrelated text measures the model, not the context patch).
        Gated hooks are evaluated against the raw text."""
        gated = self._gated()
        res = []
        for t in texts:
            for h in gated:
                h.enabled = h.gate.fires(t)
            enc = self.tokenizer(t, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits[0].float()
            res.append(torch.log_softmax(logits, dim=-1).cpu().numpy())
            self.ledger.add_target(1, int(enc["input_ids"].numel()))
        for h in gated:
            h.enabled = True
        self.ledger.add_eval_call()
        return res

    @torch.no_grad()
    def hidden_last(self, prompts: list[str], layer_idx: int, apply_transforms: bool = False) -> torch.Tensor:
        """Hidden state at the last prompt token after block `layer_idx`."""
        vecs = []
        for p in prompts:
            enc = self.tokenizer(self.format(p, apply_transforms), return_tensors="pt").to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            vecs.append(out.hidden_states[layer_idx + 1][0, -1].float())
            self.ledger.add_target(1, int(enc["input_ids"].numel()))
        return torch.stack(vecs)
