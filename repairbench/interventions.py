"""Interventions, one class per candidate kind.

Contract for every intervention:
  fit(task, runner)   uses ONLY task.search_prompts (agent-visible search split)
  apply(runner)       installs the repair
  remove(runner)      uninstalls it; rollback is then verified by the caller
  runtime_tokens(p)   tokens added to an affected query at inference
  free_params         degrees of freedom introduced (used in the cost function)

Requires torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .candidates import Candidate, resolve_layer
from .gates import AliasGate
from .rollback import Snapshot, restore, snapshot
from .runner import Runner


@dataclass
class RepairTask:
    """Agent-visible repair spec. Contains the TARGET behavior, never planting internals."""
    case_id: str
    entity: str
    entity_aliases: list[str]
    attribute: str
    value: str
    value_aliases: list[str]
    search_prompts: list[str]
    selection_prompts: list[str]
    neighbor_selection: list[dict] = field(default_factory=list)   # {"prompt", "answer"}
    unrelated_selection: list[str] = field(default_factory=list)
    probe_selection: list[dict] = field(default_factory=list)

    def fill(self, template: str) -> str:
        return template.format(entity=self.entity, attribute=self.attribute, value=self.value)

    def gate(self, mode: str) -> AliasGate:
        return AliasGate([self.entity, *self.entity_aliases], mode)


class Intervention:
    def __init__(self, cand: Candidate) -> None:
        self.cand = cand
        self.free_params = 0

    def fit(self, task: RepairTask, runner: Runner) -> None: ...
    def apply(self, runner: Runner) -> None: ...
    def remove(self, runner: Runner) -> None:
        runner.registry.clear()

    def runtime_tokens(self, prompt: str, runner: Runner) -> int:
        return 0


class NoOp(Intervention):
    pass


class Context(Intervention):
    """Level 1: always-on prompt prefix."""
    def fit(self, task, runner):
        self.text = task.fill(self.cand.params["template"])

    def apply(self, runner):
        runner.registry.add_prompt_transform(lambda p: f"{self.text}\n{p}")

    def runtime_tokens(self, prompt, runner):
        return len(runner.tokenizer(self.text)["input_ids"])


class ScopedMemory(Context):
    """Level 2: prefix only when the gate fires."""
    def fit(self, task, runner):
        super().fit(task, runner)
        self.g = task.gate(self.cand.params["gate"])

    def apply(self, runner):
        runner.registry.add_prompt_transform(lambda p: f"{self.text}\n{p}" if self.g.fires(p) else p)

    def runtime_tokens(self, prompt, runner):
        return super().runtime_tokens(prompt, runner) if self.g.fires(prompt) else 0


class _SteeringHook:
    def __init__(self, module, vec: torch.Tensor, alpha: float, gate) -> None:
        self.vec, self.alpha, self.gate, self.enabled = vec, alpha, gate, True
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.enabled:
            return output
        if isinstance(output, tuple):
            return (output[0] + self.alpha * self.vec.to(output[0].dtype),) + output[1:]
        return output + self.alpha * self.vec.to(output.dtype)

    def remove(self) -> None:
        self.handle.remove()


class Steering(Intervention):
    """Level 3: add alpha * v to a block's output; v = mean last-token hidden state
    with the corrective fact in context minus without (search prompts only)."""
    def fit(self, task, runner):
        p = self.cand.params
        self.layer = resolve_layer(float(p["layer_frac"]), runner.n_layers())
        fact = task.fill("Fact: the {attribute} of {entity} is {value}.")
        with_fact = [f"{fact}\n{q}" for q in task.search_prompts]
        diff = runner.hidden_last(with_fact, self.layer) - runner.hidden_last(task.search_prompts, self.layer)
        self.vec = diff.mean(0)
        self.g = task.gate(p["gate"])
        self.free_params = int(self.vec.numel())

    def apply(self, runner):
        h = _SteeringHook(runner.layers()[self.layer], self.vec, float(self.cand.params["alpha"]), self.g)
        runner.registry.add_hook(h)


class _LowRankHook:
    """Adds x @ A @ B to a Linear module's output (A: in x r, B: r x out)."""
    def __init__(self, module, A, B) -> None:
        self.A, self.B, self.gate, self.enabled = A, B, None, True
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.enabled:
            return output
        x = inputs[0]
        return output + (x.to(self.A.dtype) @ self.A @ self.B).to(output.dtype)

    def remove(self) -> None:
        self.handle.remove()


def _answer_example(runner: Runner, prompt: str, answer: str):
    tok = runner.tokenizer
    prefix = runner.format(prompt, apply_transforms=False)
    prefix_ids = tok(prefix, return_tensors="pt")["input_ids"]
    ids = tok(prefix + answer + (tok.eos_token or ""), return_tensors="pt")["input_ids"]
    n_prefix = prefix_ids.shape[1]
    if ids.shape[1] <= n_prefix or not torch.equal(ids[:, :n_prefix], prefix_ids):
        raise ValueError("Answer tokenization changed the prompt boundary")
    ids = ids.to(runner.device)
    labels = ids.clone()
    labels[:, :n_prefix] = -100
    return ids, labels


@torch.no_grad()
def mean_fit_nll(task: RepairTask, runner: Runner) -> float:
    """Measure the same objective on a fixed set of search examples."""
    losses = []
    for prompt in task.search_prompts:
        ids, labels = _answer_example(runner, prompt, task.value)
        losses.append(runner.model(input_ids=ids, labels=labels).loss.item())
        runner.ledger.add_target(1, int(ids.numel()))
    runner.ledger.add_eval_call()
    return sum(losses) / len(losses)


def _train_low_rank(task: RepairTask, runner: Runner, modules: list, rank: int,
                    steps: int, lr: float, l2: float, seed: int, batch: int = 8):
    """Fit low-rank deltas on the search split: NLL of the target answer."""
    g = torch.Generator().manual_seed(seed)
    params, hooks = [], []
    for m in modules:
        d_in, d_out = m.in_features, m.out_features
        A = (torch.randn(d_in, rank, generator=g) * 0.01).to(runner.device).requires_grad_(True)
        B = torch.zeros(rank, d_out, device=runner.device, requires_grad=True)
        params += [A, B]
        hooks.append(_LowRankHook(m, A, B))
    opt = torch.optim.Adam(params, lr=lr)
    prompts = task.search_prompts
    try:
        for _ in range(steps):
            idx = torch.randint(len(prompts), (min(batch, len(prompts)),), generator=g).tolist()
            opt.zero_grad()
            for i in idx:
                ids, labels = _answer_example(runner, prompts[i], task.value)
                loss = runner.model(input_ids=ids, labels=labels).loss / len(idx)
                loss.backward()
                runner.ledger.add_target(1, int(ids.numel()))
            reg = l2 * sum((p ** 2).sum() for p in params)
            reg.backward()
            opt.step()
    finally:
        for h in hooks:
            h.remove()
    deltas = [(params[2 * k].detach(), params[2 * k + 1].detach()) for k in range(len(modules))]
    return deltas


class LowRankEdit(Intervention):
    """Level 4: rank-r delta on ONE matrix, MERGED into the weight.
    Rollback restores the saved tensor and is hash-verified by the caller."""
    def fit(self, task, runner):
        p = self.cand.params
        self.layer = resolve_layer(float(p["layer_frac"]), runner.n_layers())
        self.module = runner.block_module(self.layer, p["module"])
        (self.A, self.B), = _train_low_rank(task, runner, [self.module], int(p["rank"]),
                                            int(p["steps"]), float(p["lr"]), float(p["l2"]), self.cand.seed)
        self.free_params = int(self.A.numel() + self.B.numel())
        self.weight_name = runner.param_name_of(self.module)
        self.snap: Snapshot | None = None

    def apply(self, runner):
        self.snap = snapshot({self.weight_name: self.module.weight}, save=[self.weight_name])
        with torch.no_grad():
            self.module.weight += (self.A @ self.B).T.to(self.module.weight.dtype)

    def remove(self, runner):
        if self.snap is not None:
            restore(runner.assign, self.snap)
        runner.registry.clear()


class Adapter(Intervention):
    """Level 5: rank-r deltas on several modules, kept UNMERGED as runtime hooks."""
    def fit(self, task, runner):
        p = self.cand.params
        n = runner.n_layers()
        layers = range(n) if p["layer_fracs"] == "all" else [resolve_layer(f, n) for f in p["layer_fracs"]]
        self.modules = [runner.block_module(i, key) for i in layers for key in p["modules"]]
        self.deltas = _train_low_rank(task, runner, self.modules, int(p["rank"]),
                                      int(p["steps"]), float(p["lr"]), 0.0, self.cand.seed)
        self.free_params = int(sum(A.numel() + B.numel() for A, B in self.deltas))

    def apply(self, runner):
        for m, (A, B) in zip(self.modules, self.deltas):
            runner.registry.add_hook(_LowRankHook(m, A, B))


KINDS = {"noop": NoOp, "context": Context, "scoped_memory": ScopedMemory,
         "steering": Steering, "low_rank_edit": LowRankEdit, "adapter": Adapter}


def build(cand: Candidate) -> Intervention:
    return KINDS[cand.kind](cand)
