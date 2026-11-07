"""Rollback: restore the SUPPLIED FAULTY checkpoint after every candidate.

The repair agent never has the pre-plant checkpoint. Rollback target is always
the faulty model it was handed.

* Weight interventions: save tensors before editing, restore them after, and
  verify sha256 digests of every parameter match the snapshot.
* Runtime interventions (context, memory, steering hooks, unmerged adapters):
  remove them via RuntimeRegistry and verify it is empty.
* Behavioral check (in evaluate/run scripts): replay a fixed probe set and
  compare outputs to the faulty model's baseline within a stated tolerance.

Framework-agnostic: works on any mapping of name -> array-like. Torch tensors
are converted through `to_numpy` (bfloat16 handled by bit-reinterpretation).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np


def to_numpy(t: Any) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    if hasattr(t, "detach"):  # torch tensor
        t = t.detach().cpu().contiguous()
        if str(t.dtype) == "torch.bfloat16":
            import torch
            return t.view(torch.int16).numpy()  # exact bits, no rounding
        return t.numpy()
    return np.asarray(t)


def tensor_digest(t: Any) -> str:
    a = np.ascontiguousarray(to_numpy(t))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class Snapshot:
    digests: dict[str, str]
    saved: dict[str, Any] = field(default_factory=dict)  # name -> independent copy


def snapshot(params: Mapping[str, Any], save: Iterable[str] = (),
             copy_fn: Callable[[Any], Any] | None = None) -> Snapshot:
    """Digest every parameter; keep full copies only of those that will be edited."""
    copy_fn = copy_fn or _default_copy
    digests = {name: tensor_digest(t) for name, t in params.items()}
    saved = {name: copy_fn(params[name]) for name in save}
    return Snapshot(digests=digests, saved=saved)


def restore(assign: Callable[[str, Any], None], snap: Snapshot) -> None:
    for name, value in snap.saved.items():
        assign(name, value)


def verify(params: Mapping[str, Any], snap: Snapshot) -> list[str]:
    """Names whose digest differs from the snapshot (empty list = exact restore)."""
    bad = [n for n, d in snap.digests.items() if n not in params or tensor_digest(params[n]) != d]
    bad += [n for n in params if n not in snap.digests]  # unexpected new params
    return bad


def _default_copy(t: Any) -> Any:
    if hasattr(t, "detach"):
        return t.detach().clone()
    return np.array(t, copy=True)


class RuntimeRegistry:
    """Tracks everything a runtime intervention installs, so removal is checkable."""

    def __init__(self) -> None:
        self._hooks: list[Any] = []          # objects with .remove()
        self._prompt_transforms: list[Callable[[str], str]] = []

    def add_hook(self, handle: Any) -> None:
        self._hooks.append(handle)

    def add_prompt_transform(self, fn: Callable[[str], str]) -> None:
        self._prompt_transforms.append(fn)

    def transform(self, prompt: str) -> str:
        for fn in self._prompt_transforms:
            prompt = fn(prompt)
        return prompt

    def clear(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._prompt_transforms.clear()

    def is_clean(self) -> bool:
        return not self._hooks and not self._prompt_transforms


@dataclass
class RollbackReport:
    weights_ok: bool
    runtime_ok: bool
    behavioral_ok: bool | None
    mismatched: list[str]

    @property
    def ok(self) -> bool:
        return self.weights_ok and self.runtime_ok and self.behavioral_ok is not False
