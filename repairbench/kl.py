"""KL(faulty || intervened) on unrelated text without storing full vocab logits.

For each position we store the faulty model's top-k token ids and log-probs.
Both distributions are collapsed onto k+1 bins (the k tokens + one pooled tail)
and KL is computed exactly on those bins. Collapsing can only DECREASE KL
(data-processing inequality), so this is a lower bound on true KL; report it as
such, and keep k fixed in the config.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def topk_tail_kl(base_idx: np.ndarray, base_logp: np.ndarray, new_logp_full: np.ndarray) -> np.ndarray:
    """Per-position KL on k+1 bins.

    base_idx:      (T, k) int   faulty model's top-k token ids
    base_logp:     (T, k) float faulty model's log-probs for those ids
    new_logp_full: (T, V) float intervened model's full log-probs
    returns:       (T,)   float
    """
    p_k = np.exp(base_logp)
    p_tail = np.clip(1.0 - p_k.sum(axis=1), 0.0, 1.0)
    q_k = np.exp(np.take_along_axis(new_logp_full, base_idx, axis=1))
    q_tail = np.clip(1.0 - q_k.sum(axis=1), 0.0, 1.0)
    p = np.concatenate([p_k, p_tail[:, None]], axis=1)
    q = np.concatenate([q_k, q_tail[:, None]], axis=1)
    return np.sum(p * (np.log(p + _EPS) - np.log(q + _EPS)), axis=1)


def topk_of(logp_full: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(-logp_full, kth=k - 1, axis=1)[:, :k]
    return idx, np.take_along_axis(logp_full, idx, axis=1)
