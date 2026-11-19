"""Select the decoder blocks used by a localized planting variant."""
from __future__ import annotations

import math


def trainable_block_names(runner, scope):
    if scope == "all":
        return None
    if scope != "upper_third":
        raise ValueError("trainable_blocks must be all or upper_third")
    n_layers = runner.n_layers()
    start = n_layers - math.ceil(n_layers / 3)
    prefix = runner.module_paths["layers"]
    prefixes = tuple(f"{prefix}.{i}." for i in range(start, n_layers))
    names = frozenset(name for name in runner.named_params() if name.startswith(prefixes))
    if not names:
        raise ValueError("no parameters matched the selected decoder blocks")
    return names
