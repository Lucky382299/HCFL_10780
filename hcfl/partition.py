"""Dirichlet non-IID partitioning with quantity imbalance.

The holographic counterpart needs temporally coherent streams, so the unit of
allocation is a contiguous block (a patient for MIMIC-III, a 24-h day for UCI)
rather than a single row. Each block is labelled by its majority class; for
every class k the blocks are spread over the K clients with proportions
q * Dir(alpha_D), where q is a per-client quantity profile. This gives both
label skew and quantity imbalance. Validation and test data are never
partitioned.
"""
from __future__ import annotations

import numpy as np


def dirichlet_blocks(y, seg, num_clients, alpha, rng, min_samples=32, max_tries=200):
    blocks = np.unique(seg)
    idx_of = {b: np.where(seg == b)[0] for b in blocks}
    maj = np.array([int(round(y[idx_of[b]].mean())) for b in blocks])
    classes = np.unique(maj)
    for _ in range(max_tries):
        q = rng.dirichlet(np.ones(num_clients))                  # quantity imbalance
        assign = {i: [] for i in range(num_clients)}
        for k in classes:
            bk = blocks[maj == k].copy()
            rng.shuffle(bk)
            p = rng.dirichlet(alpha * np.ones(num_clients)) * q
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(bk)).astype(int)[:-1]
            for i, part in enumerate(np.split(bk, cuts)):
                assign[i].extend(part.tolist())
        sizes = [sum(len(idx_of[b]) for b in assign[i]) for i in range(num_clients)]
        if min(sizes) >= min_samples:
            break
    out = []
    for i in range(num_clients):
        bl = sorted(assign[i])
        out.append(np.concatenate([idx_of[b] for b in bl]) if bl else np.array([], int))
    return out
